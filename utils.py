import torch
import cubo
import xarray as xr
import planetary_computer as pc
import numpy as np
import rasterio as rio
import rioxarray
from rasterio import Affine

import geopandas as gpd
import ee
import shapely
import pyproj
from typing import Callable, Dict, List, Tuple, Union, Optional
import shutil
import json
import os

import tqdm
import threading

import pathlib
from scipy.ndimage import distance_transform_edt


def align_points_to_grid(points: gpd.GeoDataFrame, 
                         resolution: int = 10,
                         roi: str = "id"
) -> List[shapely.Point]:
    """
    Aligns the points to the upper left of pixel in the image.

    Args:
        points (gpd.GeoDataFrame): A GeoDataFrame containing the points to be aligned.
        resolution (int): The resolution of the image. Defaults to 20.
        roi (str): The name of the column containing the Region of Interest (ROI) names. 
        Defaults to "id".

    Returns:
        List[shapely.Point]: A list of shapely points aligned to the grid.
    """
    # Empty list
    latlon_coords = []

    # Iterate over points
    for point in tqdm.tqdm(points.to_dict("records")):

        # Get latlon
        geom = point["geometry"]
        lat, lon = geom.coords[0]

        # Initialize ee
        ee.Initialize(project="ee-jfloresf")
        
        # Define a ee.Geometry.Point
        coords = ee.Geometry.Point(lat, lon)

        # Define a ee.ImageCollection
        collection = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(coords).first()

        # Get metadata of image collection
        proj_metadata = collection.select("B11").projection().getInfo()

        # Get crs and transform
        proj_crs = proj_metadata.get("crs")

        proj_transform = proj_metadata.get("transform")

        # Initialize pyproj transformer
        transformer = pyproj.Transformer.from_crs(f"EPSG:4326", proj_crs, always_xy=True)

        # Transform latlon to UTM
        utm_coords = transformer.transform(lat, lon)

        # Get x and y
        x = utm_coords[0]
        y = utm_coords[1]

        # Get x and y in the upper left of the pixel aligned to the grid
        x_image = proj_transform[2] + round((x - proj_transform[2]) / resolution) * resolution \
        + resolution/2

        y_image = proj_transform[5] + round((y - proj_transform[5]) / resolution) * resolution \
        + resolution/2

        # Convert to latlon
        inverse_transformer = pyproj.Transformer.from_crs(
            proj_crs, "EPSG:4326", always_xy=True
        )

        latlon_coord = inverse_transformer.transform(x_image, y_image)

        # Append to latlon list
        latlon_coords.append(shapely.Point(latlon_coord))

    # Save as geodataframe
    gdf = gpd.GeoDataFrame(data={"id": points[roi]},
                            geometry=latlon_coords, crs="EPSG:4326")

    return gdf


def timeout_handler(seconds):
    raise TimeoutError(f"Timed out. {seconds} seconds elapsed.")


def time_limit(seconds: int, func: Callable, *args, **kwargs):
    '''
    Runs a function with a time limit. If the function does not return within the 
    time limit, a TimeoutError is raised.
    
    Args:
        seconds (int): The time limit in seconds.
        func (Callable): The function to be run.
        *args: The arguments to be passed to the function.
        **kwargs: The keyword arguments to be passed to the function.
    
    Returns:
        The result of the function or None if the function timed out.
    '''
    timer = threading.Timer(seconds, timeout_handler, args=[seconds])
    timer.start()

    try:
        result = func(*args, **kwargs)
        return result 
    finally:
        timer.cancel()


def create_datacube(x: float, y: float, 
                    start_date: str, 
                    end_date: str,
                    s2_bands: List[str], 
                    tries: int = 5
) -> xr.DataArray:
    '''
    Retrieve datacube from the STAC API

    Args:
        x (float): The longitude of the point
        y (float): The latitude of the point
        start_date (str): The start date of the datacube. The format must be "YYYY-MM-DD"
        end_date (str): The end date of the datacube. The format must be "YYYY-MM-DD"
        s2_bands (List[str]): The bands to be included in the cube
        tries (int, optional): The number of times to retry. Defaults to 5.
    
    Returns:
        xr.DataArray: The datacube.
    '''
    # Planetary Computer API token
    my_token = "756c105af5a848c0b22e55969e62ab21"
    pc.settings.set_subscription_key(my_token)

    while tries > 0:
        try:
            da = cubo.create(
                lat=y,  # Central latitude of the cube
                lon=x,  # Central longitude of the cube
                collection="sentinel-2-l2a",  # Name of the STAC collection
                bands=s2_bands,  # Bands to be included in the cube
                start_date=start_date,  # Start date of the cube
                end_date=end_date,  # End date of the cube
                edge_size=512,  # Edge size of the cube (px)
                resolution=10,  # Pixel size of the cube (m)
            )
            da = da.drop_duplicates("time")

            # sort by date
            da = da.sortby("time")
            return da
        except:
            tries -= 1

    if tries == 0:
        raise Exception(f"Failed to create datacube for {x}, {y}")

    return None


def get_datacube_data(da: xr.DataArray, time: int, 
                      start_date: str,
                      end_date: str,
                      s2_bands:List[str], 
                      tries: int = 5
) -> np.ndarray:
    '''
    Retrieve datacube data.

    Args:
        da (xr.DataArray): The datacube.
        time (int): The time index of the datacube.
        start_date (str): The start date of the datacube. The format must be "YYYY-MM-DD"
        end_date (str): The end date of the datacube. The format must be "YYYY-MM-DD"
        tries (int): The number of times to retry. Defaults to 5.
    
    Returns:
        np.ndarray: The datacube data.
    '''
    while tries > 0:
        try:
            def download_data():
                data = da.isel(time=time)
                id = da.id[time].item()
                data_np = data.to_numpy()
                return data, data_np, id
            result = time_limit(60, download_data)
            return result
        except Exception as e:
            print(f"Failed to download datacube data {e}. Retrying...")
            da = create_datacube(da.attrs["central_lon"], da.attrs["central_lat"],
                                 start_date, end_date, s2_bands, tries=5)
            tries -= 1

    if tries == 0:
        raise Exception(f"Failed to download datacube layer")


def cloud_mask(data: np.ndarray, model: torch.jit.ScriptModule) -> np.ndarray:
    """
    Apply a cloud mask to the data using a PyTorch model.

    Args:
        data (np.ndarray): The data to be masked.
        model (torch.jit.ScriptModule): The PyTorch model to be used for masking.

    Returns:
        np.ndarray: The cloud mask.
    """
    data_torch = torch.from_numpy(data)[None] / 10000
    cloud_model = model(data_torch.float())
    cloud_mask = cloud_model.argmax(dim=1).squeeze().numpy()

    return cloud_mask


def download_datacube_data(roi: gpd.GeoSeries, model: torch.jit.ScriptModule,
                           start_date: str, end_date: str,
                           s2_bands: List[str], out_path: pathlib.Path
) -> None:
    """
    Download the datacube data for a given Region of Interest (ROI).

    Args:
        roi (gpd.GeoSeries): The Region of Interest (ROI).
        model (torch.jit.ScriptModule): The PyTorch model to be used for cloud masking.
        s2_bands (List[str]): The Sentinel-2 bands to be downloaded.
        out_path (pathlib.Path): The output path for the downloaded data.
    
    Returns:
        None
    """
    try:
        # Create folder for the point
        folder_name = out_path / roi.id
        folder_name.mkdir(exist_ok=True)

        # Download the cube metadata
        geom = roi.geometry
        da = create_datacube(geom.x, geom.y, start_date, end_date, s2_bands)
        
        # Retake the download
        ## get the image with the highest date
        creation_time = {x.stem: os.path.getctime(x) for x in folder_name.glob("*.tif")}

        if len(creation_time) != 0:
            position = int(np.where(np.isin(da.id.values, max(creation_time)))[0])
        else:
            position = 0        
        
        # Download the cube data
        for i, date in enumerate(da.time):
            if (position - 1) > i:
                continue
            print(f"Downloading {i+1}/{len(da.time)}")            
            # get the S2 image id
            data, data_np, id = get_datacube_data(da, i, start_date, end_date, s2_bands)
            final_name = folder_name / f"{id}.tif"

            # apply the cloud model
            mask = cloud_mask(data_np, model)

            # if the cloud cover is greater than 10%, skip the image
            if (np.sum(mask > 0) / mask.size) > 0.1:
                print(f"Skipping {id} due to cloud cover")
                continue

            # Save the image as a GeoTIFF file
            crs = da.epsg.item()
            data_sp = data.rio.write_crs("epsg:{}".format(crs))
            data_sp.rio.to_raster(
                final_name,
                dtype="uint16",
                compress="lzw",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
        print(f"Downloaded ROI: {roi.id} successfully")

    except Exception as e:
        print(e)
        pass   
    

def omit_black_files(tif_files: List[pathlib.Path],
                     model: torch.jit.ScriptModule
) -> List[pathlib.Path]:
    """
    Omit files with black areas with a cloud percentage greater than 10% in a list 
    of tif files.

    Args:
        tif_files (List[pathlib.Path]): List of tif files.
        model (torch.jit.ScriptModule): The PyTorch model to be used for cloud masking.

    Returns:
        List[pathlib.Path]: List of tif files without black areas.
    """
    notblack_files = []
    for i, tif_file in enumerate(tif_files):
        with rio.open(tif_file) as src:
            # Read the data and calculate the cloud mask
            data = src.read().astype(np.int16)
            mask = cloud_mask(data, model)

            # Calculate the percentage of cloud cover
            cloud_perc = np.sum(mask > 0) / mask.size

            # Sum the values of all bands for each pixel
            sum_arr = np.nansum(data, axis=0)
            
            # Check if there are any black pixels or if the cloud cover is more than 10%
            if np.all(sum_arr != 0) and cloud_perc < 0.1:             
                notblack_files.append(tif_file)
                print(f"Not found black areas in image with \
                     {cloud_perc*100:.2f}% of cloud percentage")

        print(f"[{i+1}/{len(tif_files)}]:Processed {tif_file}")
    return notblack_files


def read_bands(src: rio.DatasetReader, band_list: List[int]) -> List[np.ndarray]:
    '''
    Read bands from a raster file.

    Args:
        src (rio.DatasetReader): The raster file.
        band_list (List[int]): The bands to be read.
    
    Returns:
        List[np.ndarray]: The bands.
    '''
    bands = [src.read(band) * 0.0001 for band in band_list]
    return bands


def normalize_band(array: np.ndarray) -> np.ndarray:
     '''
     Normalize a band.

    Args:
        array (np.ndarray): The band array.
    
    Returns:
        np.ndarray: The normalized band.
     '''
     array_max = np.nanmax(array)
     array_min = np.nanmin(array)
     normalize_array = ((array - array_min) / (array_max - array_min))
     return normalize_array


def get_rgb(red: np.ndarray, 
            green: np.ndarray, 
            blue: np.ndarray, 
            factor: Union[int, float]
) -> np.ndarray:
    '''
    Create RGB image with a custom contrast factor.

    Args:
        red (np.ndarray): The red band.
        green (np.ndarray): The green band.
        blue (np.ndarray): The blue band.
        factor (Union[int, float]): The contrast factor.
    
    Returns:
        np.ndarray: The RGB image.
    '''
    nred, ngreen, nblue = [normalize_band(channel) for channel in [red, green, blue]]
    
    # Stack the RGB bands
    rgb = np.dstack((nred, ngreen, nblue))

    # Apply the contrast factor
    rgb_contrast = np.clip(rgb * factor, 0, 255)
    
    return rgb_contrast

        
def rio_calculate_indexs(src: rio.DatasetReader, index: str="nbr") -> np.ndarray:
    '''
    Calculates Indexes of a Sentinel-2 image.

    Args:
        src (rio.DatasetReader): The Sentinel-2 image.
        index (str): The index to be calculated. Defaults to "nbr". 
        Other options are "ndvi","ndwi" and "badi".
    Returns:
        np.ndarray: The indexes of the Sentinel-2 image.
    '''
    bands = [3, 4, 5, 6, 7, 8, 9, 11, 12]

    b3, b4, b5, b6, b7, b8, b8a, b11, b12 = read_bands(src, bands)

    if index == "nbr":
        # Calculate Index
        calc_index = (b8 - b12) / (b8 + b12)

    elif index == "ndvi":
        # Calculate Index
        calc_index = (b8 - b4) / (b8 + b4)
    
    elif index == "ndwi":
        # Calculate Index
        calc_index = (b3 - b8) / (b3 + b8)

    elif index == "badi":
        # Assign bands to descriptive variable names       
        f1 = ((b12 + b11) - (b8 + b8a)) / np.sqrt((b12 + b11) + (b8 + b8a))
        f2 = 2 - np.sqrt((b6 * b7 * (b8 + b8a)) / (b4 + b5))
        
        # Calculate Index        
        calc_index = f1 * f2
    
    return calc_index


def create_burned_mask( src_tuple: Union[rio.DatasetReader, Tuple[rio.DatasetReader]],
                        type: str="mono",
                        index: str="badi",
                        mode: str= "less",
                        threshold: float=0.3
) -> np.ndarray:
    """
    Create a burned area mask.

    Args:
        src_tuple (Union[rio.DatasetReader, Tuple[rio.DatasetReader]]): The source image or 
        a tuple of source images.
        index (str, optional): The index to be calculated. Defaults to "dbadi".
        mode (str, optional): The mode to be applied. Defaults to "less".
        threshold (float, optional): The threshold value. Defaults to 0.3.

    Returns:
        np.ndarray: _description_
    """
        
    if isinstance(src_tuple, tuple) and type == "diff":
        ref = src_tuple[0]
        src = src_tuple[1]

        # Apply difference index
        pre = rio_calculate_indexs(ref, index)
        post = rio_calculate_indexs(src, index)
        mask = pre - post

    elif isinstance(src_tuple, rio.DatasetReader) and type == "mono":
        src = src_tuple
        # Apply selected index
        mask = rio_calculate_indexs(src, index)

    # Apply threshold
    if mode == "greater":
        mask[mask <= threshold] = np.nan
        mask[mask > threshold] = 1
        
    elif mode == "less":
        mask[mask > threshold] = np.nan
        mask[mask <= threshold] = 1

    # Apply NDVI mask
    ndvi = rio_calculate_indexs(src, index="ndvi")
    ndvi[ndvi > 0.2] = np.nan
    ndvi[ndvi <= 0.2] = 1

    # Apply NDWI mask
    ndwi = rio_calculate_indexs(src, index="ndwi")
    ndwi[ndwi > 0] = np.nan
    ndwi[ndwi <= 0] = 1

    # Apply the masks to BADI
    burn_masked = mask * ndvi * ndwi

    return burn_masked

def dbadi_filter(roi_groups: Dict[str, List[pathlib.Path]]) -> List[pathlib.Path]:
    """ 
    Filter the images with burning (+ 1 ha) using the dbadi index.

    Args:
        roi_groups (Dict[str, List[pathlib.Path]]): The dictionary with the ROIs and the list of 
        Sentinel-2 images ordered by date.  
    
    Returns:
        List[pathlib.Path]: The list of images with burning.
    """
    # Iterate over the ROIs
    burning_files = []
    for roi, files in roi_groups.items():
        for i in range(len(files) - 1):  # -1 because we compare with the next file
            with rio.open(files[i]) as src1, rio.open(files[i+1]) as src2:
                # Check if the images have the same shape
                dbadi_masked = create_burned_mask((src1, src2), type="diff", 
                                                index="badi", mode="less", 
                                                threshold=-0.4) 
                nbr_masked = create_burned_mask(src2, type="mono",
                                                index="nbr", mode="less", 
                                                threshold=-0.15) 
                mask = dbadi_masked * nbr_masked

                # Evaluate if there is burning
                if np.nansum(mask) >= 100:  # Assuming 1 pixel is 0.01 ha
                    burning_files.append([files[i+1], np.nansum(mask)])       
                    print(f"[{i+1}/{len(files)}]: Burning in {files[i+1]}")
                else:
                    print(f"[{i+1}/{len(files)}]: No burning in {files[i+1]}")
        
        print(f"Processed ROI: {roi}")


def create_composite(roi_groups: Dict[str, List[str]], out_path: pathlib.Path) -> None:
    """ 
    Create a composite image for each ROI.

    Args:
        -roi_groups (Dict[str, List[str]]): The dictionary with the ROIs and the list of 
        Sentinel-2 images ordered by date.
        -out_path (pathlib.Path): The output path for the composite images.
    
    Returns:
        None
    """ 
    for n, roi_name in enumerate(roi_groups.keys()):
        files = roi_groups[roi_name]
        # Choice 25 aleatory images
        files = np.random.choice(files, 25)
        # Stack the images in a list
        composite = []
        for file in files:
            with rio.open(file) as src:
                profile = src.profile
                composite.append(src.read())

        # Get the median of composite
        composite = np.stack(composite, axis=0)
        median = np.nanmedian(composite, axis=0) 
        # Save the composite
        folder = out_path / roi_name
        folder.mkdir(exist_ok=True)

        with rio.open(folder/f"{roi_name}.tif", "w", **profile) as dst:
            dst.write(median)
        
        # Print the process
        print(f"[{n+1}/{len(roi_groups.keys())}]: Processed {roi_name}")


def create_iris_dir(df: gpd.GeoDataFrame,
                    dir_path: pathlib.Path = pathlib.Path("D:/iris"),
                    source_path: pathlib.Path = pathlib.Path("D:/scburning")
) -> None:
    '''
    Create the IRIS directory structure.

    Args:
        csv_path (str): The path to the CSV file with pats to the images. The column
        "file" must contain the path to the images.
        dir_path (pathlib.Path): The path to the IRIS directory to be created.
        source_path (pathlib.Path): The path to the source directory where the images 
        are located.
    '''

    # Create the IRIS directory
    for i, row in df.iterrows():
        name = pathlib.Path(row["file"]).stem
        roi = pathlib.Path(row["file"]).parent.stem
        dir_folder = dir_path / f"{roi}__{name}"
        dir_folder.mkdir(exist_ok=True, parents=True)

        # Create the images folder
        img_folder = dir_path / "images"
        img_folder.mkdir(exist_ok=True, parents=True)

        # Copy the files
        source_file = pathlib.Path(row["file"])
        shutil.copy2(source_file, dir_folder / "S2.tif")

        # Copy the reference file
        reference_folder = pathlib.Path(f"{source_path}/reference")
        reference_file = reference_folder /  roi / f"{roi}.tif"
        shutil.copy2(reference_file, dir_folder / "S2_ref.tif")

        # Create the label file
        with rio.open(dir_folder / "S2.tif") as src:
            profile = src.profile

            # Create metadata as json
            empty_dict = {}
            empty_dict["spacecraft_id"] = "Sentinel2"
            empty_dict["scene_id"] = f"{roi}__{name}"
            
            # Get the centroid of the raster
            x, y = src.xy(src.height//2, src.width//2)

            # Convert to EPSG:4326 using pyproj
            transformer = pyproj.Transformer.from_crs("epsg:32717", "epsg:4326", 
                                                      always_xy=True)
            lon, lat = transformer.transform(x, y)
            empty_dict["location"] = [lat, lon]
            empty_dict["resolution"] = 10

            # Save the metadata as json
            with open(dir_folder / "metadata.json", "w") as f:
                json.dump(empty_dict, f)

            badi_masked = create_burned_mask(src, "mono", "badi", 
                                             "greater", 0.3)

            # Create a new RGB image with the same shape as badi_masked
            badi_masked_rgb = np.zeros((*badi_masked.shape, 3), dtype=np.uint8)

            # Set the RGB values for badi_masked == 1
            badi_masked_rgb[badi_masked == 1] = [245, 39, 39]

            # Set the RGB values for badi_masked != 1
            badi_masked_rgb[badi_masked != 1] = [255, 255, 255]

            # Save the label file
            profile.update(count=3,  # Change count to 3 for RGB image
                        width=512, 
                        height=512,
                        dtype=np.uint8
                        )
            
            img_folder2 = img_folder / f"{roi}__{name}"
            img_folder2.mkdir(exist_ok=True, parents=True)

            # Convert to RGB png        
            with rio.open(img_folder2/"mask.png", "w", **profile) as dst:
                dst.write(badi_masked_rgb.transpose((2, 0, 1)), indexes=[1, 2, 3])  
        
        # Create the segmentation directory
        seg_folder = dir_path / "sugarcane-burn.iris" / "segmentation"
        seg_folder.mkdir(exist_ok=True, parents=True)

        seg_folder2 = seg_folder / f"{roi}__{name}"
        seg_folder2.mkdir(exist_ok=True, parents=True)

        mask_folder = img_folder / f"{roi}__{name}"
        mask_folder.mkdir(exist_ok=True, parents=True)

        # Convert the image to npy format
        with rio.open(mask_folder / "mask.png") as src:
            data = src.read()

            # Prepare boolean arrays for each color
            is_white = (data[0] == 255) & (data[1] == 255) & (data[2] == 255) # -99
            is_red = (data[0] == 245) & (data[1] == 39) & (data[2] == 39) # 1
            is_purple = (data[0] == 218) & (data[1] == 20) & (data[2] == 255) # 0

            # Create a new array with 3 channels
            arr3d = np.stack([is_white, is_red, is_purple], axis=-1)

            # Array for red vs non-red (purple and white are false, red is true)
            arr = is_red

            # Save the final arrays
            np.save(seg_folder2 / "1_final.npy", arr3d)
            np.save(seg_folder2 / "1_user.npy", arr)
            
        print(f"[{i+1}/{len(df)}]: Processed {row['file']}")


def IRISsave(in_folder: str, out_folder: str, images: str) -> bool:
    """ Create the input and target folders for the IRIS project

    Args:
        in_folder (str): The folder where the input and target folders are located
        out_folder (str): The folder where the new input and target folders will be created
        images (str): The name of the images to be processed 
        (e.g. "ROI_0071_06__S2A_MSIL2A_20200521T153631_R068_T17MPN_20200910T104928")
    Returns:
        bool: True is the process was successful
    """
    
    # Create the output folder
    out_folder = pathlib.Path(out_folder)
    
    out_input = out_folder / "S2"
    out_input_ref = out_folder / "S2ref"
    out_target = out_folder / "target"
    
    out_folder.mkdir(exist_ok=True)
    
    out_input.mkdir(exist_ok=True)
    out_input_ref.mkdir(exist_ok=True)
    out_target.mkdir(exist_ok=True)
        
    # Get the list of folders in the target iris folder
    target_folder = "%s/sugarcane-burning.iris/segmentation/" % in_folder
    target_folders = list(pathlib.Path(target_folder).glob("*"))
    target_folders.sort()
    
    S2_folders = list(pathlib.Path(in_folder).glob("*"))
    
    # remove the "cloud-segmentation.iris" and "images" folders
    S2_folders = [x for x in S2_folders if x.name not in 
                  ["sugarcane-burning.iris", "images", "metadata.json"]]
    S2_folders.sort()
    
    # Check 01
    if len(target_folders) == 0:
        print("No folders found")
        return None
    
    # Check 02
    if len(S2_folders) != len(target_folders):
        print("Number of folders do not match: %d != %d" % (len(S2_folders), len(target_folders)))
        return None
        
    # Check 03 - does the name of the folders match?
    for i in range(len(S2_folders)):
        f1name = S2_folders[i].name
        f2name = target_folders[i].name
        if f1name != f2name:
            print("Folder names do not match: %s != %s" % (f1name, f2name))
            return None
    
    # Create pairs
    if images == "all":
        dataset = list(zip(S2_folders, target_folders))
    else:
        dataset = [(x, y) for x, y in zip(S2_folders, target_folders) if x.name == images]
    
    # Get IRIS project parameters    
    metadata_file = "%s/metadata.json" % in_folder
    with open(metadata_file) as f:
        mask = json.load(f)["segmentation"]["mask_area"] 
        # e.g. (250, 250, 2250, 2250) (xmin, ymin, xmax, ymax)
    
    # Save the output
    for _, (input_folder, target_folder) in enumerate(dataset):
        print("Processing %s" % input_folder.name)
        
        # Obtain the S2 data
        S2file = "%s/S2.tif" % input_folder
        with rio.open(S2file) as src:
            S2data = src.read()
            S2metadata = src.meta
            S2_masked = S2data[:, mask[1]:mask[3], mask[0]:mask[2]]
 
        # Obtain the S2 data
        S2reffile = "%s/S2_ref.tif" % input_folder
        with rio.open(S2reffile) as src:
            S2refdata = src.read()
            S2ref_masked = S2refdata[:, mask[1]:mask[3], mask[0]:mask[2]]

        # Modify affine, height and width to match the new data
        new_affine = S2metadata["transform"]
        scale_x = S2metadata["transform"][0]
        scale_y = S2metadata["transform"][4]
        new_affine = Affine(
            scale_x, new_affine[1], (new_affine[2] + mask[0]*10),
            new_affine[3], scale_y, (new_affine[5] - mask[1]*10)
        ) 
        #(x_res, row_rotation, x_origin, col_rotation, y_res, y_origin)
        
        new_height = S2_masked.shape[1]
        new_width = S2_masked.shape[2]
        
        ## upgrade metadata
        S2metadata["transform"] = new_affine
        S2metadata["height"] = new_height
        S2metadata["width"] = new_width
        
        target_metadata = S2metadata.copy()
        target_metadata.update({"dtype":np.int8})
        target_metadata["count"] = 1
        
        # Obtain the target data
        targetfile1 = "%s/1_user.npy" % target_folder
        targetdata1 = np.load(targetfile1)
        targetfile2 = "%s/1_final.npy" % target_folder
        targetdata2 = np.load(targetfile2)
        
        # final mask
        burn_mask = np.argmax(targetdata2, axis = -1) * targetdata1
        burn_mask.astype(np.int8)

        # Margin of 16 pixels
        burn_mask[burn_mask == 0] = -99
        burn_mask[:16, :16] = -99
        burn_mask[-16:, -16:] = -99

        # Convert No Burn in 0
        burn_mask[burn_mask == 2] = 0
                                        
        # Write the results
        with rio.open("%s/%s.tif" % (out_input, input_folder.name), "w", **S2metadata) as dst:
            dst.write(S2_masked)
        
        with rio.open("%s/%s.tif" % (out_input_ref, input_folder.name), "w", **S2metadata) as dst:
            dst.write(S2ref_masked)
        
        with rio.open("%s/%s.tif" % (out_target, input_folder.name), "w", **target_metadata) as dst:
            dst.write(burn_mask, 1)
    
    return True


def calculate_slope(dem_m: np.ndarray) -> np.ndarray:
    """
    Calculates the slope percentage from a Digital Elevation Model (DEM).

    Args:
    - dem_m (np.ndarray): DEM array.

    Returns:
    - np.ndarray: Slope percentage array.
    """
    gradient_y, gradient_x = np.gradient(dem_m, edge_order=1)
    slope_percentage = np.sqrt(gradient_x**2 + gradient_y**2) / 10 * 100
    return slope_percentage


def calculate_distances(
    cover: np.ndarray,
    resolution: int
    ) -> np.ndarray:
    """
    Calculates the distance to the nearest cell with a value of 5 in the cover array.

    Args:
    - cover (np.ndarray): Cover array.
    - resolution (int): Spatial resolution.

    Returns:
    - np.ndarray: Distance array.
    """
    distance_to_5 = distance_transform_edt(cover != 5) # 5 for Crops LULC class
    distance = distance_to_5 * resolution
    return distance


def generate_lc_slope_tensor(
    lat: float,
    lon: float,
    date_start: str,
    date_end: str
) -> Optional[torch.Tensor]:
    """
    Generates a tensor with land cover and slope percentage bands.

    Args:
    - lat (float): Latitude of the location.
    - lon (float): Longitude of the location.
    - date_start (str): Start date in 'YYYY-MM-DD' format.
    - date_end (str): End date in 'YYYY-MM-DD' format.
    - save_path (str): File path where the tensor will be saved.

    Returns:
    - Optional[torch.Tensor]: Generated tensor,
    or None if the date range is invalid.
    """

    # Creating data cube for land cover (Annual LULC)
    landcover_dc = cubo.create(
        lat=lat,
        lon=lon,
        collection="io-lulc-9-class",
        start_date=date_start,
        end_date=date_end,
        edge_size = 512,
        resolution = 10
    )[0, 0]    
    
    landcover_np = landcover_dc.to_numpy()
    landcover_dist = calculate_distances(landcover_np, 10)
    landcover_dist = landcover_dist[None, None].repeat(2, axis=0)

    # Calculating Slope
    dem_dc = cubo.create(
        lat=lat,
        lon=lon,
        collection="nasadem",
        start_date="2000-02-20",
        end_date="2000-02-20",
        edge_size = 512,
        resolution = 10
    )[0, 0]

    dem_np = dem_dc.to_numpy()
    slope_percentage = calculate_slope(dem_np)
    slope_percentage = slope_percentage[None, None].repeat(2, axis=0)

    # Stacking all bands
    full_tensor = np.concatenate([
        landcover_dist.astype(np.float32),
        slope_percentage.astype(np.float32)
    ], axis=1)

    # Calculate the range for lat and lon values based on the center and edge size
    # half the extent in meters
    half_extent_meters = 5120 / 2  
    
    # Convert meters to degrees latitude
    half_extent_degrees_lat = half_extent_meters / 111320  
    
    # Convert meters to degrees longitude
    half_extent_degrees_lon = half_extent_meters / (111320 * np.cos(np.radians(lat)))  

    lat_vals = np.linspace(lat - half_extent_degrees_lat, 
                           lat + half_extent_degrees_lat, 
                           512)
    lon_vals = np.linspace(lon - half_extent_degrees_lon, 
                           lon + half_extent_degrees_lon, 
                           512)
    
    # From numpy to xarray
    extra_bands = ["lcdist", "slope_percentage"]
    full_tensor = xr.DataArray(
        full_tensor,
        dims=["time", "bands", "lat", "lon"],
        coords={
            "time": np.arange(full_tensor.shape[0]),
            "bands": extra_bands,
            "lat": lat_vals,
            "lon": lon_vals
        }
    )

    return full_tensor

   
def calculate_class_weights(target_files: List[pathlib.Path]) -> List[float]:
    """
    Calculate class weights for a set of target files.

    Args:
    - target_files (List[pathlib.Path]): List of target files.

    Returns:
    - List[float]: Weights of binary classes.
    """
    class_counts = [0, 0]

    for target_path in target_files:
        with rio.open(target_path) as tgt:
            mask = tgt.read(1)

            class_counts[0] += (mask == 0).sum()
            class_counts[1] += (mask == 1).sum()
   
    total_counts = sum(class_counts)
    class_weights = [total_counts / class_counts[0], total_counts / class_counts[1]]
    return class_weights


def calculate_mean_std(input_files: List[pathlib.Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and standard deviation of a set of input files.

    Args:
    - input_files (List[pathlib.Path]): List of input files.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Mean and standard deviation.
    """
    mean, std = [], []
    for i, file in enumerate(input_files):
        with rio.open(file) as src:
            array = src.read()
            mean.append(np.nanmean(array, axis=(1, 2)))
            std.append(np.nanstd(array, axis=(1, 2)))
            print(f"[{i+1}/{len(input_files)}]: Processed {file.name}")
    
    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)

    return mean, std


def vectorize_raster(array: np.ndarray,
                     crs: str, 
                     transform: rio.transform.Affine,
                     out_file: pathlib.Path
    ) -> None:
    """
    Vectorize a raster array and export it to a vector file.

    Args:
        array (np.ndarray): Raster array to be vectorized.
        crs (str): EPSG code of the raster.
        transform (rio.transform.Affine): Affine transformation of the raster.
        out_file (pathlib.Path): Path to the output vector file.
    """

    # Create a list to store the geometries and areas
    burn_polygons = []
    area_list = []

    # Vectorize the raster array
    shapes_generator = rio.features.shapes(array, transform=transform)
    for geom, value in shapes_generator:
        if value == 1:  # Only consider the burned areas
            burn_polygons.append(shapely.geometry.shape(geom))
            # Calculating the area in ha
            area = shapely.geometry.shape(geom).area
            area_list.append(area * (0.0001))

    # Create a GeoDataFrame with the geometries and areas
    burn_gdf = gpd.GeoDataFrame({'area': area_list,
                                 'geometry': burn_polygons}, crs=crs)

    # Export the GeoDataFrame to a GeoJSON file
    burn_gdf.to_file(out_file, driver='GeoJSON')