## Inference file
import argparse
import pathlib
import pandas as pd
from collections import defaultdict
import shutil
import pyproj
import rasterio as rio
from utils import (load_config, download_datacube_data, 
                   omit_black_files, dbadi_filter, 
                   rio_calculate_indexs, generate_lc_slope_tensor
                   )
from evaluator import save_ensemble_prob_models, make_predict_lr

# Define the inference function
def inference(config_path):
    # Load configuration file
    config = load_config(config_path)
    # Load parameters from config
    inference_config = config['inference_config']
    INFERENCE_FOLDER = pathlib.Path(inference_config['inference_folder'])
    lat = float(inference_config['lat'])
    lon = float(inference_config['lon'])
    start_date = inference_config['start_date']
    end_date = inference_config['end_date']
    cloud_model = inference_config['cloud_model']
    checkpoint_path = inference_config['checkpoint_path']
    gbm_models = inference_config['gbm_models']
    stacking_model = inference_config['stacking_model']

    ## Download the data
    S2BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", 
               "B09", "B11", "B12"]
    download_datacube_data((lat, lon), cloud_model, start_date, end_date, 
                           S2BANDS, INFERENCE_FOLDER / "S2")

    s2_files = list((INFERENCE_FOLDER / "S2").glob("*.tif"))

    ## Apply cloud, omit black areas and dBADI
    optimized_files = omit_black_files(s2_files, cloud_model)

    ## Group files by ROI and sort by date
    roi_groups = defaultdict(list)
    for file in optimized_files:
        roi_groups[file.parent.stem].append(file)

    # Sort each group by date
    for roi, files in roi_groups.items():
        files.sort(key=lambda x: pd.to_datetime(x.stem.split("_")[2]))

    burning_files = dbadi_filter(roi_groups)   
    ## Save in new folder
    s2_filter_folder = INFERENCE_FOLDER / "S2_filtered"
    s2_filter_folder.mkdir(exist_ok=True)

    for file in burning_files:
        ## Using shutil.copy
        shutil.copy(file, s2_filter_folder / file.name)

    print(f"Found {len(burning_files)} files to process.")

    ## Apply the models
    ## Create the folders
    NBR_FOLDER = INFERENCE_FOLDER / "nbr"
    BADI_FOLDER = INFERENCE_FOLDER / "badi"
    SLOPE_FOLDER = INFERENCE_FOLDER / "slope"
    NDVI_FOLDER = INFERENCE_FOLDER / "ndvi"
    NDWI_FOLDER = INFERENCE_FOLDER / "ndwi"
    LANDCOVER_FOLDER = INFERENCE_FOLDER / "dlc"

    NBR_FOLDER.mkdir(exist_ok=True)
    NDVI_FOLDER.mkdir(exist_ok=True)
    NDWI_FOLDER.mkdir(exist_ok=True)
    BADI_FOLDER.mkdir(exist_ok=True)
    SLOPE_FOLDER.mkdir(exist_ok=True)
    LANDCOVER_FOLDER.mkdir(exist_ok=True)

    ## Iterate over the files
    s2_filtered_files = sorted(list(s2_filter_folder.glob("*.tif")))
    for i, file in enumerate(s2_filtered_files):
        with rio.open(file) as src:
            # From Affine meta, get the center of the image
            affine = src.transform
            center_x = affine[2] + affine[0] * src.width / 2
            center_y = affine[5] + affine[4] * src.height / 2
            # Get the crs
            crs = src.crs.to_string()
            # Get the lon and lat
            transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
            lon, lat = transformer.transform(center_x, center_y)

            # Get the date
            date = pathlib.Path(file).stem.split("__")[1].split("_")[2]
            start_date = f'{date[:4]}-01-01'
            end_date = f'{date[:4]}-12-31'
            
            # Copy the profile
            profile = src.profile.copy()

            # Get the slope
            slope_lc = generate_lc_slope_tensor(lat = lat, 
                                                lon = lon, 
                                                date_start = start_date, 
                                                date_end = end_date)

            # Get the NBR and BADI indices
            nbr = rio_calculate_indexs(src, index="nbr")
            badi = rio_calculate_indexs(src, index="badi")
            ndvi = rio_calculate_indexs(src, index="ndvi")
            ndwi = rio_calculate_indexs(src, index="ndwi")
    
            profile = src.profile.copy()
            profile.update(count=1, dtype=rio.float32)
            # Save the indices
            with rio.open(NBR_FOLDER / file.name, "w", **profile) as dst:
                dst.write(nbr[None])

            with rio.open(BADI_FOLDER / file.name, "w", **profile) as dst:
                dst.write(badi[None])

            with rio.open(NDVI_FOLDER / file.name, "w", **profile) as dst:
                dst.write(ndvi[None])

            with rio.open(NDWI_FOLDER / file.name, "w", **profile) as dst:
                dst.write(ndwi[None])  

            with rio.open(LANDCOVER_FOLDER / file.name, "w", **profile) as dst:
                dst.write(slope_lc[0][0], 1)
            
            # Get the land cover
            with rio.open(SLOPE_FOLDER / file.name, "w", **profile) as dst:
                dst.write(slope_lc[0][1], 1)

            print(f"[{i+1}/{len(s2_files)}]: Processed {file.name}")

    ## Zip files
    nbr_files = sorted(list(NBR_FOLDER.glob("*.tif")))
    badi_files = sorted(list(BADI_FOLDER.glob("*.tif")))
    slope_files = sorted(list(SLOPE_FOLDER.glob("*.tif")))
    ndvi_files = sorted(list(NDVI_FOLDER.glob("*.tif")))
    ndwi_files = sorted(list(NDWI_FOLDER.glob("*.tif")))
    landcover_files = sorted(list(LANDCOVER_FOLDER.glob("*.tif")))

    input_files = list(zip(s2_filtered_files, nbr_files, badi_files, 
                        slope_files, ndvi_files, ndwi_files, 
                        landcover_files))

    ## Generate the predictions
    save_ensemble_prob_models(ckpt_path=checkpoint_path,
                            gbm_path=gbm_models,
                            image_paths=input_files,
                            config=config,
                            out_folder=INFERENCE_FOLDER / "predictions")

    ## Apply the stacking model
    pred_files = list((INFERENCE_FOLDER / "predictions").glob("*.tif"))

    for i, file in enumerate(pred_files):
        with rio.open(file) as src:
            pred = make_predict_lr(src.read(), stacking_model)
            with rio.open(file, "w", **src.profile) as dst:
                dst.write(pred, 1)
            
            print(f"[{i+1}/{len(pred_files)}]: Processed {file.name}")
        
    print("Inference finished! The results are saved in the predictions folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config YAML file.')
    args = parser.parse_args()
    
    # Run inference
    inference(config_path=args.config)
                          
                          




