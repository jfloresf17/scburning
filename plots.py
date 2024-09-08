import pathlib
from typing import List, Tuple, Union
import rasterio as rio
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from utils import (read_bands, get_rgb, create_burned_mask, 
                   vectorize_raster, rio_calculate_indexs)
from evaluator import make_predict_lr


def vis_image(path: pathlib.Path, 
              roi_name: str,
              factor_list: Union[List[int], List[float]], 
              out_dir: str
):
    '''
    Plot RGB images for burn indexs and masks.

    Args:
        path (pathlib.Path): The path to the image.
        roi_name (str): The name of the Region of Interest (ROI).
        factor_list (List[int, float]): The contrast factor for each plot.
        out_dir (str): The output directory.
    '''
    # Open a raster file    
    with rio.open(path) as src:
        # Normalize the bands
        bands = [2, 3, 4, 8, 9, 11, 12]

        # Assign bands to descriptive variable names
        b2, b3, b4, b8, b8a, b11, b12 = read_bands(src, bands)

        # Create a figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15),
                                gridspec_kw={'hspace': 0.15, 'wspace': -0.5})

        # Get RGB images
        true_color_432 = get_rgb(b4, b3, b2, factor_list[0])
        false_color_843 = get_rgb(b8, b4, b3, factor_list[1])
        swir_8A124 = get_rgb(b8a, b12, b4, factor_list[2])
        agriculture_1182 = get_rgb(b11, b8, b2, factor_list[3])

        # Subplot 1: Natural Color (B4, B3, B2)
        axs[0, 0].imshow(true_color_432)
        axs[0, 0].set_title("Natural Color (B4, B3, B2)")

        # Subplot 2: Infrared Color (B8, B4, B3)
        axs[0, 1].imshow(false_color_843)
        axs[0, 1].set_title("Infrared Color (B8, B4, B3)")

        # Subplot 3: Short-Wave Infrared (B8A, B12, B4)
        axs[1, 0].imshow(swir_8A124)
        axs[1, 0].set_title("Short-Wave Infrared (B8A, B12, B4)")

        # Subplot 4: Agriculture (B11, B8, B2)
        axs[1, 1].imshow(agriculture_1182)
        axs[1, 1].set_title("Agriculture (B11, B8, B2)")

        # Index calculation
        nbr_mask = create_burned_mask(src, "mono", "nbr", "less", -0.15) 

        ## Plot NBR mask
        axs[2, 0].imshow(true_color_432)
        axs[2, 0].imshow(nbr_mask, cmap=ListedColormap(['red']))
        axs[2, 0].set_title("Natural Color with NBR mask")

        # Subplot 6: Natural Color with BADI, NDVI, and NDWI mask
        ## Calculate BADI index
        badi_mask = create_burned_mask(src, "mono", "badi", "greater", 0.3)

        ## Plot BADI mask
        axs[2, 1].imshow(true_color_432)
        axs[2, 1].imshow(badi_mask, cmap=ListedColormap(['#8c00ff']))
        axs[2, 1].set_title("Natural Color with BADI mask")

        for ax in axs.flat:
            ax.axis('off')
        
        # Convert the filename to a datetime string
        datetime = path.stem.split("__")[1].split("_")[2]

        date_info = f"{datetime[:4]}/{datetime[4:6]}/{datetime[6:8]}"

        # Add a general title to the figure
        fig.suptitle(f"{roi_name} - {date_info}", fontsize=16, 
                     fontweight='normal', ha='center', x=0.508, y=0.925)

        # Save the figure as a PNG file
        # Create a directory for each Region of Interest (ROI)
        roi_folder = pathlib.Path(out_dir)/roi_name
        roi_folder.mkdir(parents=True, exist_ok=True)

        # Save the figure as a PNG file
        output_file = f"{roi_folder}/{path.stem}.png"
        print(f"Output File: {output_file}")  # Print the output file path

        fig.savefig(output_file, dpi=300, bbox_inches='tight')

        plt.close()
        plt.clf()        

def vis_roc_curve (y_test: np.ndarray,
                   y_pred_proba: np.ndarray,
                   out_file: str
) -> None: 
    """
    Visualize the ROC curve.

    Args:
        y_test (np.ndarray): The true values.
        y_pred_proba (np.ndarray): The predicted probabilities.
        out_file (str): The output file.
    
    Returns:
        None
    """
    ## Get the false positive rate, true positive rate, and thresholds
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    ## Calculate the area under the curve
    roc_auc = auc(fpr, tpr)

    ## Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    plt.clf()


def vis_results(source_folder: pathlib.Path,
                filenames: List[str],
                threshold: float,
                model: str,
                factor: int,
                out_dir: pathlib.Path
) -> None:
    """
    Visualize the results of the predictions.

    Args:
        source_folder (pathlib.Path): The path to the source folder.
        filenames (List[str]): The list of filenames.
        threshold (float): The threshold value.
        model (str): The model name. Only three options: "Stacking", "U-Net", "GBDT".
        factor (int): The brightness factor.
        out_dir (pathlib.Path): The output directory.

    Returns:
        None 
    """                
    # Create the paths for each file

    image_path = [source_folder /f"S2/{filename}.tif" for filename in filenames]
    prob_path = [source_folder / f"probabilities/{filename}.tif" for filename in filenames]

    out_folder = out_dir / "results"
    out_folder.mkdir(exist_ok=True, parents=True)

    # Create the figure and plots
    fig, axs = plt.subplots(3, 6, figsize=(18,9), 
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    
    # The titles for each subplot
    titles = ['Zoomed Area', 'Agriculture Color', f'{model} (p ≥ {threshold:.2f})']
    for i, title in enumerate(titles):
        axs[i, 0].text(0.5, 0.5, title, ha='center', va='center', fontsize=14, 
                       fontweight='bold')
        axs[i, 0].axis('off')

    # Configure only the first row subplots to have axes turned off
    for ax in axs[0, :]:
        ax.axis('off')  # Turn off all axes in the first row
    
    # Configure each subplot for the image
    for ax in axs.flat:
        ax.set_xticks([]) 
        ax.set_yticks([])  
        ax.set_aspect('auto') 

    # Iterate over each filename and plot the results according to the model
    for col, (img_path, prob_path) in enumerate(zip(image_path, prob_path)):
        with rio.open(img_path) as src:
            agriculture_color = np.transpose(src.read([11, 8, 2]), (1, 2, 0))
            agriculture_color = agriculture_color * factor / 10000
        
        with rio.open(prob_path) as prob:
            if model == "Stacking":
                prediction =  make_predict_lr(prob.read(), 
                                              "models/stacking_model.pkl")
                prediction = (prediction >= threshold).astype(int)

            elif model == "U-Net":
                prediction = prob.read(1)
                prediction = (prediction >= threshold).astype(int)
            
            elif model == "GBDT":
                prediction = prob.read(2)
                prediction = (prediction >= threshold).astype(int)

        # Plot the images
        axs[1, col+1].imshow(agriculture_color)
        axs[2, col+1].imshow(prediction, cmap=ListedColormap(['purple','red']))
        
    # Save the figure
    plt.tight_layout(pad=0)
    plt.savefig(out_folder / f"{model}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.clf()


def vis_oefaval(gdf : gpd.GeoDataFrame,
                emerg_filename: str,
                source_folder: pathlib.Path,
                factor: float,
                threshold_list: List[float],
                out_img_dir: pathlib.Path,
                out_vector_dir: pathlib.Path
) -> None:
    
    """ 
    Visualize the results for the emergency validation.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame with the emergency data.
        emerg_filename (str): The emergency filename.
        source_folder (pathlib.Path): The source folder.
        factor (float): The brightness factor.
        threshold_list (List[float]): The list of threshold values.
        out_img_dir (pathlib.Path): The output image directory.
        out_vector_dir (pathlib.Path): The output vector directory.
    
    Returns:
        None
    """

    # Get the emergency ID and the date
    date = emerg_filename.split("__")[1].split("_")[2]
    date = f"{date[:4]}/{date[4:6]}/{date[6:8]}"

    # Define the paths for the image and the probabilities
    image_path = source_folder / f"S2/{emerg_filename}.tif"
    prob_path = source_folder / f"probabilities/{emerg_filename}.tif"

    # Filter the GeoDataFrame by the emergency ID
    gdf_filtered = gdf[gdf["id"] == emerg_filename.split("__")[0]]

    # Get the emergency date and the ROI name
    datetime = gdf_filtered["FE_FECHA_EMERG"].values[0]
    emergency_date = pd.to_datetime(datetime).strftime("%Y/%m/%d")
    roi_name = gdf_filtered["id"].values[0].split("_")[1]

    # Create the output folder
    out_folder = out_img_dir / "oefa_val"
    out_folder.mkdir(exist_ok=True, parents=True)

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # 2 filas y 3 columnas

    # Configure the title of the figure
    fig.suptitle(f"Emergency - {roi_name} on {emergency_date} | Sentinel-2 ({date})", fontsize=16)

    # Open the image and the probabilities
    with rio.open(image_path) as src, rio.open(prob_path) as prob:
        # Get the crs and transform of the image 
        crs = src.crs.to_string()
        transform = src.transform 
       
       # Get the bounds of the image
        xmin, ymin, xmax, ymax = src.bounds
        
        # Reproject the GeoDataFrame to the image crs
        gdf_filtered = gdf_filtered.to_crs(crs)
  
        # Agriculture composite
        agriculture = np.dstack((src.read(11), src.read(8), src.read(2))) * factor / 10000

        # Plot the Agriculture Color
        axs[0, 0].imshow(agriculture, extent= [xmin, xmax, ymin, ymax])
        axs[0, 0].scatter(gdf_filtered.geometry.x.values[0], 
                          gdf_filtered.geometry.y.values[0], 
                          facecolors='none', edgecolors='blue', s=50)
        axs[0, 0].set_title("Agriculture (B11, B8, B2)")

        # Read the probabilities and apply the stacking model
        array = prob.read()
        ensemble = make_predict_lr(array, "models/stacking_model.pkl")

        # Define the thresholds
        for i, threshold in enumerate(threshold_list):
            # Create a binary mask using the threshold
            tgt = np.where(ensemble >= threshold, 1, 0)

            # Export the masks to a vector file
            VECTOR_FOLDER = out_vector_dir / f"emergency_{roi_name}"
            VECTOR_FOLDER.mkdir(exist_ok=True, parents=True)

            vectorize_raster(tgt, crs, transform, VECTOR_FOLDER / f"{threshold}.json")
            
            # Determinar la posición del subplot
            row, col = divmod(i + 1, 3)           

            # Graficar la máscara binaria con la imagen de fondo
            axs[row, col].imshow(agriculture, extent=[xmin, xmax, ymin, ymax])
            axs[row, col].scatter(gdf_filtered.geometry.x.values[0], 
                                  gdf_filtered.geometry.y.values[0], 
                                  facecolors='none', edgecolors='blue', s=50)            
            axs[row, col].imshow(np.where(tgt == 1, 1, np.nan), 
                                 cmap=ListedColormap(['red']), alpha=0.5, 
                                 extent=[xmin, xmax, ymin, ymax])
            axs[row, col].set_title(f"Threshold: {threshold:.2f}")

        # Desactivar los ejes de todos los subplots
        for ax in axs.flatten():
            ax.axis("off")

    # Guardar y mostrar la figura
    plt.savefig(out_folder / f"{roi_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.clf()


def get_sorted_files(pattern: str,
                     folder: pathlib.Path 
) -> List[pathlib.Path]:
    """
    Get the sorted files by date from a folder using a pattern.

    Args:
        pattern (str): The pattern to filter the files.
        folder (pathlib.Path): The folder path.
    
    Returns:
        List[pathlib.Path]: The list of sorted files.
    """ 

    files = sorted(list(folder.glob(pattern)))
    
    return sorted(files, key=lambda f: pd.to_datetime(f.stem.split("__")[1].split("_")[2]))


def load_files_and_apply_model(file_list: List[pathlib.Path],
                               prob_list: List[pathlib.Path], 
                               index: int, 
                               factor: float,   
                               threshold:float, 
                               out_vector_dir: pathlib.Path
) -> Tuple[np.ndarray, np.ndarray, pathlib.Path]:
    """
    Load the files and apply the model to generate the binary mask.

    Args:
        file_list (List[pathlib.Path]): The list of files.
        prob_list (List[pathlib.Path]): The list of probabilities.
        factor (float): The brightness factor.
        index (int): The index of the file.
        threshold (float): The threshold value.
        out_vector_dir (pathlib.Path): The output vector directory.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, pathlib.Path]: The source in agriculture 
        composite, mask, and filename.
    """
        
    with rio.open(file_list[index]) as src:
        # Create the agriculture composite
        agr_composite = np.dstack((src.read(11), src.read(8), src.read(2))) * factor / 10000
        # Generate the  binary mask from the stacking probabilities
        prob = rio.open(prob_list[index]).read()
        ensemble = make_predict_lr(prob, "models/stacking_model.pkl")
        mask = (ensemble >= threshold).astype(int)
        ndvi = rio_calculate_indexs(src, "ndvi")
        ndvi = (ndvi <= 0.2).astype(int)
        ndwi = rio_calculate_indexs(src, "ndwi")
        ndwi = (ndwi <= 0.0).astype(int)

        # Apply the mask to the binary mask
        mask = mask * ndvi * ndwi
        # Get the date and convert it to a strftime format
        date = file_list[index].stem.split('__')[1].split('_')[2]
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')

        # Vectorize the mask and save it with the date as the filename
        vectorize_raster(mask, src.crs.to_string(), src.transform, 
                         out_vector_dir / f"{date_str}.json")
    
    return agr_composite, mask, file_list[index]


def vis_time_series(pilot_folder: pathlib.Path,
                    specific_file: str,
                    factor: float,
                    out_dir: pathlib.Path,
                    threshold: List[float],
                    out_vector_dir: pathlib.Path                     
) -> None:
    """
    Visualize the time series of the pilot data.

    Args:
        pilot_folder (pathlib.Path): The pilot folder.
        specific_file (str): The specific file.
        factor (float): The brightness factor.
        out_dir (pathlib.Path): The output directory.
        threshold (List[float]): The list of threshold values.
        out_vector_dir (pathlib.Path): The output vector directory.
    
    Returns:
        None
    """
    # Get the sorted files for the pilot data and their probabilities    
    sorted_pilot_files = get_sorted_files("*S2/*.tif", pilot_folder)
    sorted_prob_files = get_sorted_files("probabilities/*.tif", pilot_folder)

    # Get the indices of the specific file (2 image before and 2 image after)
    index = next((i for i, f in enumerate(sorted_pilot_files) if f.name == specific_file), None)
    if index is not None and 2 <= index <= len(sorted_pilot_files) - 3:
        indices = range(index - 2, index + 3)
        results = [load_files_and_apply_model(sorted_pilot_files, sorted_prob_files, i,
                                              factor, threshold[n], out_vector_dir) 
                   for n,i in enumerate(indices)]
                
    # Visualize the results
    fig = plt.figure(figsize=(25, 5))
    gs = gridspec.GridSpec(1, 5)

    for i, (agr_composite, mask, file_name) in enumerate(results):
        ax = plt.subplot(gs[i])
        ax.imshow(agr_composite)
        ax.contour(mask, levels=[0.5], colors='red', linewidths=0.75, linestyles='solid')
        date_str = pd.to_datetime(file_name.stem.split('__')[1].split('_')[2]).strftime('%Y-%m-%d')
        ax.set_title(f"{date_str} | ({np.sum(mask) * 0.01:.2f} ha)", fontsize=20)
        ax.axis('off')
        
    central_date = pd.to_datetime(results[2][2].stem.split('__')[1].split('_')[2]).strftime('%Y-%m-%d')
    
    out_folder = out_dir / "pilot"
    out_folder.mkdir(exist_ok=True, parents=True)

    # Guardar y mostrar la figura
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    plt.savefig(out_folder / f"{central_date}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.clf()


def vis_comparation(filename: str,
                    gdf: gpd.GeoDataFrame,
                    source_folder: pathlib.Path,
                    factor: float,
                    threshold: List[float],
                    out_dir: pathlib.Path
) -> None:
    """
    Visualize the comparison between the models.

    Args:
        filename (str): The filename.
        gdf (gpd.GeoDataFrame): The GeoDataFrame.
        source_folder (pathlib.Path): The source folder.
        factor (float): The brightness factor.
        threshold (List[float]): The list of threshold values.
        out_dir (pathlib.Path): The output directory.
    
    Returns:
        None
    """
    # Get the date from the filename
    date = filename.split("__")[1].split("_")[2]
    date = f"{date[:4]}/{date[4:6]}/{date[6:8]}"

    # Define the paths for the image and the probabilities
    image_path = source_folder / f"S2/{filename}.tif"
    prob_path = source_folder / f"probabilities/{filename}.tif"

    # Filter the GeoDataFrame by the emergency ID
    gdf_filtered = gdf[gdf["id"] == filename.split("__")[0]]

    # Convert the emergency date to a string 
    emerg_date = gdf_filtered["FE_FECHA_EMERG"].values[0]
    emergency_date = pd.to_datetime(emerg_date).strftime("%Y/%m/%d")

    # Get the ROI name
    roi_name = gdf_filtered["id"].values[0].split("_")[1]

    # Create the output folder
    out_folder = out_dir / "comparation"
    out_folder.mkdir(exist_ok=True, parents=True)

    # Create the figure and subplots
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    # Open the image and the probabilities
    with rio.open(image_path) as src, rio.open(prob_path) as prob:
        crs = src.crs.to_string()
        xmin, ymin, xmax, ymax = src.bounds

        gdf_filtered = gdf_filtered.to_crs(crs)

        # Create the agriculture composite
        combined_image = np.dstack((src.read(11), src.read(8), src.read(2))) * factor / 10000

        # Read the probabilities and apply the stacking model
        gbm = prob.read(2)
        unet = prob.read(1)
        ensemble = make_predict_lr(prob.read(), "models/stacking_model.pkl")

        # Apply the threshold to the probabilities
        ## BADI
        badi = create_burned_mask(src, "mono", "badi", "greater", threshold[0])

        ## Models 
        unet_binary = np.where(unet >= threshold[1], 1, np.nan)
        gbm_binary = np.where(gbm >= threshold[2], 1, np.nan)
        ensemble_binary = np.where(ensemble >= threshold[3], 1, np.nan)     
        
        # Plot 1: Agriculture Color
        axs[0].imshow(combined_image, extent=[xmin, xmax, ymin, ymax])
        axs[0].scatter(gdf_filtered.geometry.x.values[0], 
                        gdf_filtered.geometry.y.values[0], 
                        facecolors='none', edgecolors='blue', s=50)
        axs[0].set_title("Agriculture (B11, B8, B2)")

        # Plot 2: BADI
        axs[1].imshow(combined_image, extent=[xmin, xmax, ymin, ymax])
        axs[1].imshow(badi, cmap=ListedColormap(['red']), alpha=0.5, extent=[xmin, xmax, ymin, ymax])
        axs[1].set_title(f"BADI (threshold={threshold[0]:.2f})")

        # Plot 3: UNET
        axs[2].imshow(combined_image, extent=[xmin, xmax, ymin, ymax])
        axs[2].imshow(unet_binary, cmap=ListedColormap(['red']), extent=[xmin, xmax, ymin, ymax])
        axs[2].set_title(f"UNET (Threshold={threshold[1]:.2f})")

        # Plot 4: GBM
        axs[3].imshow(combined_image, extent=[xmin, xmax, ymin, ymax])
        axs[3].imshow(gbm_binary, cmap=ListedColormap(['red']), extent=[xmin, xmax, ymin, ymax])
        axs[3].set_title(f"GBM (Threshold={threshold[2]:.2f})")

        # Plot 5: Ensemble
        axs[4].imshow(combined_image, extent=[xmin, xmax, ymin, ymax])
        axs[4].imshow(ensemble_binary, cmap=ListedColormap(['red']), extent=[xmin, xmax, ymin, ymax])
        axs[4].set_title(f"Stacking (Threshold={threshold[3]:.2f})")

        # Desactivar los ejes de todos los subplots
        for ax in axs:
            ax.axis("off")

    fig.suptitle(f"Emergency - {roi_name} on {emergency_date} | Sentinel-2 ({date})", fontsize=16, y=1.1)
    # Guardar y mostrar la figura
    plt.savefig(out_folder / f"{roi_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.clf()
