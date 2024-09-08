import numpy as np
import rasterio as rio
import pathlib
import matplotlib.pyplot as plt

# Load the dataset
DATASET_FOLDER = pathlib.Path("D:/dataset/database")
PROB_FOLDER = DATASET_FOLDER / "probabilities"
prob_files = list(PROB_FOLDER.glob("*.tif"))

DATAFOLDER = pathlib.Path("D:\dataset\emergencies\S2")
datafiles = list(DATAFOLDER.glob("*.tif"))
data = [f.name for f in datafiles]
import geopandas as gpd
import pyproj
gdf = gpd.read_file("data/vector/historical_data_aligned.geojson")

for i, filename in enumerate(data):
    filename = pathlib.Path(filename).stem
    # Plot the four images (original, ML, DL, ensemble)
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    fig.suptitle(filename)

    folder = "D:/dataset/emergencies"
    image_path = f"{folder}/S2/{filename}.tif"
    prob_path = f"{folder}/probabilities/{filename}.tif"    
    gdf_filtered = gdf[gdf["id"] == filename.split("__")[0]]  
    
    with rio.open(image_path) as src, rio.open(prob_path) as prob:
        # Get the crs
        crs = src.crs.to_string()
        axs[0,0].imshow(prob.read(1))
        axs[0,0].set_title("Unet Model")

        axs[0,1].imshow(prob.read(2))
        axs[0,1].set_title("LightGBM Model")

        axs[1,0].imshow(np.dstack((src.read()[4], src.read()[3], src.read()[2]))/10000)
        axs[1,0].set_title("Original")

        array = prob.read()
        ensemble =  make_predict_lr(array, "models/stacking_model.pkl")
        axs[1,1].imshow(ensemble) # Ensemble model
        axs[1,1].set_title("Ensemble Model")

        tgt = ensemble.copy() 
        # 0.11366338311039614
        # 0.13264731249651068
        tgt[tgt >= 0.250] = 1
        tgt[tgt < 0.250] = np.nan

        ## Plot the gpd
        xmin, ymin, xmax, ymax = src.bounds
        axs[1,2].imshow(np.dstack((src.read()[4], src.read()[3], src.read()[2]))/10000,
                        extent=[xmin, xmax, ymin, ymax])
        
        # Reproject the GeoDataFrame to match the image's CRS
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        gdf_filtered['x'], gdf_filtered['y'] = transformer.transform(gdf_filtered.geometry.x.values[0], 
                                                                     gdf_filtered.geometry.y.values[0])

        # Plot the points
        axs[1, 2].scatter(gdf_filtered['x'], gdf_filtered['y'], facecolors='none', 
                          edgecolors='blue', s=10)

        axs[1,2].imshow(tgt, cmap=ListedColormap(['red']), extent=[xmin, xmax, ymin, ymax])
        axs[1,2].set_title("Binary Mask")

        for ax in axs.flatten():
            ax.axis("off")

        plt.savefig(f"D:/img4/{filename}_ensemble_model.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.clf()
        
        print(f"[{i+1}/{len(data)}] {filename} processed")
