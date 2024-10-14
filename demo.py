from utils import *
import geopandas as gpd
import einops
import pathlib
from collections import defaultdict
import numpy as np
import rasterio as rio
import torch
import matplotlib.pyplot as plt
import pyproj
import pandas as pd
import seaborn as sns
from evaluator import (save_ensemble_prob_models, make_predict_ml,
                       evaluate_metrics_ml)
from model import lgb_model, stacking_classifier
from sklearn.model_selection import train_test_split
import joblib
from plots import *


# FIRST STEP: ALIGN THE POINTS AND DOWNLOAD THE IMAGES
## Load deep learning model
model = torch.jit.load("data/model/cloudmodel.pt")

## Load buffer areas 
rois = gpd.read_file("data/vector/rois.geojson")
## Convert to centroid
rois["geometry"] = rois.centroid
## Transform the points to EPSG:4326
rois = rois.to_crs("epsg:4326")  

## Align the points to the grid
aligned_rois = align_points_to_grid(rois, 10, "id")

## Save the aligned points
aligned_rois.to_file("data/vector/aligned_rois.geojson", driver="GeoJSON")

## Define the image folder
SCBURNING_FOLDER = pathlib.Path("D:/scburning")
S2BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

## Iterate over the points
for roi in aligned_rois.itertuples(): 
    download_datacube_data(roi, model,"2017-01-01","2022-12-31",S2BANDS, SCBURNING_FOLDER)  
   
# SECOND STEP: PREPROCESS THE IMAGES
## Iterate over the folders
tif_files = [x for folder in SCBURNING_FOLDER.iterdir() if folder.is_dir() 
             and folder.name != "reference" for x in folder.rglob("*.tif")]
len(tif_files) # 15,132 tif files

## Filter the black and cloudly images
optimized_files = omit_black_files(tif_files, model)
len(optimized_files) # 74 for 79 ROIS dont have black areas in all images (10,149 tif files)

## Group files by ROI and sort by date
roi_groups = defaultdict(list)
for file in optimized_files:
    roi_groups[file.parent.stem].append(file)

# Sort each group by date
for roi, files in roi_groups.items():
    files.sort(key=lambda x: pd.to_datetime(x.stem.split("_")[2]))

burning_files = dbadi_filter(roi_groups)    
len(burning_files) # 3,206 tif files

## Save as CSV and plot the data
burning_pixels_df = pd.DataFrame(burning_files, columns=["file", "burning_pixels"])
burning_pixels_df["ha"] = burning_pixels_df["burning_pixels"] * 100 / 10000

x = burning_pixels_df.ha.values
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(x=x, ax=ax_box, width=0.5, fill=True, color="lightblue")
# Overlay the outline points using stripplot with filled markers
sns.histplot(x=x, bins=40, ax=ax_hist, color="lightblue")

# Mean and median
median_value = burning_pixels_df["ha"].median()    
ax_hist.axvline(median_value, color='green', linestyle='--', linewidth=1, 
                label=f'Median: {median_value:.2f} ha')

mean_value = burning_pixels_df["ha"].mean()
ax_hist.axvline(mean_value, color='red', linestyle='--', linewidth=1, 
                label=f'Mean: {mean_value:.2f} ha')

# Add a legend
ax_hist.legend()
ax_box.set(yticks=[])

# Remove the top and right spines from plot(s)
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)

ax_hist.set_xlabel("Pseudo burned areas (ha)")
ax_hist.set_ylabel("Number of images")

plt.savefig("data/img/boxplot_burning.png", dpi=300, bbox_inches="tight")
plt.close()
plt.clf()

## Divide the data in three groups: "Lower Whisker", "Lenght Box", "Upper Whisker"
a = burning_pixels_df["ha"]
percentiles = np.percentile(a, [0, 25, 75, 100])
burning_pixels_df["group"] = pd.cut(a, percentiles, 
                                    labels=["Lower Whisker", "Lenght Box", 
                                            "Upper Whisker"])

## Add ROI column
burning_pixels_df["roi"] = burning_pixels_df["file"].apply(lambda x: x.parent.stem)
burning_pixels_df = burning_pixels_df[["roi", "file", "burning_pixels", "ha", 
                                       "group"]]

## Create tables for each group
lower_whisker = burning_pixels_df.query("group == 'Lower Whisker'") # 799 images
lenght_box = burning_pixels_df.query("group == 'Lenght Box'") # 1,594 images
upper_whisker = burning_pixels_df.query("group == 'Upper Whisker'") # 802 images

## Save the tables
lower_whisker.to_csv("data/table/lower_whisker.csv", index=False)
lenght_box.to_csv("data/table/length_box.csv", index=False)
upper_whisker.to_csv("data/table/upper_whisker.csv", index=False)

## Generate visualizations for each file in lenght box
df_n = pd.read_csv("data/table/length_box.csv", sep=";")
for i, row in df_n.iterrows():
    vis_image(path=pathlib.Path(row["file"]), factor_list=[1.75]*4, 
              out_dir="D:/length_box")
    print(f"[{i+1}/{len(df_n)}]:Processed {row['file']}")

## After manually review the images, we can get 1,250 images with burning
df_n.fillna("-", inplace=True)
final_df = df_n[df_n["analyse"] == 1]

## Group by ROI
final_df.groupby("roi").size().sort_values(ascending=False) # 66 of 74 ROIS

# THIRD STEP: CREATE A IRIS DIRECTORY
## Create a reference folder
REFERENCE_FOLDER = pathlib.Path("D:/scburning/reference")
REFERENCE_FOLDER.mkdir(exist_ok=True, parents=True)
## Generate a composite as reference
create_composite(roi_groups, REFERENCE_FOLDER)

# FOURTH STEP: SAVE THE IMAGES FROM THE IRIS DIRECTORY
## Order according format: name/S2, name/S2_ref, name/label
IRIS_FOLDER = pathlib.Path("D:/iris")
create_iris_dir(final_df, IRIS_FOLDER, SCBURNING_FOLDER)

## After labeling the images, we create a new folder with the input and target images
DATASET_FOLDER = pathlib.Path("D:/dataset/database")
labeled_df = pd.read_csv("data/table/final_burning.csv", sep=";")
labeled_df = labeled_df[labeled_df["final_mask"] == 1]

counter = 0
for path in labeled_df["id"].values:
    IRISsave(IRIS_FOLDER, DATASET_FOLDER, path) 
    counter += 1
    print("[%d/%d]" % (counter,len(labeled_df)))

# FIFTH STEP: CREATE A STRUCTURED DATASET WITH OTHER VARIABLES
## Open the folder with input and target images
S2_FOLDER = DATASET_FOLDER / "S2"
TARGET_FOLDER = DATASET_FOLDER / "target"

## Create the folders
NBR_FOLDER = DATASET_FOLDER / "nbr"
BADI_FOLDER = DATASET_FOLDER / "badi"
SLOPE_FOLDER = DATASET_FOLDER / "slope"
NDVI_FOLDER = DATASET_FOLDER / "ndvi"
NDWI_FOLDER = DATASET_FOLDER / "ndwi"
LANDCOVER_FOLDER = DATASET_FOLDER / "dlc"

NBR_FOLDER.mkdir(exist_ok=True)
NDVI_FOLDER.mkdir(exist_ok=True)
NDWI_FOLDER.mkdir(exist_ok=True)
BADI_FOLDER.mkdir(exist_ok=True)
SLOPE_FOLDER.mkdir(exist_ok=True)
LANDCOVER_FOLDER.mkdir(exist_ok=True)

## Iterate over the files
s2_files = sorted(list(S2_FOLDER.glob("*.tif")))
target_files = sorted(list(TARGET_FOLDER.glob("*.tif")))


for i, file in enumerate(s2_files):
    with rio.open(file) as src:
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
        print(f"[{i+1}/{len(s2_files)}]: Processed {file.name}")
        

for i, file in enumerate(target_files):
    with rio.open(file) as src:
        # From Affine meta, get the center of the image
        affine = src.transform
        center_x = affine[2] + affine[0] * src.width / 2
        center_y = affine[5] + affine[4] * src.height / 2

        crs = src.crs.to_string()

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

        # Save the slope
        with rio.open(LANDCOVER_FOLDER / file.name, "w", **profile) as dst:
            dst.write(slope_lc[0][0], 1)
        
        # Get the land cover
        with rio.open(SLOPE_FOLDER / file.name, "w", **profile) as dst:
            dst.write(slope_lc[0][1], 1)

        print(f"[{i+1}/{len(target_files)}]: Processed {file.name}")


## Create table for the GBM model
nbr_files = sorted(list(NBR_FOLDER.glob("*.tif")))
badi_files = sorted(list(BADI_FOLDER.glob("*.tif")))
slope_files = sorted(list(SLOPE_FOLDER.glob("*.tif")))
ndvi_files = sorted(list(NDVI_FOLDER.glob("*.tif")))
ndwi_files = sorted(list(NDWI_FOLDER.glob("*.tif")))
landcover_files = sorted(list(LANDCOVER_FOLDER.glob("*.tif")))

tensor_files = list(zip(s2_files, nbr_files, badi_files, 
                        slope_files, ndvi_files, ndwi_files,
                        landcover_files, target_files))

table = []
for i, zip_file in enumerate(tensor_files):
    S2, nbr, badi, slope, dlc, target = zip_file
    filename = pathlib.Path(S2).name

    with rio.open(S2) as src1, rio.open(nbr) as src2, rio.open(badi) as src3, \
        rio.open(slope) as src4, rio.open(dlc) as src5, rio.open(target) as src6:

        s2_bands = src1.read().astype(np.int16) / 10000
        other_bands = np.stack([src2.read(1), src3.read(1), src4.read(1), src5.read(1)], 
                                axis=0).astype(np.float32)
            
        # If other bands has nan values, replace them with -5
        other_bands = np.nan_to_num(other_bands, nan=-5)
        full_bands = np.concatenate((s2_bands, other_bands), axis=0)

        x_2d = einops.rearrange(full_bands, 'c h w -> c (h w)')  
        target = src6.read(1)
        mask_2d = einops.rearrange(target != -99, 'h w -> (h w)')

        arr = x_2d[:, mask_2d.nonzero()[0]].T  

        table.extend(arr)
        print(f"[{i+1}/{len(tensor_files)}]: Processed {file.name}")
    
## Create dataframe and save it as CSV
gbm_df = pd.DataFrame(table, columns=["S2_B02", "S2_B03", "S2_B04", "S2_B05", "S2_B06", "S2_B07",
                                      "S2_B08", "S2_B8A", "S2_B11", "S2_B12", "NBR", "BADI", 
                                      "SLOPE", "NDVI", "NDWI", "DIST_CROPS_COVER", "TARGET"])   

gbm_df.to_csv(DATASET_FOLDER/"gbm.csv", index=False)


## Split the data
X_train, X_test, y_train, y_test = train_test_split(gbm_df[:,:-1], gbm_df[:,-1],
                                                    test_size=0.2, random_state=42)

## Save the data
gbm_train = pd.concat([X_train, y_train], axis=1)
gbm_train.to_csv(DATASET_FOLDER/"gbm_train.csv", index=False)

gbm_test = pd.concat([X_test, y_test], axis=1)
gbm_test.to_csv(DATASET_FOLDER/"gbm_test.csv", index=False)

## Train a model using gbm_train.csv
from model import lgb_model
lgb_model(DATASET_FOLDER/"gbm_train.csv")

lgb_threshold = 0.1335276980033621
gbm_path = "models/gbm"
FUSED_TARGET_FOLDER = DATASET_FOLDER / "gbm"
FUSED_TARGET_FOLDER.mkdir(exist_ok=True)

## Fuse target and gbm probabilities
for i, zip_file in enumerate(tensor_files):
    S2, nbr, badi, slope, dlc, target = zip_file
    filename = pathlib.Path(S2).name

    with rio.open(S2) as src1, rio.open(nbr) as src2, rio.open(badi) as src3, \
        rio.open(slope) as src4, rio.open(dlc) as src5, rio.open(target) as src6:

        s2_bands = src1.read().astype(np.int16) / 10000
        other_bands = np.stack([src2.read(1), src3.read(1), src4.read(1), src5.read(1)], 
                                axis=0).astype(np.float32)
            
        # If other bands has nan values, replace them with -5
        other_bands = np.nan_to_num(other_bands, nan=-5)
        full_bands = np.concatenate((s2_bands, other_bands), axis=0)
        
        # Get the probabilities
        gbm_pred = make_predict_ml(full_bands, gbm_path, nmodels=10)
        gbm_pred = (gbm_pred >= lgb_threshold).astype(np.uint16)

        target = src6.read(1)

        # Fuse the probabilities with the target
        gbm_pred[target == 1] = 1
        gbm_pred[target == 0] = 0

        # Save the probabilities
        profile = src6.profile.copy()
        profile.update(count=1, dtype=np.uint16)

        
        with rio.open(FUSED_TARGET_FOLDER / filename, "w", **profile) as dst:
            dst.write(gbm_pred, 1)

        print(f"[{i+1}/{len(tensor_files)}]: Processed {filename}")


# SIXTH STEP: RUN THE LR META MODEL
## Load the checkpoint path
checkpoint_path = "checkpoints/resnet50_norm_075.ckpt"
predict_bands = list(zip(s2_files, nbr_files, badi_files, slope_files, 
                         ndvi_files, ndwi_files, landcover_files)) 
config = "config.yaml"
gbm_path = "models/gbm"

out_folder = DATASET_FOLDER / "probabilities"
out_folder.mkdir(exist_ok=True)

## Generate the new dataset probabilities
for i in range(0,len(predict_bands)):
    save_ensemble_prob_models(checkpoint_path, gbm_path, 
                              predict_bands[i], config, out_folder)
    
    print(f"[{i+1}/{len(predict_bands)}]: Processed image")

## Run LR meta model
prob_files = sorted(list(out_folder.glob("*.tif")))
data = []
for i, (target_file, prob_file) in enumerate(zip(target_files, prob_files)):
    with rio.open(prob_file) as src, rio.open(target_file) as tgt:
        array = src.read()   
        target = tgt.read(1)
        new_arr = np.concatenate([array, target[None]], axis=0)
        x_2d = einops.rearrange(new_arr, 'c h w -> c (h w)')
        mask_2d = einops.rearrange(new_arr[-1,:,:] != -99, 'h w -> (h w)')
        arr = x_2d[:, mask_2d.nonzero()[0]].T        
        ## Extend arr to create a table data
        data.extend(arr)
        print(f"[{i+1}/{len(prob_files)}]: Processed {prob_file.name}")

lr_df = pd.DataFrame(data, columns = ["Unet", "LightGBM","target"])
lr_df.to_csv(DATASET_FOLDER/"prob_dataset.csv", index=False)

## Split the data
lr_df = pd.read_csv(DATASET_FOLDER/"prob_dataset.csv")
X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(lr_df[["Unet", "LightGBM"]], 
                                                        lr_df["target"],
                                                        test_size=0.2, random_state=42)

## Save the data
lr_train = pd.concat([X_ptrain, y_ptrain], axis=1)
lr_train.to_csv(DATASET_FOLDER/"lr_train.csv", index=False)

lr_test = pd.concat([X_ptest, y_ptest], axis=1)
lr_test.to_csv(DATASET_FOLDER/"lr_test.csv", index=False)

## Train a model using lr_train.csv
stacking_clf = stacking_classifier()
stacking_clf.fit(X_ptrain, y_ptrain)

## Save the model
joblib.dump(stacking_clf, "models/stacking_model.pkl")

## Evaluate the model
model = joblib.load("models/stacking_model.pkl")
y_test_pred = model.predict_proba(X_ptest)[:, 1]
evaluate_metrics_ml(y_ptest, y_test_pred)


# SEVENTH STEP: DOWNLOAD THE IMAGES FOR OEFA VALIDATION AND PILOT AREA
## According to roi and date (in range of 3 days), filter by historical data
## Open the historical data
historical_data = gpd.read_file("data/vector/historical_data.geojson")

## Filter cloud and black images in a new folder
before = historical_data["FE_FECHA_EMERG"] - pd.Timedelta(days=7)
after = historical_data["FE_FECHA_EMERG"] + pd.Timedelta(days=7)

# Get tiles of 512*512 meters
data = align_points_to_grid(historical_data, 10, "FE_FECHA_EMERG")

# Create a new geodataframe
gdf = gpd.GeoDataFrame(historical_data[historical_data.columns[:-1]], 
                       geometry=data["geometry"])
gdf["id"] = ["emergency_" + str(i).zfill(4) for i in range(len(gdf))]

## Change the order of the columns
gdf = gdf[['id','TX_ADMINISTRADO', 'FE_FECHA_EMERG', 'TX_DESCRIPCION', 'NU_COORD_ESTE',
       'NU_COORD_NORTE', 'NU_COORD_ZONA', 'longitud', 'latitud', 'geometry']]

gdf.to_file("data/vector/historical_data_aligned.geojson", driver="GeoJSON")

EMERG_FOLDER = pathlib.Path("D:/dataset/emergencies")
for i, point in enumerate(gdf.itertuples()):

    ## get the image with the highest date
    before = point.FE_FECHA_EMERG - pd.Timedelta(days=7)
    before_str = before.strftime("%Y-%m-%d")
    after = point.FE_FECHA_EMERG + pd.Timedelta(days=7)
    after_str = after.strftime("%Y-%m-%d")

    # Download the cube data
    download_datacube_data(point, model, before_str, after_str, S2BANDS, EMERG_FOLDER)


## Download a PILOT area given a entire year
gpd_pilot = gpd.read_file("data/vector/laredo.geojson")
data_pilot = align_points_to_grid(gpd_pilot , 10, "id")

data_pilot["id"] = [f"laredo_{str(i).zfill(4)}" for i in range(len(data_pilot))]
## Create a new geodataframe
gdf_pilot = gpd.GeoDataFrame(data=data_pilot["id"], 
                             geometry=data_pilot["geometry"])


PILOT_FOLDER = pathlib.Path("D:/dataset/pilot")
for point in gdf_pilot.itertuples():
    # Download the cube data
    download_datacube_data(point, model, "2023-01-01", "2024-08-01", S2BANDS, PILOT_FOLDER)

# EIGHTH STEP: PLOTING THE RESULTS 
## For the different models ---------------------------------------------------
DATASET_FOLDER = pathlib.Path("D:/dataset/database")
IMG_FOLDER = pathlib.Path("data/img")
factor = 1.9
threshold = 0.75
filenames = [
    "ROI_0001_03_04__S2B_MSIL2A_20181213T153609_R068_T17MPP_20201008T095322",
    "ROI_0020_01__S2A_MSIL2A_20170907T155221_R111_T17MNQ_20210201T204538",
    "ROI_0097_04__S2A_MSIL2A_20210403T152641_R025_T17LQL_20210404T135846",
    "ROI_0011_01__S2B_MSIL2A_20180927T155219_R111_T17MNQ_20201009T053858",
    "ROI_0094_02__S2B_MSIL2A_20210322T153619_R068_T17MQM_20210323T110554",
]
models = ["Stacking", "U-Net", "GBDT"]

## For each model --------------------------------------------------------------
for i, model in enumerate(models):
    vis_results(DATASET_FOLDER, filenames, threshold, model, factor, IMG_FOLDER)
    print(f"[{i+1}/{len(models)}]: Processed {model}")

## For OEFA validation --------------------------------------------------------
from plots import vis_oefaval
import pathlib
import geopandas as gpd
EMERG_FOLDER = pathlib.Path("D:/dataset/emergencies")
VECTOR_FOLDER = pathlib.Path("data/vector")
IMG_FOLDER = pathlib.Path("data/img")
gdf = gpd.read_file("data/vector/historical_data_aligned.geojson")
factor_list = [1.75,1.75,2.5,1.75,2.5]
threshold_list = [0.2, 0.3, 0.50, 0.75, 0.90]
emerg_filenames = [
    "emergency_0016__S2B_MSIL2A_20220215T153609_R068_T17MQM_20220223T184134",
    "emergency_0028__S2B_MSIL2A_20220429T155219_R111_T17MNQ_20220430T092038",
    "emergency_0456__S2A_MSIL2A_20210506T153621_R068_T17MQM_20210508T060153",
    "emergency_0068__S2A_MSIL2A_20221117T153621_R068_T17MPM_20221118T061813",
    "emergency_0376__S2A_MSIL2A_20200511T153621_R068_T17MQM_20200917T183941",
    ]

for i, (factor, emerg_filename) in enumerate(zip(factor_list, emerg_filenames)):
    vis_oefaval(gdf, emerg_filename, EMERG_FOLDER, factor, 
                threshold_list, IMG_FOLDER, VECTOR_FOLDER) 
    print(f"[{i+1}/{len(emerg_filenames)}]: Processed {emerg_filename}")

## For PILOT area --------------------------------------------------------------
from plots import vis_time_series, vis_comparation
import pathlib
PILOT_FOLDER = pathlib.Path("D:/dataset/pilot")
IMG_FOLDER = pathlib.Path("data/img")
VECTOR_FOLDER = pathlib.Path("data/vector")
factor = 1.9
specific_files = [
    "laredo__S2A_MSIL2A_20230506T153621_R068_T17MQM_20230506T235102.tif",
    "laredo__S2B_MSIL2A_20231107T153619_R068_T17MQM_20231107T211537.tif",
    "laredo__S2B_MSIL2A_20240326T153629_R068_T17MQM_20240326T225923.tif",
]
thresholds_pilot1 = [0.85, 0.98, 0.85, 0.75, 0.75]
thresholds_pilot2 = [0.85, 0.80, 0.80, 0.80, 0.80]
thresholds_pilot3 = [0.85, 0.75, 0.75, 0.80, 0.80]

threshold_union = [thresholds_pilot1, thresholds_pilot2, thresholds_pilot3]

for i, (threshold, file) in enumerate(zip(threshold_union, specific_files)):
    vis_time_series(PILOT_FOLDER, file, factor, IMG_FOLDER, threshold, 
                    VECTOR_FOLDER)
    print(f"[{i+1}/{len(specific_files)}]: Processed {file}")

## For the different models ---------------------------------------------------
model_filenames = [
    "emergency_0475__S2A_MSIL2A_20211122T153621_R068_T17MQM_20211123T071242",
    "emergency_0036__S2A_MSIL2A_20220531T153631_R068_T17MQM_20220601T165748",
    "emergency_0495__S2A_MSIL2A_20210409T155221_R111_T17MMQ_20210411T023449",
    "emergency_0016__S2B_MSIL2A_20220215T153609_R068_T17MQM_20220223T184134",
    "emergency_0475__S2A_MSIL2A_20211122T153621_R068_T17MQM_20211123T071242"
]

model_thresholds = [0.15, 0.75, 0.15, 0.20]
factor = 1.9

for i, file in enumerate(model_filenames):
    vis_comparation(file, gdf, EMERG_FOLDER, factor, model_thresholds, IMG_FOLDER) 
    print(f"[{i+1}/{len(model_filenames)}]: Processed {file}")
