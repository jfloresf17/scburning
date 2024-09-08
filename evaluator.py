import lightgbm as lgb
import einops
import pathlib
import numpy as np
import torch
from model import lit_model
import rasterio as rio
import yaml
import joblib
import rasterio as rio
from typing import List
import numpy as np
from sklearn.metrics import (log_loss, roc_curve, auc, 
                             f1_score, recall_score, 
                             precision_score, jaccard_score, 
                             cohen_kappa_score)


# Gradient Boosting Machine (GBM)
def make_predict_ml(tensor: np.ndarray, 
                    model_path: str, 
                    nmodels: int=10
) -> np.ndarray:
    """
    Make prediction with a Gradient Boosting Machine model.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor with shape (c, h, w).
    model_path : str
        Path to the folder with the models.
    nmodels : int
        Number of models to use for the ensemble.
    
    Returns
    -------
    np.ndarray
        Prediction with shape (h, w).
    """
    # choose randomly between the 0-9 models
    nvalue = np.random.choice(range(10), nmodels, replace=False)

    # Combine al models
    models = []
    for i in nvalue:
        models.append(lgb.Booster(model_file=model_path + f"/model_{str(i)}" + ".txt"))

    # Make prediction
    x_2d = einops.rearrange(tensor, 'c h w -> c (h w)')
    y_pred = []
    for model in models:
        y_hat_1d = model.predict(x_2d.T, num_iteration=model.best_iteration)
        y_hat_2d = einops.rearrange(y_hat_1d, '(h w) -> h w', 
                                    h=tensor.shape[1], 
                                    w=tensor.shape[2]
                                    )
        y_pred.append(y_hat_2d)
    
    return np.min(y_pred, axis=0)


# Open the pre-trained model
def load_model(config: str, 
               checkpoint_path: str
) -> lit_model:
    """
    Load the pre-trained model.

    Parameters
    ----------
    config : str    
        Path to the configuration file.
    checkpoint_path : str
        Path to the checkpoint file.
    
    Returns
    -------
    lit_model
        The model with the pre-trained weights.
    """
    model = lit_model(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_predict_tensor(image_path: List[pathlib.Path]
) -> np.ndarray:
    """
    Get the input tensor for the deep learning model.

    Parameters
    ----------
    image_path : List[pathlib.Path]
        List with the paths to the images.

    Returns
    -------
    np.ndarray
        The input tensor for the deep learning model with 
        shape (c, h, w).   
    """
    # Load the images paths
    s2, nbr, badi, slope, ndvi, ndwi, dlc = image_path

    # Open the files
    with rio.open(s2) as src1, rio.open(nbr) as src2, \
        rio.open(badi) as src3, rio.open(slope) as src4, \
        rio.open(ndvi) as src5, rio.open(ndwi) as src6, \
        rio.open(dlc) as src7:

        ## Usefull bands for Sentinel-2
        usefull_bands = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

        ## Read the arrays and create the input tensor
        bands_src = src1.read(usefull_bands).astype(np.int16) / 10000

        tensor = np.concatenate([bands_src, src2.read(), src3.read(), src4.read(), 
                                    src5.read(), src6.read(), src7.read()], axis=0)
        
        fill_tensor = np.nan_to_num(tensor, nan=-5)

    return fill_tensor

# Get the input tensor
def preprocess_image(image_path: List[pathlib.Path], 
                     config: str
) -> torch.Tensor:
    """
    Get the input tensor for the deep learning model.

    Parameters
    ----------
    image_path : List[pathlib.Path]
        List with the paths to the images.

    config : str
        Path to the configuration file.
    
    Returns 
    -------
    torch.Tensor    
        The input tensor for the deep learning model with shape (1, c, h, w).

    """
    # Open configuration file
    with open(config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load the images paths
    fill_tensor = get_predict_tensor(image_path)

    # Create the input tensor
    input = torch.from_numpy(fill_tensor).float()

    # Load the arrays
    std_arr = np.asarray(config["model_config"]["std"])
    mean_arr = np.asarray(config["model_config"]["mean"])

    ## Normalize the input
    if config["dataset_config"]["normalize"] == True:
        std = torch.tensor(std_arr, dtype=torch.float32).view(-1, 1, 1)
        mean = torch.tensor(mean_arr, dtype=torch.float32).view(-1, 1, 1)
        input = (input - mean) / std 

    return input.unsqueeze(0)  # Añadir dimensión de batch


# Make prediction with the deep learning model (DL)
def make_predict_dl(config: str,
                    image_path: List[pathlib.Path], 
                    checkpoint_path: str
) -> np.ndarray:
    """
    Make prediction with the deep learning model.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    
    image_path : List[pathlib.Path]
        List with the paths to the images.

    checkpoint_path : str
        Path to the checkpoint file.

    Returns
    -------
    np.ndarray
        Prediction with shape (h, w).    
    """
    # Load the model with the pre-trained weights
    model = load_model(config, checkpoint_path)
    
    # Get the input tensor
    input_tensor = preprocess_image(image_path, config).to(model.device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output).float()

    # Convert the tensor to numpy
    pred = probabilities.squeeze().cpu().numpy()
    return pred


def save_ensemble_prob_models(ckpt_path: str,
                              gbm_path: str, 
                              image_paths: List[pathlib.Path], 
                              config: str, 
                              out_folder: pathlib.Path
) -> None:
    """
    Save the ensemble probabilities in a new tensor with shape (2, h, w).

    Parameters
    ----------
    ckpt_path : str
        Path to the deep learning model checkpoint.
    gbm_path : str
        Path to the folder with the GBM models.
    image_paths : List[pathlib.Path]
        List with the paths to the images.
    config : str
        Path to the configuration file in deep learning model.
    out_folder : pathlib.Path
        Path to the folder where the tensor will be saved

    Returns
    -------
    None
    """

    # Get the first image path
    S2, *_ = image_paths

    # Load the images paths
    filename = pathlib.Path(S2).name

    # Deep learning model
    wildfire_prob_01 = make_predict_dl(config, image_paths, ckpt_path)

    # Get the input tensor for the GBM model
    fill_tensor = get_predict_tensor(image_paths) 

    # Probabilities 
    wildfire_prob_02 = make_predict_ml(fill_tensor, gbm_path, nmodels=10)    

    # Create a new tensor with the probabilities
    stacked_tensor = np.stack([wildfire_prob_01, wildfire_prob_02], 
                    axis=0).astype(np.float32)        
   
    # Save the tensor
    with rio.open(S2) as src1:
        profile = src1.profile.copy()
        profile.update(count=2, dtype=rio.float32)
        out_folder.mkdir(parents=True, exist_ok=True)
    
        with rio.open(out_folder/filename, "w", **profile) as dst:
            dst.write(stacked_tensor)        


# Make prediction with the logistic regression model (LR)
def make_predict_lr(tensor: np.ndarray,
                    model_path: str
) -> np.ndarray:
    """
    Make prediction with the logistic regression model.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor with shape (2, h, w).
    model_path : str
        Path to the model file.
    
    Returns
    -------
    np.ndarray
        Prediction with shape (h, w).
    """
    # Load model from pkl file
    model = joblib.load(model_path)

    # Reshape the tensor to match the expected input format (h*w, c)
    x_2d = einops.rearrange(tensor, 'c h w -> (h w) c')
    
    # Make prediction
    y_hat_1d = model.predict_proba(x_2d)[:, 1]

    # Reshape back to the original image dimensions
    y_hat_2d = y_hat_1d.reshape(tensor.shape[1], tensor.shape[2])
  
    return y_hat_2d


def evaluate_metrics_ml(y_test, y_pred_proba):
    logloss = log_loss(y_test, y_pred_proba)

    ## Evaluate the model
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ## Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    ## Metrics
    f1 = f1_score(y_test, y_pred_optimal)
    recall = recall_score(y_test, y_pred_optimal)
    precision = precision_score(y_test, y_pred_optimal)
    jaccard = jaccard_score(y_test, y_pred_optimal)
    kappa = cohen_kappa_score(y_test, y_pred_optimal)

    print("################## METRICS IN TEST DATASET ##################")
    print(f"The optimal threshold is: {optimal_threshold:.3f}")
    print(f"Log Loss: {logloss:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"IoU: {jaccard:.3f}")
    print(f"Cohen Kappa: {kappa:.3f}")
    print("##############################################################")