import torch
import os
import polars
import pathlib
import wandb

import lightgbm as lgb
import sklearn.model_selection as skms
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from sklearn.linear_model import LogisticRegression
from wandb.integration.lightgbm import wandb_callback, log_summary
import torchmetrics

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Define a machine learning model
# set parameters, bagging = bootstrap aggregation (make predictions based on multiple models)

# Set token wandb login
os.environ["WANDB_API_KEY"] = "317cfe042e74632a8151c8e0bfb620d64a847da3"

params = {
    'task': 'train', # For training
    'boosting_type': 'gbdt', # Gradient Boosting Decision Tree
    'objective': 'binary', # Binary target feature
    'metric': {'l2', 'auc', 'binary_logloss'}, #Square loss, Area under curve, Binary log loss
    'metric_freq': 1, # Metric output frequency
    'num_leaves': 31, # Maximum number of leaves in one tree
    'data_sample_strategy': 'goss', # Data sampling strategy
    'learning_rate': 0.5, # Learning rate, controls size of a gradient descent step
    'feature_fraction': 0.9, # LightGBM will randomly select part of features (columns) in a dataset.
    'verbose': 1, # For printing information
    'device' : 'gpu', # You can use GPU using 'gpu'
    'gpu_device_id' : 0, # GPU device ID
}

# Define LightGBM model function using 10-fold cross validation
def lgb_model(config, params=params,  n_fold=10):
    # Initialize wandb
    wandb.init(project="scburning", name="lightgbm", config=params)

    # Load dataset
    csv = config['train_config']['csv_path']
    dataset = polars.read_csv(csv)

    # from polars to torch
    dataset_tensor = torch.from_numpy(dataset.to_numpy())

    feature_names = ["S2_B02", "S2_B03", "S2_B04", "S2_B05", "S2_B06", "S2_B07",
                     "S2_B08", "S2_B8A", "S2_B11", "S2_B12", "NBR", "BADI", 
                     "SLOPE", "NDVI", "NDWI", "DIST_CROPS_COVER"]

    # Split the dataset in train and val 10 times (10-fold cross validation)
    skf = skms.StratifiedKFold(n_fold, shuffle=True, random_state=42)
    X = dataset_tensor[:, :-1]
    y = dataset_tensor[:, -1]

    # split into train and test
    for index, (train_index, val_index) in enumerate(skf.split(X, y)):
        print("TRAIN:", train_index, "TEST:", val_index)
        # Split the dataset in train and test
        X_train, X_test = X[train_index].numpy(), X[val_index].numpy()
        y_train, y_test = y[train_index].numpy().astype(int), y[val_index].numpy().astype(int)

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(data=X_train, label=y_train, feature_name= feature_names)
        lgb_eval = lgb.Dataset(data=X_test, label=y_test, reference=lgb_train, 
                               feature_name= feature_names)

        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=20,            
            valid_sets=lgb_eval,
            valid_names=('validation'),
            callbacks=[
                wandb_callback(),
                lgb.early_stopping(stopping_rounds=50)
            ]
        )

        # save model checkpoint in wandb
        log_summary(gbm, save_model_checkpoint=True)

        # Save the model
        MODELS_PATH = pathlib.Path("./models/gbm")
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        gbm.save_model("models/gbm/model_{}.txt".format(index))

 
# Define the class labels
class_labels = {0: "no burning", 1: "burning"}

# Define the dice loss function
def dice_loss(y_hat, y):
    """
    Calculates the dice loss between predicted and ground truth masks.

    Args:
        y_hat (torch.Tensor): Predicted mask tensor.
        y (torch.Tensor): Ground truth mask tensor.

    Returns:
        torch.Tensor: Dice loss value.
    """
    smooth = 1e-6
    y_hat = torch.sigmoid(y_hat).view(-1)
    y = y.view(-1)
    intersection = (y_hat * y).sum()
    union = y_hat.sum() + y.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice


# Define the Cohen Kappa metric
class BinaryCohenKappa:
    def __init__(self):
        self.reset()

    def update(self, preds, targets, threshold=0.5):
        preds = (preds > threshold).cpu().numpy().astype(int)
        targets = targets.detach().cpu().numpy().astype(int)

        if self.y_pred is None or self.y_true is None:
            self.y_pred = preds.flatten()
            self.y_true = targets.flatten()
        else:
            self.y_pred = np.concatenate((self.y_pred, preds.flatten()))
            self.y_true = np.concatenate((self.y_true, targets.flatten()))

    def compute(self):
        return cohen_kappa_score(self.y_true, self.y_pred)

    def reset(self):
        self.y_true = None
        self.y_pred = None


# Define a deep learning model
class lit_model(pl.LightningModule):
    def __init__(self, config):
        super(lit_model, self).__init__()
        self.save_hyperparameters()

        model_config = config["model_config"]
        train_config = config["train_config"]    

        self.model = smp.Unet(
                    encoder_name=model_config["encoder_name"],   
                    encoder_weights=model_config["encoder_weights"],
                    in_channels=model_config["in_channels"],
                    classes=model_config["classes"]
                )

        self.lr = train_config["lr"]
        self.alpha = model_config["alpha"]
        self.weight = model_config["weight"]
        self.threshold = train_config["threshold"]

        self.weights = torch.tensor([self.weight], dtype=torch.float32).to(self.device)
        self.wce = torch.nn.BCEWithLogitsLoss(pos_weight=self.weights)

        self.train_f1 = torchmetrics.F1Score(task='binary', threshold=self.threshold)
        self.val_f1 = torchmetrics.F1Score(task='binary', threshold=self.threshold)
        self.test_f1 = torchmetrics.F1Score(task='binary', threshold=self.threshold)

        self.train_precision = torchmetrics.Precision(task='binary', threshold=self.threshold)
        self.val_precision = torchmetrics.Precision(task='binary', threshold=self.threshold)
        self.test_precision = torchmetrics.Precision(task='binary', threshold=self.threshold)

        self.train_recall = torchmetrics.Recall(task='binary', threshold=self.threshold)
        self.val_recall = torchmetrics.Recall(task='binary', threshold=self.threshold)
        self.test_recall = torchmetrics.Recall(task='binary', threshold=self.threshold)

        self.train_iou =  torchmetrics.JaccardIndex(task='binary', threshold=self.threshold)
        self.val_iou = torchmetrics.JaccardIndex(task='binary', threshold=self.threshold)
        self.test_iou = torchmetrics.JaccardIndex(task='binary', threshold=self.threshold)

        self.train_kappa = BinaryCohenKappa()
        self.val_kappa = BinaryCohenKappa()
        self.test_kappa = BinaryCohenKappa()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)       

        self.log('train_precision', self.train_precision(y_hat, y), sync_dist=True)
        self.log('train_recall', self.train_recall(y_hat, y), sync_dist=True)
        self.log('train_iou', self.train_iou(y_hat, y), sync_dist=True)
        self.log('train_f1', self.train_f1(y_hat, y), sync_dist=True)

        self.train_kappa.update(y_hat, y, threshold=self.threshold)
        kappa = self.train_kappa.compute()
        self.log('train_kapppa', kappa, sync_dist=True)
        self.train_kappa.reset()

        # Loss function
        wce_loss = self.wce(y_hat, y)
        dice = dice_loss(y_hat, y)

        # Weighted sum of the losses (weighted by alpha)
        combined_loss = (1 - self.alpha) * wce_loss + self.alpha * dice
        self.log('train_loss', combined_loss, sync_dist=True)
        return {'loss': combined_loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)   

        self.log('val_precision', self.val_precision(y_hat, y), sync_dist=True)
        self.log('val_recall', self.val_recall(y_hat, y), sync_dist=True)
        self.log('val_iou', self.val_iou(y_hat, y), sync_dist=True)
        self.log('val_f1', self.val_f1(y_hat, y), sync_dist=True)

        self.val_kappa.update(y_hat, y, threshold=self.threshold)
        kappa = self.val_kappa.compute()
        self.log('val_kappa', kappa, sync_dist=True)
        self.val_kappa.reset()

        # Loss function
        wce_loss = self.wce(y_hat, y)
        dice = dice_loss(y_hat, y)

        # Weighted sum of the losses (weighted by alpha)
        combined_loss = (1 - self.alpha) * wce_loss + self.alpha * dice
        self.log('val_loss', combined_loss, sync_dist=True)

        # Logging the first images of the batch
        if self.current_epoch % 10 == 0:  # Logging every 10 epochs
            # Get the RGB image from the 12 bands
            x = x[:, 0:3]
            x = x.permute(0, 2, 3, 1)
            y_hat = (torch.sigmoid(y_hat) > self.threshold).float()

            self.logger.experiment.log({
                "image": wandb.Image(x[0].cpu().detach().numpy()*255, masks={
                    "predictions": {
                        "mask_data": y_hat[0][0].cpu().detach().numpy(),
                        "class_labels": class_labels
                    },
                    "ground_truth": {
                        "mask_data": y[0][0].cpu().detach().numpy(),
                        "class_labels": class_labels
                    }
                })
            })
        return {'val_loss': combined_loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)         

        self.log('test_precision', self.test_precision(y_hat, y), sync_dist=True)
        self.log('test_recall', self.test_recall(y_hat, y), sync_dist=True)
        self.log('test_iou', self.test_iou(y_hat, y), sync_dist=True)
        self.log('test_f1', self.test_f1(y_hat, y), sync_dist=True)

        self.test_kappa.update(y_hat, y, threshold=self.threshold)
        kappa = self.test_kappa.compute()
        self.log('test_kapppa', kappa, sync_dist=True)
        self.test_kappa.reset()

        # Loss function
        wce_loss = self.wce(y_hat, y)
        dice = dice_loss(y_hat, y)

        # Weighted sum of the losses (weighted by alpha)
        combined_loss = (1 - self.alpha) * wce_loss + self.alpha * dice
        self.log('test_loss', combined_loss, sync_dist=True)
        return {'test_loss': combined_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, 
                                                    gamma=0.1)
        return [optimizer], [scheduler]


# Define the StackingProbClassifier class
class StackingProbClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, meta_model):
        self.meta_model = meta_model
    
    def fit(self, X, y):
        self.meta_model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.meta_model.predict(X)
    
    def predict_proba(self, X):
        return self.meta_model.predict_proba(X)


# Define the stacking_classifier function
def stacking_classifier():
    # Definir el meta-modelo
    meta_model = LogisticRegression(class_weight='balanced')
    # Crear el StackingProbClassifier
    stacking_clf = StackingProbClassifier(meta_model=meta_model)
    # Entrenar el StackingProbClassifier
    return stacking_clf