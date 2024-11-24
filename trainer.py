import typer 
import pytorch_lightning as pl
from dataloader import SCBurningDataModule
from model import lit_model
import pathlib
import torch
from utils import load_config

# Set the seed for reproducibility
from sklearn.model_selection import train_test_split
torch.random.manual_seed(42)

# Set the precision
torch.set_float32_matmul_precision('high')

# Create the app for the training
app = typer.Typer()
@app.command()    

def train(config_path: str = typer.Option(..., "--config", "-c", 
                                          help="Path to the config YAML file")):
    # Define the dataset folder
    config = load_config(config_path)
    train_config = config["train_config"]
    DATASET_FOLDER = pathlib.Path(train_config["dataset_path"])

    # Open file directories
    S2_FOLDER = DATASET_FOLDER / "S2"
    s2_files = sorted(list(S2_FOLDER.glob("*.tif")))

    NBR_FOLDER = DATASET_FOLDER / "nbr"
    nbr_files = sorted(list(NBR_FOLDER.glob("*.tif")))
 
    BADI_FOLDER = DATASET_FOLDER / "badi"    
    badi_files = sorted(list(BADI_FOLDER.glob("*.tif")))

    NDVI_FOLDER = DATASET_FOLDER / "ndvi"
    ndvi_files = sorted(list(NDVI_FOLDER.glob("*.tif")))

    NDWI_FOLDER = DATASET_FOLDER / "ndwi"
    ndwi_files = sorted(list(NDWI_FOLDER.glob("*.tif")))

    SLOPE_FOLDER = DATASET_FOLDER / "slope"
    slope_files = sorted(list(SLOPE_FOLDER.glob("*.tif")))

    DLC_FOLDER = DATASET_FOLDER / "dlc"
    dlc_files = sorted(list(DLC_FOLDER.glob("*.tif")))

    # Get the target files
    target_files = sorted(list((DATASET_FOLDER / "gbm").glob("*.tif")))
 
    # Zip the files in this order
    files = list(zip(s2_files, nbr_files, badi_files, slope_files, 
                    ndvi_files, ndwi_files, dlc_files, target_files))

    # Split the dataset
    ttrain, test_files = train_test_split(files, 
                        train_size=train_config["train_test_split"], 
                        random_state=42)
    
    train_files, val_files = train_test_split(ttrain, 
                            train_size=train_config["train_val_split"], 
                            random_state=42)

    # Create dataloader
    datamodule = SCBurningDataModule(train= train_files, 
                                     val= val_files, 
                                     test= test_files,
                                     batch_size=train_config["batch_size"],
                                     num_workers=train_config["num_workers"])
    datamodule.setup()

    # Initialize the model
    model = lit_model(config=config)

    # Initialize the callbacks
    early_stopping = pl.callbacks.EarlyStopping(        
        monitor="val_loss",
        patience=10,
        mode="min"
    )
   
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename=f"{train_config['ckpt_name']}", 
        mode="min",
        save_top_k=1
    )

    callbacks = [early_stopping, checkpoint]

    logger = pl.loggers.WandbLogger(project="scburning")       

    # Initialize the trainer
    trainer = pl.Trainer(
            strategy="ddp",
            accelerator="gpu",
            devices=torch.cuda.device_count(),
            max_epochs=config["train_config"]["max_epochs"],
            callbacks=callbacks,
            precision="16-mixed",
            log_every_n_steps=20,
            logger=logger
        )

    # Train the model
    trainer.fit(model, datamodule=datamodule)
   
   # Test the model
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    # Run the training
    app()
