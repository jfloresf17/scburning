import torch
import rasterio as rio
import pytorch_lightning as pl
import numpy as np
import yaml


with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_config = config["model_config"]   
dataset_config = config["dataset_config"]
    
   
# Scburning dataset
class SCBurningDataset(torch.utils.data.Dataset):
    def __init__(self, files, normalize=False):
        self.files = files
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the files
        x, y = self.files[idx]

        s2, nbr, badi, slope, ndvi, ndwi, dlc = x

        # Load the arrays
        std_arr = np.asarray(model_config["std"])
        mean_arr = np.asarray(model_config["mean"])

        # Open the files
        with rio.open(s2) as src1, rio.open(nbr) as src2, \
             rio.open(badi) as src3, rio.open(slope) as src4, \
             rio.open(ndvi) as src5, rio.open(ndwi) as src6, \
             rio.open(dlc) as src7, rio.open(y) as tgt:

            if config["model_config"]["in_channels"] == 16:
                try:
                    ## Usefull bands for Sentinel-2
                    usefull_bands = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

                    ## Read the arrays and create the input tensor
                    bands_src = src1.read(usefull_bands).astype(np.int16) / 10000

                    tensor = np.concatenate([bands_src, src2.read(), src3.read(), src4.read(), 
                                            src5.read(), src6.read(), src7.read()], axis=0)
                    
                    fill_tensor = np.nan_to_num(tensor, nan=-5)
                    input = torch.from_numpy(fill_tensor).float()

                    ## Normalize the input
                    if self.normalize == True:
                        std = torch.tensor(std_arr, dtype=torch.float32).view(-1, 1, 1)
                        mean = torch.tensor(mean_arr, dtype=torch.float32).view(-1, 1, 1)
                        input = (input - mean) / std 

                    label = tgt.read(1).astype(np.int16)     
                    target = torch.from_numpy(label).float().unsqueeze(0)   
                
                except Exception as e:
                    raise Exception(f"The number of channels is not correct: {e}")
                
        return input, target


class SCBurningDataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, 
                 batch_size=32,
                 num_workers=16,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):        
        
        if stage == 'fit' or stage is None:
            self.train_dataset = SCBurningDataset(self.train, 
                                                  normalize=dataset_config["normalize"])
            self.val_dataset = SCBurningDataset(self.val, 
                                                normalize=dataset_config["normalize"])

        if stage == 'test' or stage == 'predict':
            self.test_dataset = SCBurningDataset(self.test, 
                                                 normalize=dataset_config["normalize"])


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    


