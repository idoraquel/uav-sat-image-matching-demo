from torchvision.models import resnet18, resnet34
from torch import nn
from torchvision.transforms.v2 import Resize, Compose
from torchvision.transforms import ToTensor, Normalize, CenterCrop
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from tqdm import tqdm


class PatchedImagesDataset(Dataset):
        def __init__(self, sat_images, uav_images, patch_size):
            self.sat_images = sat_images
            self.uav_images = uav_images
            self.patch_size = patch_size  # should match the feature extractor dimensions
            self.transform = Compose([
                Resize(1120),  # 5 patches by x 224. Smaller size (height)
                CenterCrop(1120),  # make it square for convenience
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.patch_id = 0
        
        def __len__(self):
            return min(len(self.sat_images), len(self.uav_images)) * 25
        
        def __getitem__(self, idx):
            image_id = idx // 25
            self.patch_id = idx % 25
            if self.patch_id == 0:
                self.current_sat_image = self.transform(Image.open(self.sat_images[image_id]))
                self.current_uav_image = self.transform(Image.open(self.uav_images[image_id]))
            return self.extract_patches()
        
        def extract_patches(self):
            x1 = (self.patch_id) // 5 * self.patch_size
            y1 = (self.patch_id) % 5 * self.patch_size
            x2 = x1 + self.patch_size
            y2 = y1 + self.patch_size
            self.patch_id += 1
            sat_patch, uav_patch = (im[:, x1:x2, y1:y2] for im in (self.current_sat_image, self.current_uav_image)) 
            
            return sat_patch, uav_patch


if __name__ == "__main__":

    import pandas as pd

    df_test = pd.read_csv("data/test/pairs_with_results.csv")

    # model = resnet18(weights="ResNet18_Weights.DEFAULT")
    model = resnet34(weights="ResNet34_Weights.DEFAULT")
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.cuda()
    
    
    sat_image_paths = df_test.loc[:, "sat_image"]
    uav_image_paths = df_test.loc[:, "uav_image"]

    # Define the patch size
    patch_size = 224

    # Create the dataset
    dataset = PatchedImagesDataset(sat_image_paths, uav_image_paths, patch_size)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False, num_workers=4)
    distances = []
    
    for sats, uavs in tqdm(dataloader):
        with torch.no_grad():
            sat_feat = feature_extractor(sats.cuda())
            uav_feat = feature_extractor(uavs.cuda())
            manh = torch.sum(torch.abs(sat_feat - uav_feat)).item()
            distances.append(manh)
    

    # df_test["resnet_18_manh"] = distances
    df_test["resnet_34_manh"] = distances
    df_test.to_csv("data/test/pairs_with_results.csv", index=False)