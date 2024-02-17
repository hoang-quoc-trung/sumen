import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.dataset import image_augmentation


class Sumen_Dataset(Dataset):
    def __init__(
        self,
        df,
        phase,
        root_dir,
        tokenizer,
        processor,
        image_size,
        max_length=512,
    ):
        self.df = df
        self.root_dir = root_dir
        self.phase = phase 
        self.tokenizer= tokenizer
        self.processor = processor
        self.image_size = image_size
        self.max_length = max_length
        self.train_transform = image_augmentation.train_transform()
        
    def __len__(self,):
        return len(self.df)
    
    def __getitem__(self, idx):
        latex_sequence = self.df.latex.iloc[idx]
        image = self.df.image_filename.iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        
        # Image augmentation
        if self.phase == 'train':
            img = cv2.imread(img_path)
            if(len(img.shape)<3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.train_transform(im=img)
            img = Image.fromarray(img)
        else:
            img = Image.open(img_path).convert("RGB")
                
        try:
            pixel_values = self.processor(
                images=img,
                return_tensors="pt",
                data_format="channels_first"
            ).pixel_values
        except ZeroDivisionError: 
            # Image error, empty latex in image
            print("Error in image:", image)
            pixel_values = torch.zeros((3, self.image_size[0], self.image_size[1]))
            latex_sequences = ""
    
        latex_sequences = self.tokenizer(
            latex_sequence,
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        ).input_ids
        latex_sequences = [eq if eq != self.tokenizer.pad_token_id else -100 for eq in latex_sequences]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(latex_sequences)}
        return encoding