"""
Trainer
"""

from pytorch_lightning import Trainer, Callback
from dataset import EfficientDetDataModule, EfficientDetDataset
from model import EfficientDetModel
from vehicle_dataset import CarsDatasetAdaptor

import pandas as pd
import os


def train_cars(dataset_path):
    # dataset
    train_ds_path = os.path.join(dataset_path, 'training_images')
    df = pd.read_csv(os.path.join(dataset_path, 'train_solution_bounding_boxes (1).csv'))

    train_ds = CarsDatasetAdaptor(train_ds_path, df)

    dataset = EfficientDetDataModule(train_ds=train_ds, val_ds=train_ds)

    # model
    model = EfficientDetModel(
        num_classes=1,
        img_size=512,
        model_arc='tf_efficientnetv2_b0',
        pretrained=False)

    # trainer
    trainer = Trainer(accelerator='cpu', max_epochs=5, num_sanity_val_steps=1)
    trainer.fit(model=model, datamodule=dataset)

if __name__ == '__main__':
    train_cars('./data/vehicle')