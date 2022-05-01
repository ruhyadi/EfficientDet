"""
Dataset module
"""

from matplotlib import image
import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader


class EfficientDetDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms):
        self.dataset = dataset_adaptor
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, bboxes, class_labels, image_id = self.dataset.get_data(index)

        sample = {
            'image': np.array(image, dtype=np.float32),
            'bboxes': bboxes,
            'labels': class_labels
        }

        # augmented sample
        if self.transforms is not None:
            sample = self.transforms(**sample)

        sample['bboxes'] = np.array(sample['bboxes'])
        image = sample['image']
        bboxes = sample['bboxes']
        labels = sample['labels']

        # new height and width due to augmentation
        _, new_h, new_w = image.shape
        # convert to yxyx
        sample['bboxes'][:, [0, 1, 2, 3]] = sample['bboxes'][:, [1, 0, 3, 2]]

        target = {
            'bboxes': torch.as_tensor(sample['bboxes'], dtype=torch.float32),
            'labels': torch.as_tensor(labels),
            'image_id': torch.tensor([image_id]),
            'img_size': (new_h, new_w),
            'img_scale': torch.tensor([1.0]),
        }

        return image, target, image_id

class EfficientDetDataModule(LightningDataModule):
    def __init__(
        self,
        train_ds,
        val_ds,
        train_tfms=None,
        val_tfms=None,
        num_workers=2,
        batch_size=16,
        ):
        super().__init__()
        self.save_hyperparameters()

    def train_dataset(self):
        return EfficientDetDataset(
            self.hparams.train_ds, self.hparams.train_tfms
        )
    
    def train_dataloader(self):
        train_dataset = self.train_dataset()
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
        )

        return train_loader

    def val_dataset(self):
        return EfficientDetDataset(
            self.hparams.val_ds, self.hparams.val_tfms
        )
        
    def val_dataloader(self):
        val_dataset = self.val_dataset()
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
        )
        
        return val_loader

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack([torch.tensor(image, dtype=torch.float32) for image in images])
        images = images.float()

        boxes = [target['bboxes'].float for target in targets]
        labels = [target['labels'].float for target in targets]
        img_size = torch.tensor([target['img_size'] for target in targets])
        img_scale = torch.tensor([target['img_scale'] for target in targets])

        annotations = {
            'bbox': boxes,
            'cls': labels,
            'img_size': img_size,
            'img_scale': img_scale
        }

        return images, annotations, targets, image_ids

if __name__ == '__main__':
    import pandas as pd
    from vehicle_dataset import CarsDatasetAdaptor

    train_ds_path = './data/vehicle/training_images'
    df = pd.read_csv('./data/vehicle/train_solution_bounding_boxes (1).csv')

    train_ds = CarsDatasetAdaptor(train_ds_path, df)

    dataset = EfficientDetDataset(dataset_adaptor=train_ds, transforms=None)

    for batch in dataset:
        image, target, image_id = batch
        print(image.shape)
        print(target)
        print(image_id)
        break