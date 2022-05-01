"""
Model module
"""

import torch
from pytorch_lightning import LightningModule

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

class EfficientDetModel(LightningModule):
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 512,
        threshold: float = 0.2,
        lr: float = 0.0001,
        iou_threshold: float = 0.44,
        inference_tfms = None,
        model_arc: str = 'tf_efficientnetv2_b0',
        pretrained: bool = False,
        ):

        super().__init__()
        self.save_hyperparameters()
        self.model = self.create_model(
            self.hparams.num_classes,
            self.hparams.img_size,
            self.hparams.model_arc,
            self.hparams.pretrained,
        )

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch
        
        # calculate loss
        losses = self.model(images, annotations)

        # logging
        logging_losses = {
            'class_loss': losses['class_loss'].detach(),
            'box_loss': losses['box_loss'].detach(),
        }

        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/class_loss', logging_losses['class_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/box_loss', logging_losses['box_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)
        detections = outputs['detections']

        batch_predictions = {
            'predictions': detections,
            'targets': targets,
            'image_ids': image_ids,
        }

        logging_losses = {
            'class_loss': outputs['class_loss'].detach(),
            'box_loss': outputs['box_loss'].detach(),
        }

        self.log('valid/loss', outputs['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('valid/class_loss', logging_losses['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('valid/box_loss', logging_losses['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}

    def create_model(
        self,
        num_classes: int = 1,
        image_size: int = 512,
        architecture: str = 'tf_efficientnetv2_b0',
        pretrained: bool = False
        ):

        # sign new model params
        efficientdet_model_param_dict[architecture] = dict(
            name=architecture,
            backbone_name=architecture,
            backbone_args=dict(drop_path_rate=0.1),
            url=''
            )

        # config
        config = get_efficientdet_config(model_name=architecture)
        config.update({'num_classes': num_classes})
        config.update({'image_size': (image_size, image_size)})

        # initiate model
        net = EfficientDet(config, pretrained_backbone=pretrained)
        net.class_net = HeadNet(config, num_classes)

        return DetBenchTrain(net, config)

if __name__ == '__main__':
    model = EfficientDetModel(model_arc='tf_efficientnetv2_b0', pretrained=False)
    print(model)