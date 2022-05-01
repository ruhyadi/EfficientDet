"""
Vehicle dataset adaptor
"""
import numpy as np
import pandas as pd
import os
from PIL import Image

class CarsDatasetAdaptor:
    def __init__(self, images_dir, annotations_dataframe):
        self.images_dir = images_dir
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()

    def __len__(self):
        return len(self.images)

    def get_data(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.images_dir, image_name))
        # bbox in pascal voc: [xmin, ymin, xmax, ymax]
        bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ['xmin', 'ymin', 'xmax', 'ymax']
        ].values
        class_labels = np.ones(len(bboxes))

        return image, bboxes, class_labels, index