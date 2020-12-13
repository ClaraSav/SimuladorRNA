import numpy as np

import cv2


class ImageSettings:

    def __init__(self, img_dim=28, classes=1):
        self.img_size = img_dim
        self.img_size_flat = self.img_size * self.img_size
        self.img_shape = (self.img_size, self.img_size)
        self.img_shape_full = (self.img_size, self.img_size, 1)
        self.num_classes = classes

    def image_to_matrix(self, url):
        """
          This method transforms an image into a 28x28 matrix
        """
        image = cv2.imread(url)
        image.resize([self.img_size, self.img_size])
        image_array = np.asarray(image)
        return image_array
