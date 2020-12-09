class ImageSettings:
    def __init__(self, img_dim, classes):
      self.img_size = img_dim
      self.img_size_flat = self.img_size * self.img_size
      self.img_shape = (self.img_size, self.img_size)
      self.img_shape_full = (self.img_size, self.img_size, 1)
      self.num_classes = classes
