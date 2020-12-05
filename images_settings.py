class ImageSettings:
    def __init__(self, img_dim, classes):
      self.img_zise = img_dim
      self.img_size_flat = self.img_zise * self.img_zise
      self.img_shape = (self.img_size, self.img_zise)
      self.img_shape_full = (self.img_zise, self.img_zise)
      self.num_classes = classes
