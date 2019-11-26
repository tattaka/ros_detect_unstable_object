import sys


class Config(object):
    def __init__(self):
        self.input_shape = (256, 256)
        self.trained_model_path = 'color'
        self.color = True
        self.device = 0
        self.num_class = 1
        self.backborn = 'seresnext50'
