import sys


class Config(object):
    def __init__(self):
        self.input_shape = (256, 256)
        self.trained_model_path = '/root/HSR/catkin_ws/src/cv_detect_unstable_object/train_results/color/snapshot_model_f1max.npz'
        self.color = True
        self.device = 0
        self.n_class = 1
        #self.backborn = 'seresnext50'
        self.backborn = 'resnet50'
