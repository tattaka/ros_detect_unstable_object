 #! /usr/bin/env python

import cv2
import os
import time

import actionlib
import cv_bridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
from detect_unstable_object.msg import DetectUnstableObjectAction,DetectUnstableObjectGoal, DetectUnstableObjectResult
from sensor_msgs.msg import CompressedImage
import chainer
from chainer import links as L
from chainer import functions as F
from chainercv import transforms
from importlib import import_module
from models import ResUNet


class UnstableObjectDetector:

    def __init__(self, config_name='detect_cfg'):

        self.net = None
        self.config = None
        self._bridge = cv_bridge.CvBridge()

        self.config = self.get_config(config_name)
        self.get_model()

        self.load_model()
        self.server = actionlib.SimpleActionServer('detect_unstable_object/action_server', DetectUnstableObjectAction,
                                                   self.action_call_back, auto_start=False)
        self.compressed_subscriber = rospy.Subscriber(
            'detect_unstable_object/image_sub/compressed', CompressedImage, self.call_back, queue_size=1)
        self.image_pub = rospy.Publisher(
            'detect_unstable_object/result_pub', Image, queue_size=1)
        self.compressed_image_pub = rospy.Publisher(
            'detect_unstable_object/result_pub/compressed', CompressedImage, queue_size=1)
        self.server.start()

    def action_call_back(self, goal):
        image = self._bridge.imgmsg_to_cv2(goal.image, desired_encoding="rgb8")
        msg = self.eval_image(image)
        msg = self._bridge.cv2_to_imgmsg(msg, "bgr8")
        if not self.server.is_preempt_requested():
            if msg is not None:
                result = DetectUnstableObjectResult()
                result.predict_image = msg
                result.id = goal.id
                self.server.set_succeeded(result)

                result_msg = msg
                result_msg.header.stamp = rospy.Time.now()
                self.image_pub.publish(result_msg)

    def call_back(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
        result = self.eval_image(image)
        result_msg = self._bridge.cv2_to_compressed_imgmsg(result, dst_format='jpg')
        result_msg.header.stamp = rospy.Time.now()
        self.compressed_image_pub.publish(result_msg)

    def get_config(self, config_name):
        opts = import_module('configs.' + config_name)
        return opts.Config()

    def get_model(self):
        self.net = ResUNet(return_hidden=False, n_class=self.config.n_class, backborn=self.config.backborn,
                           pretrained_model=None, reduce_param=1, color=self.config.color)

    def load_model(self):
        chainer.serializers.load_npz(
            self.config.trained_model_path, self.net)
        device = self.config.device
        if device >= 0:
            chainer.cuda.get_device(device).use()
            self.net.to_gpu()

    def eval_image(self, image):
        start = time.time()
        image_color, param = self.preprocesss(image)
        if self.config.color:
            image = L.model.vision.resnet.prepare(
                image_color, size=self.config.input_shape)
        else:
            image = cv2.cvtColor(image_color.transpose(
                (1, 2, 0)), cv2.COLOR_RGB2GRAY)[None, :, :]
            image = image / 255.0
        pred_heatmap = self.net.predict(self.net.xp.asarray(
            image[None, :, :, :].astype(np.float32)))
        end = time.time()
        rospy.loginfo('prediction finished. (%f [sec])' % (end - start, ))
        pred_heatmap = chainer.cuda.to_cpu(pred_heatmap.data[0][0])
        pred_heatmap = self.postprocess(pred_heatmap)
        #print(image_color.shape)
        pred_fusion = self.make_fusion_image(pred_heatmap, image_color)
        #pred_fusion = self.inverse_resize_contain(pred_fusion, param)
        pred_fusion = self.inverse_resize_contain(pred_heatmap, param)
        cv2.imwrite("test_result.png", pred_fusion)
        return pred_fusion

    def preprocesss(self, image):
        image = image.transpose((2, 0, 1))
        ori_size = (image.shape[1], image.shape[2])
        image, param = transforms.resize_contain(
            image, size=self.config.input_shape, fill=0, return_param=True)
        param['ori_size'] = ori_size
        return image, param

    def postprocess(self, pred_heatmap):
        pred_heatmap = F.tanh(pred_heatmap).data
        pred_heatmap = (pred_heatmap*255).astype(np.uint8)
        pred_heatmap = cv2.applyColorMap(
            pred_heatmap, cv2.COLORMAP_HOT)
        return pred_heatmap

    def make_fusion_image(self, pred_heatmap, image):
        pred_fusion = image.copy().transpose(1, 2, 0)
        pred_fusion[:, :, 0:2] = pred_fusion[:, :, 0:2] * \
            (1-pred_heatmap[:, :, 2]/255.)[:, :, None]
        #print((pred_heatmap[:, :, 2]/255.).max())
        return pred_fusion

    def inverse_resize_contain(self, image, param):
        y_offset, x_offset, scaled_size, ori_size = param['y_offset'], param[
            'x_offset'], param['scaled_size'], param['ori_size']
        image = image[y_offset:y_offset+scaled_size[0],
                      x_offset:x_offset+scaled_size[1], :]
        image = cv2.resize(image, ori_size[::-1])
        return image
