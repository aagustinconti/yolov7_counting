# https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/

import torch
import torchvision
from torchvision import transforms

import cv2
import numpy as np
import time

# Utilities
from utils.general import non_max_suppression
from utils.datasets import letterbox
from utils.plots import output_to_keypoint

# Custom functions
from utils.custom_functions import detect, draw_roi, load_roi
from tracker.tracking_function import tracking, load_deepsort

# Deep Sort
from deep_sort.deep_sort import DeepSort


"""


"""


class YoloSortCount():

    def __init__(self):

        # Init variables
        self.video_path = 0

        self.show_img = True
        self.inv_h_frame = False
        self.hold_img = False

        self.save_vid = False
        self.save_loc = "result"

        self.model_path = 'pretrained_weights/yolov7.pt'
        self.graphic_card = 0
        self.class_ids = []
        self.img_sz = 640
        self.color = (0, 255, 0)
        self.conf_thres = 0.25
        self.iou_thres = 0.65

        self.roi = [0, 0, 1, 1]
        self.roi_color = (255, 255, 255)

        self.deep_sort_model = "osnet_x1_0"
        self.ds_max_dist = 0.1
        self.ds_max_iou_distance = 0.7
        self.ds_max_age = 30
        self.ds_n_init = 3
        self.ds_nn_budget = 100
        self.ds_color = (0, 0, 255)

        self.show_configs = False

        # Pre defined
        self.device = None
        self.detection_model = None
        self.names = None

        self.tracking_model = None

        self.cap = None
        self.orig_w = None
        self.orig_h = None
        self.orig_fps = None

        self.frame_count = 0
        self.fps = 0
        self.total_fps = 0

        self.exec_time_yolo = None
        self.exec_time_ds = None

        self.counted = []
        self.classes_after_ds = {}

        self.stopped = False
        self.avg_fps = 0


    def load_device(self, graphic_card):

        try:
            device = torch.device("cuda:"+str(graphic_card))
            return device

        except Exception as err:
            raise SystemError(
                'Error while trying to use Graphic Card. Please check that it is available.')

    def load_detection_model(self, model_path, device):

        try:

            # Load all characteristics of YOLOv7x model
            weigths = torch.load(model_path)

            # Send model characteristics to the graphic card
            model = weigths['model']
            model = model.half().to(device)
            _ = model.eval()

            # Get model class names
            names = model.module.names if hasattr(
                model, 'module') else model.names

            return model, names

        except Exception as err:
            raise ImportError(
                'Error while trying to load the detection model. Please check that.')

    def load_tracking_model(self, deep_sort_model, max_dist, max_iou_distance, max_age, n_init, nn_budget):
        try:
            deepsort = DeepSort(deep_sort_model,
                                max_dist=max_dist,
                                max_iou_distance=max_iou_distance,
                                max_age=max_age, n_init=n_init, nn_budget=nn_budget,
                                use_cuda=True)
            return deepsort

        except Exception as err:
            raise ImportError(
                'Error while trying to load the tracking model. Please check that.')

    def load_video_capture(self, video_path):

        try:

            cap = cv2.VideoCapture(video_path)

            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_fps = cap.get(cv2.CAP_PROP_FPS) % 100

            return cap, orig_w, orig_h, orig_fps
        
        except Exception as err:
            raise ImportError(
                'Error while trying read the video. Please check that.')

    def load_save_vid(self):
        return True

    def load_roi(self):
        # Call this method previously to call it in run method
        return True

    def detect(self):
        return True

    def count(self):
        return True

    def run(self):

        self.device = self.load_device(self.graphic_card)

        self.detection_model, self.names = self.load_detection_model(
            self.model_path, self.device)
        self.tracking_model = self.load_tracking_model(
            self.deep_sort_model, self.ds_max_dist, self.ds_max_iou_distance, self.ds_max_age, self.ds_n_init, self.ds_nn_budget)

        self.cap, self.orig_w, self.orig_h, self.orig_fps = self.load_video_capture(self.video_path)

        if self.save_vid:
            self.load_save_vid()

        # Run detection
        while (self.cap.isOpened()):
            break

        return True

    def __str__(self):

        if self.show_configs:

            return_str = f"""\n

            Video path selected: {str(self.video_path)}\n\n
            
            Show image: {str(self.show_img)}\n
            Invert frame: {str(self.inv_h_frame)}\n
            Hold image: {str(self.hold_img)}\n\n

            Save results: {str(self.save_vid)}\n
            Results path: {str(self.save_loc)}\n\n

            Detection model path selected: {str(self.model_path)}\n
            Graphic card selected: {str(self.graphic_card)}\n
            Class Id's selected: {str(self.class_ids)}\n\n
            Model image size selected: {str(self.img_sz)}\n
            Detection color selected: {str(self.color)}\n
            Detection confidence threshold selected: {str(self.conf_thres)}\n\n

            ROI selected: {str(self.roi)}\n
            ROI color selected: {str(self.roi_color)}\n\n

            Deep Sort model selected: {str(self.deep_sort_model)}\n
            Deep Sort max. distance selected: {str(self.ds_max_dist)}\n
            Deep Sort max. age selected: {str(self.ds_max_age)}\n
            Deep Sort color selected: {str(self.ds_color)}\n\n
      
            """

            return return_str
        else:
            return "\n\nThis is an instance of the class YoloSortCount().\n\n"



# Test

test = YoloSortCount()

test.show_configs = True

print(test)

test.run()