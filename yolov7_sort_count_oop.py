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

        self.result = None

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

    def load_save_vid(self, save_loc, orig_w, orig_h):

        result = cv2.VideoWriter(save_loc+'.avi',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, (orig_w, orig_h))
        return result

    def load_roi(self):
        # Call this method previously to call it in run method
        return True

    class YoloDetect():

        def __init__(self, frame, model, device, names, show_img, color, img_sz, class_ids, conf_thres, iou_thres):

            try:
                # Frame

                self.det_out_frame = frame

                # initialize vectors
                self.det_out_coords = []
                self.det_out_classes = []

                img = cv2.cvtColor(self.det_out_frame, cv2.COLOR_BGR2RGB)

                # reshape the frames to the adecuate w and h
                img = letterbox(img, img_sz, stride=64, auto=True)[0]

                # get image data to use for rescaling
                img0 = img.copy()

                # transform the image to tensor and send the tensor of the image to the device
                img = transforms.ToTensor()(img)
                img = torch.tensor(np.array([img.numpy()]))
                img = img.to(device)
                img = img.half()

                # time to count fps
                start_time = time.time()

                # get the output of the model
                with torch.no_grad():
                    pred, _ = model(img)

                # calculate fps
                end_time = time.time()

                self.det_delta_time = end_time - start_time

                # remove the noise of the output (NMS: a technique to filter the predictions of object detectors.)
                pred = non_max_suppression(pred, conf_thres, iou_thres)

                # process the information of the filtered output and return the main characteristics [batch_id, class_id, x, y, w, h, conf]
                self.det_output = output_to_keypoint(pred)

                # for detection in frame
                for idx in range(self.det_output.shape[0]):

                    # Separate by class id
                    if (int(self.det_output[idx][1]) in class_ids) or (class_ids == []):

                        # Rescale boxes (Rescale coords (xyxy) from img0 to frame)
                        self.det_output[idx][2:6] = self.scale_coords_custom(
                            img0.shape[0:2], self.det_output[idx][2:6], self.det_out_frame.shape).round()

                        # generate coord to bounding boxes
                        xmin, ymin = (self.det_output[idx, 2]-self.det_output[idx, 4] /
                                      2), (self.det_output[idx, 3]-self.det_output[idx, 5]/2)
                        xmax, ymax = (self.det_output[idx, 2]+self.det_output[idx, 4] /
                                      2), (self.det_output[idx, 3]+self.det_output[idx, 5]/2)

                        # xyxy
                        coord_bb = [xmin, ymin, xmax, ymax]

                        # [class id, class name, confidence]
                        class_detected = [names[int(self.det_output[idx][1])], int(
                            self.det_output[idx][1]), round(self.det_output[idx][6], 2)]

                        # fill the output list
                        self.det_out_coords.append(coord_bb)
                        self.det_out_classes.append(class_detected)

                        # draw bounding boxes, classnames and confidence
                        if show_img:
                            self.draw_bbox(self.det_out_frame, coord_bb, color,
                                           class_detected[0], class_detected[2])

            except Exception as err:
                raise ImportError(
                    'Error while trying instantiate the detection object. Please check that.')

        def scale_coords_custom(self, img1_shape, coords, img0_shape):

            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

            coords[0] -= pad[0]  # x padding
            coords[2] -= pad[0]  # x padding
            coords[1] -= pad[1]  # y padding
            coords[3] -= pad[1]  # y padding
            coords[:] /= gain

            return coords

        def draw_bbox(self, frame, coords, color, names, confidence):

            # draw bounding box
            frame = cv2.rectangle(
                frame,
                (int(coords[0]), int(coords[1])),
                (int(coords[2]), int(coords[3])),
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA
            )

            # write confidence and class names
            cv2.putText(frame, f"{names}: {confidence}", (int(coords[0]), int(coords[1])-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

            return True

        def __str__(self):

            output_text_detection = f"""
            BBxes coords: {self.det_out_coords}\n
            Classes Detected: {self.det_out_classes}\n
            Exec. time YOLOv7x model: {self.det_delta_time} [s]\n\n
            
            """

            return output_text_detection

    def count(self):
        return True

    def run(self):

        self.device = self.load_device(self.graphic_card)

        self.detection_model, self.names = self.load_detection_model(
            self.model_path, self.device)
        self.tracking_model = self.load_tracking_model(
            self.deep_sort_model, self.ds_max_dist, self.ds_max_iou_distance, self.ds_max_age, self.ds_n_init, self.ds_nn_budget)

        self.cap, self.orig_w, self.orig_h, self.orig_fps = self.load_video_capture(
            self.video_path)

        if self.save_vid:
            self.result = self.load_save_vid(
                self.save_loc, self.orig_w, self.orig_h)

        # Run detection
        while (self.cap.isOpened()):

            # get the frames
            ret, self.frame = self.cap.read()

            # To show image correctly (IE: web camera)
            if self.inv_h_frame:
                self.frame = cv2.flip(self.frame, 1)

            # if the video has not finished yet
            if ret:

                detection = self.YoloDetect(self.frame, self.detection_model, self.device, self.names,
                                            self.show_img, self.color, self.img_sz, self.class_ids, self.conf_thres, self.iou_thres)

                detection.det_out_coords
                detection.det_out_classes
                detection.det_delta_time
                print(detection)

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
