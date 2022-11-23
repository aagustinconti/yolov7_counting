# https://github.com/WongKinYiu/yolov7

# Pytorch
import torch
import torchvision
from torchvision import transforms

# Basics
import cv2
import numpy as np
import time

# Utilities
from utils.general import non_max_suppression
from utils.datasets import letterbox
from utils.plots import output_to_keypoint


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
        DETECTION:\n
        BBxes coords: {self.det_out_coords}\n
        Classes Detected: {self.det_out_classes}\n
        Exec. time YOLOv7x model: {self.det_delta_time} [s]\n\n
        
        """

        return output_text_detection
