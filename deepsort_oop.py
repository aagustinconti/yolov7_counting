# https://github.com/dongdv95/yolov5/blob/master/Yolov5_DeepSort_Pytorch/track.py

# Deep Sort
from deep_sort.deep_sort import DeepSort

# Basics
import cv2
import numpy as np
import time


class DeepSortTrack():

    def __init__(self, coords, classes_detected, deepsort, frame, show_img, ds_color, names):

        self.ds_out_frame = frame
        self.ds_delta_time = time.time()
        self.ds_out_tracking = []

        try:

            xywhs = self.xyxy2xywh(np.array(coords))
            confs = np.array([[elem[2]] for elem in classes_detected])
            clss = np.array([[elem[1]] for elem in classes_detected])

            if coords != []:

                # pass detections to deepsort
                start_time = time.time()
                outputs = list(deepsort.update(
                    xywhs, confs, clss, self.ds_out_frame))
                end_time = time.time()

                self.ds_delta_time = end_time - start_time

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        ds_cpoint = tuple(self.xyxy2cxcy(output[0:4]))
                        id = output[4]
                        cls = output[5]

                        self.ds_out_tracking.append([ds_cpoint, id, cls])

                        if show_img:
                            cv2.circle(
                                self.ds_out_frame, (ds_cpoint[0], ds_cpoint[1]), radius=0, color=ds_color, thickness=3)
                            cv2.putText(self.ds_out_frame, f"{names[cls]}: {id}", (ds_cpoint[0]-10, ds_cpoint[1]-7), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, ds_color, 1)

            else:
                start_time = time.time()
                deepsort.increment_ages()
                ds_output = []
                end_time = time.time()

                self.ds_delta_time = end_time - start_time

        except Exception as err:
            raise ImportError(
                'Error while trying instantiate the tracking object. Please check that.')

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

    def xyxy2xywh(self, x):
        """
        WHAT IT DOES:
            - Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
            - xywhs is making negative h because some ymin and ymax are inverted or they have the same dimention,
                so the resize of the frame has an error when the method _resize tries to resize the bboxes.
                This function solve this problem.

        INPUTS:
            x = [xmin,ymin,xmax,ymax] -> List of coordinates of a bounding box.

        OUTPUTS:
            y = [xleft,ytop,width,height] -> List of 

        """

        y = np.copy(x)

        for i in range(len(x)):
            if x[i][3] <= x[i][1]:
                x[i][3] = y[i][1] + 1
                x[i][1] = y[i][3]

        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height

        return y

    def xyxy2cxcy(self, x):
        """"
        WHAT IT DOES:
            Convert nx4 boxes from [x1, y1, x2, y2] to [cx,xy] where xy1=top-left, xy2=bottom-right

        INPUTS:
            x = [x1, y1, x2, y2] -> xy1=top-left, xy2=bottom-right

        OUTPUTS:
            y = [cx,xy] -> Centroid of bounding box.

        """
        y = np.copy(x[:2])
        y[0] = ((x[2] - x[0])) / 2 + x[0]  # x center
        y[1] = ((x[3] - x[1])) / 2 + x[1]  # y center

        return y

    def __str__(self):

        output_text_tracking = f"""
        TRACKING:\n
        Classes Detected: {self.ds_out_tracking}\n
        Exec. time DeepSort model: {self.ds_delta_time} [s]\n\n
        
        """

        return output_text_tracking
