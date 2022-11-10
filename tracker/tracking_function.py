# https://github.com/dongdv95/yolov5/blob/master/Yolov5_DeepSort_Pytorch/track.py

from deep_sort.deep_sort import DeepSort
import time
import numpy as np
import cv2


def xyxy2xywh(x):

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
            x[i][3] = y[i][1] +1
            x[i][1] = y[i][3]

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    
    return y


def xyxy2cxcy(x):
    """"
    WHAT IT DOES:
        Convert nx4 boxes from [x1, y1, x2, y2] to [cx,xy] where xy1=top-left, xy2=bottom-right

    INPUTS:
        x = [x1, y1, x2, y2] -> xy1=top-left, xy2=bottom-right

    OUTPUTS:
        y = [cx,xy] -> Centroid of bounding box.

    """
    y = np.copy(x[:2])
    y[0] = ((x[2] - x[0])) / 2 + x[0] # x center
    y[1] = ((x[3] - x[1])) / 2 + x[1] # y center

    return y


def load_deepsort(deep_sort_model="osnet_x1_0", max_dist=0.1, max_iou_distance=0.7, max_age=30, n_init=3, nn_budget=100):
    """
    WHAT IT DOES:
        Generate the tracking of the detections.

    INPUTS:
        MODEL_TYPE = "osnet_x1_0" -> Model of deep sort used. You can leave this empty and the program can recomend it one.
        MAX_DIST = 0.1 -> The matching threshold. Samples with larger distance are considered an invalid match.
        MAX_IOU_DISTANCE = 0.7 -> Gating threshold. Associations with cost larger than this value are disregarded.
        MAX_AGE = 30 -> Maximum number of missed misses before a track is deleted.
        N_INIT = 3 -> Number of frames that a track remains in initialization phase.
        NN_BUDGET = 100 -> Maximum size of the appearance descriptors gallery.

    OUTPUTS:
        ds_output = [[ds_cpoint, id, cls],...,[ds_cpoint_n, id_n, cls_n]] ->  Returns the list of lists with the centroid, id of detection and class name.
        delta_time = -> Time of processing the tracking model.

    """

    deepsort = DeepSort(deep_sort_model,
                        max_dist=max_dist,
                        max_iou_distance=max_iou_distance,
                        max_age=max_age, n_init=n_init, nn_budget=nn_budget,
                        use_cuda=True)
    return deepsort




def tracking(coords, classes_detected, deepsort, names, frame, ds_color = (0,0,255),show_img=True):
    
    xywhs = xyxy2xywh(np.array(coords))
    confs = np.array([[elem[2]] for elem in classes_detected])
    clss = np.array([[elem[1]] for elem in classes_detected])

    if coords != []:

        # pass detections to deepsort
        start_time = time.time()
        outputs = list(deepsort.update(xywhs, confs, clss, frame))
        end_time = time.time()

        delta_time = end_time - start_time
        
        ds_output = []
        # draw boxes for visualization
        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                ds_cpoint = tuple(xyxy2cxcy(output[0:4]))
                id = output[4]
                cls = output[5]

                ds_output.append([ds_cpoint, id, cls])

                if show_img:
                    cv2.circle(frame, (ds_cpoint[0], ds_cpoint[1]), radius=0, color=ds_color, thickness=3)
                    cv2.putText(frame, f"{names[cls]}: {id}", (ds_cpoint[0]-10, ds_cpoint[1]-7), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, ds_color, 1)

    else:
        start_time = time.time()
        deepsort.increment_ages()
        ds_output = []
        end_time = time.time()

        delta_time = end_time - start_time

    return ds_output, delta_time
