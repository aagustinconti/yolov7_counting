# YOLOR general utils -> Custom

import torch  # Docker implemented
import torchvision  # Docker implemented
from torchvision import transforms

import cv2
import numpy as np
import time

# Utilities
from utils.general import non_max_suppression
from utils.datasets import letterbox
from utils.plots import output_to_keypoint


def scale_coords_custom(img1_shape, coords, img0_shape):
    """
    WHAT IT DOES:
        Rescale coords (xyxy) from img1_shape to img0_shape.

    INPUTS:
        img1_shape =  img1.shape -> Shape of resized image.
        coords = [xmin,ymin,xmax,ymax] -> coords of raw bounding boxes.
        img0_shape = img0.shape -> Shape of original image.

    OUTPUTS:
        coords = [xmin,ymin,xmax,ymax] -> Resized coords of bounding boxes.
    """

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


def draw_bbox(frame, coords, color, names, confidence):
    """
    WHAT IT DOES:
        Draw bboxes on the frame with names and confidence.

    INPUTS:
        frame -> Frame to draw.
        coords = [xmin,ymin,xmax,ymax] -> Bounding box coords of detection.
        color = (B,G,R) -> Color of bounding box.
        names = names -> Names of detection (List of model)
        confidence = confidence -> Confidences of detections.

    OUTPUTS:
        Only returns a boolean confirmation: True.
    """

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


def draw_roi(roi, roi_color, frame):
    """
    INPUTS:
        roi= ->
        roi_color = (B,G,R) ->
        frame ->

    OUTPUTS:
        Only returns a boolean confirmation: True.

    """
    # extract values
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi[0][0], roi[0][1], roi[1][0], roi[1][1]

    # draw roi
    frame = cv2.rectangle(
        frame,
        (int(roi_xmin), int(roi_ymin)),
        (int(roi_xmax), int(roi_ymax)),
        color=roi_color,
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return True


class coordinateStore:
    """
    Class to capture the click event and get the coordinates.

    1. Instantiate an object of class
    2. Capture the event with self.select_point()
    3. Use the self.point variable.
    """

    def __init__(self):
        self.point = False

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.point = (x, y)


def load_roi(cap, roi_color=(255, 255, 255), inv_h_frame=False):
    """
    WHAT IT DOES:
        Ask for the ROI.

    INPUTS:
        cap -> Capture of frame or image with cv2.videoCapture().
        roi_color = (BGR) -> Color of the ROI.
        inv_h_frame = False -> To invert the image horizontally.

    OUTPUTS:
        roi = [roi_xmin,roi_ymin,roi_xmax, roi_ymax] -> ROI bounding box.

    """

    ret, frame = cap.read()

    # To show image correctly (IE: web camera)
    if inv_h_frame:
        frame = cv2.flip(frame, 1)

    if ret:

        for i in range(4):

            # Instantiate class
            coordinates = coordinateStore()

            if i == 1:
                while True:
                    text_frame = frame.copy()
                    cv2.putText(text_frame, f"No points selected. Please choose your 'top-left' point double-clicking on the image.\n\nPlease press 'q' to continue after you've selected the point.", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, roi_color, 1)

                    selected_min = coordinates.point
                    if selected_min:
                        cv2.circle(text_frame, selected_min, radius=0,
                                   color=roi_color, thickness=10)

                    cv2.imshow('Selecting ROI', text_frame)

                    # capturing click event
                    cv2.setMouseCallback(
                        'Selecting ROI', coordinates.select_point)

                    if cv2.waitKey(22) & 0xFF == ord('q'):
                        break

                cv2.destroyAllWindows()

            if i == 2:
                while True:
                    text_frame = frame.copy()
                    cv2.putText(text_frame, f"'top-left' point have already selected.\nPlease choose your 'bottom-right' point double-clicking on the image.\n\nPlease press 'q' to continue after you've selected the point..", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, roi_color, 1)

                    selected_max = coordinates.point
                    # Only if the point is defined
                    if selected_max:
                        cv2.circle(text_frame, selected_max, radius=0,
                                   color=roi_color, thickness=10)

                    cv2.imshow('Selecting ROI', text_frame)

                    # capturing click event
                    cv2.setMouseCallback(
                        'Selecting ROI', coordinates.select_point)

                    if cv2.waitKey(22) & 0xFF == ord('q'):
                        break

                cv2.destroyAllWindows()

            if i == 3:
                # [(xmin,ymin),(xmax,ymax)]
                roi = [selected_min, selected_max]

                while True:
                    text_frame = frame.copy()
                    cv2.putText(text_frame, f"The roi has already done. Please press 'q' to continue with the detection.", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, roi_color, 1)

                    # draw roi
                    text_frame = cv2.rectangle(
                        text_frame,
                        roi[0],
                        roi[1],
                        color=roi_color,
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

                    cv2.imshow('Selecting ROI', text_frame)
                    if cv2.waitKey(22) & 0xFF == ord('q'):
                        break

                cv2.destroyAllWindows()

    return roi


def detect(frame, model, device, names, show_img=True, color=(0, 255, 0), img_sz=640, class_ids=[], conf_thres=0.25, iou_thres=0.65):
    """
    WHAT IT DOES:
        Detect objects in a frame.    

    INPUTS:
        frame -> Original frame
        model -> Pytorch object: model used for detection
        device -> PyTorch object: device selected, IE: Graphic card
        names -> List of classes.
        show_img = True -> To show images
        color = (0, 255, 0) -> Color of bounding boxes in BGR format.
        img_sz = 640 -> Image size accepted by the model to do the process.
        class_ids = [] -> Selected class ID, empty list means 'all of classes'.
        conf_thres =0.25 -> Confidence threshold, there is a detection if the confidence is greather than the threshold.
        iou_thres = 0.65 ->  Intersection over Union is an evaluation metric used to measure the accuracy of an object detector on a particular dataset.

    OUTPUTS:
        coords_bb = [[xmin,ymin,xmax,ymax],...,[coords_n]] -> List of coordinates of boundin boxes of each detection.
        classes_detected = [[name,class_id, confidence],...,[classes_detected_n]] -> List of list with name, class_id and confidence of each detection.
        delta_time = time.time() object -> Process time of the detection model.

    """

    # initialize vectors
    coords_bb = []
    classes_detected = []

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    delta_time = end_time - start_time

    # remove the noise of the output (NMS: a technique to filter the predictions of object detectors.)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # process the information of the filtered output and return the main characteristics [batch_id, class_id, x, y, w, h, conf]
    output = output_to_keypoint(pred)

    # for detection in frame
    for idx in range(output.shape[0]):

        # Separate by class id
        if (int(output[idx][1]) in class_ids) or (class_ids == []):

            # Rescale boxes (Rescale coords (xyxy) from img0 to frame)
            output[idx][2:6] = scale_coords_custom(
                img0.shape[0:2], output[idx][2:6], frame.shape).round()

            # generate coord to bounding boxes
            xmin, ymin = (output[idx, 2]-output[idx, 4] /
                          2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4] /
                          2), (output[idx, 3]+output[idx, 5]/2)

            # xyxy
            coord_bb = [xmin, ymin, xmax, ymax]

            # [class id, class name, confidence]
            class_detected = [names[int(output[idx][1])], int(
                output[idx][1]), round(output[idx][6], 2)]

            # fill the output list
            coords_bb.append(coord_bb)
            classes_detected.append(class_detected)

            # draw bounding boxes, classnames and confidence
            if show_img:
                draw_bbox(frame, coord_bb, color,
                          class_detected[0], class_detected[2])

    return coords_bb, classes_detected, delta_time
