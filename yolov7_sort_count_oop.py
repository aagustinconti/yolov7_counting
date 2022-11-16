# https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/

# Pytorch
import torch
import torchvision

# Basics
import cv2

# Classes
from detection_oop import YoloDetect
from deepsort_oop import DeepSortTrack
from count_oop import Count


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
        self.show_detections = False
        self.show_tracking = False
        self.show_count = False

        # Pre defined
        self.names = None

        self.orig_w = None
        self.orig_h = None
        self.orig_fps = None

        self.stopped = False
        self.avg_fps = 0

        self.out_frame = None

        self.result = None

    def load_device(self, graphic_card):

        try:
            device = torch.device("cuda:"+str(graphic_card))
            return device

        except Exception as err:
            raise SystemError(
                'Error while trying to use Graphic Card. Please check that it is available.')

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

    def run(self):

        device = self.load_device(self.graphic_card)

        detection_model, self.names = YoloDetect.load_detection_model(
            self.model_path, device)

        tracking_model = DeepSortTrack.load_tracking_model(
            self.deep_sort_model, self.ds_max_dist, self.ds_max_iou_distance, self.ds_max_age, self.ds_n_init, self.ds_nn_budget)

        cap, self.orig_w, self.orig_h, self.orig_fps = self.load_video_capture(
            self.video_path)

        if self.save_vid:
            self.result = self.load_save_vid(
                self.save_loc, self.orig_w, self.orig_h)

        frame_count = 0
        total_fps = 0

        # Run detection
        while (cap.isOpened()):

            # Get frame
            ret, self.frame = cap.read()

            # To show image correctly (IE: web camera)
            if self.inv_h_frame:
                self.frame = cv2.flip(self.frame, 1)

            # If the video has not finished yet
            if ret:

                # Run Detection model
                detection = YoloDetect(self.frame, detection_model, device, self.names, self.show_img,
                                       self.color, self.img_sz, self.class_ids, self.conf_thres, self.iou_thres)

                if self.show_detections:
                    detection.__str__()

                if detection.det_out_coords != []:

                    # Run Sort model
                    tracking = DeepSortTrack(detection.det_out_coords, detection.det_out_classes,
                                             tracking_model, detection.det_out_frame, self.show_img, self.ds_color, self.names)

                    if self.show_tracking:
                        tracking.__str__()

                    # Count
                    count = Count(tracking.ds_out_tracking,
                                  self.roi, self.names)

                    if self.show_count:
                        count.__str__()

                    # Calculate fps (Aproximate: 25-30 FPS GEFORCE 1060 Max-Q Design)
                    fps = 1 / (detection.det_delta_time +
                               tracking.ds_delta_time)

                    total_fps += fps
                    frame_count += 1

                    # Show the processed frame
                    if self.show_img:

                        self.out_frame = tracking.ds_out_frame

                        # draw ROI
                        #draw_roi(roi, roi_color, frame)

                        # draw fps
                        cv2.putText(self.out_frame, f"{fps:.3f} FPS (YOLO + SORT)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, self.color, 1)

                        # draw counter
                        cv2.putText(self.out_frame, f"COUNTER = {count.counter_text}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, self.color, 1)

                        # show the frame
                        cv2.imshow('PROCESSED FRAME', self.out_frame)

                        # wait q to exit
                        if self.hold_img:
                            if cv2.waitKey(0) & 0xFF == ord('q'):
                                self.stopped = True
                                break
                        else:
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.stopped = True
                                break

            else:
                self.stopped = False
                break

            if self.save_vid:
                self.result.write(self.frame)

        # Close the videocapture
        cap.release()

        # To save the video
        if self.save_vid:
            self.result.release()

        # Avg fps
        if frame_count > 0:
            self.avg_fps = total_fps / frame_count

        # Close all windows
        if self.show_img:
            cv2.destroyAllWindows()

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

            Show Detections: {self.show_detections}
            Show Tracking: {self.show_tracking}
            Show Count: {self.show_count}

            """

            return return_str

        else:
            pass
