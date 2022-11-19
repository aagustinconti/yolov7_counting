# https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/

# Pytorch
import torch
import torchvision

# Deep Sort
from deep_sort.deep_sort import DeepSort

# Basics
import cv2

# Logs
import logging

# Time
import time

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
        self.ends_in_sec = None
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

        self.roi = []
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
        self.names = None

        self.orig_w = None
        self.orig_h = None
        self.orig_fps = None

        self.stopped = False
        self.avg_fps = 0

        self.out_frame = None

        # Detection
        self.detection = None
        self.tracking = None
        self.count = None


        # Count
        self.count_out_classes = {}
        self.counted = []

        # Debug
        logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)

        self.show_detection = False
        self.show_tracking = False
        self.show_count = False


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

        try:

            result = cv2.VideoWriter(save_loc+'.avi',
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     10, (orig_w, orig_h))
            return result

        except Exception as err:
            raise ImportError(
                'Error while trying write the results. Please check that.')

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

    def run(self):

        # Debug
        if self.show_configs:
            logging.info(f'\n{self.__str__()}')    
        
        logging.info('Setting up the device...')
        device = self.load_device(self.graphic_card)
        logging.info('Device has setted up.')

        logging.info('Loading the detection model...')
        detection_model, self.names = self.load_detection_model(
            self.model_path, device)
        logging.info('The detection model has loaded.')

        logging.info('Loading the tracking model...')
        tracking_model = self.load_tracking_model(
            self.deep_sort_model, self.ds_max_dist, self.ds_max_iou_distance, self.ds_max_age, self.ds_n_init, self.ds_nn_budget)
        logging.info('The tracking model has loaded.')

        logging.info('Loading the video capture...')
        cap, self.orig_w, self.orig_h, self.orig_fps = self.load_video_capture(
            self.video_path)
        logging.info('The video capture has loaded.')

        if self.save_vid:
            logging.info('Loading the results capture...')
            result = self.load_save_vid(
                self.save_loc, self.orig_w, self.orig_h)
            logging.info('The results capture has laoded.')

        frame_count = 0
        total_fps = 0
        
        start_ends_in_sec = time.time()

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
                self.detection = YoloDetect(self.frame, detection_model, device, self.names, self.show_img,
                                            self.color, self.img_sz, self.class_ids, self.conf_thres, self.iou_thres)

                if self.detection.det_out_coords != []:

                    # Run Sort model
                    self.tracking = DeepSortTrack(self.detection.det_out_coords, self.detection.det_out_classes,
                                                  tracking_model, self.detection.det_out_frame, self.show_img, self.ds_color, self.names)

                    # Count
                    if not self.roi:
                        self.roi = [0, 0, self.orig_w, self.orig_h]

                    self.count = Count(self.tracking.ds_out_tracking,
                                       self.roi, self.names, self.count_out_classes, self.counted)
                    
                    self.count_out_classes = self.count.count_out_classes
                    self.counted = self.count.counted

                    # Calculate fps (Aproximate: 25-30 FPS GEFORCE 1060 Max-Q Design)
                    fps = 1 / (self.detection.det_delta_time +
                               self.tracking.ds_delta_time)

                    total_fps += fps
                    frame_count += 1

                    # Debug
                    if self.show_detection:
                        logging.debug(f'\n{self.detection.__str__()}')
                    if self.show_tracking:
                        logging.debug(f'\n{self.tracking.__str__()}')
                    if self.show_count:
                        logging.debug(f'\n{self.count.__str__()}')

                    # Show the processed frame
                    if self.show_img:

                        self.out_frame = self.tracking.ds_out_frame

                        # draw ROI

                        cv2.rectangle(
                            self.out_frame,
                            (int(self.roi[0]), int(self.roi[1])),
                            (int(self.roi[2]), int(self.roi[3])),
                            color=self.roi_color,
                            thickness=1,
                            lineType=cv2.LINE_AA
                        )

                        # draw fps
                        cv2.putText(self.out_frame, f"{fps:.3f} FPS (YOLO + SORT)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, self.color, 1)

                        # draw counter
                        cv2.putText(self.out_frame, f"COUNTER = {self.count.counter_text}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, self.color, 1)

                        # show the frame
                        cv2.imshow('PROCESSED FRAME', self.out_frame)

                        # wait q to exit
                        if self.hold_img:
                            if cv2.waitKey(0) & 0xFF == ord('q'):
                                logging.info('Exiting by keyboard...')
                                self.stopped = True
                                break
                        else:
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                logging.info('Exiting by keyboard...')
                                self.stopped = True
                                break
                    else:
                        # ends when the time of excecution is greather than ends_in_sec 
                        end_ends_in_sec = time.time()
                        
                        if self.ends_in_sec == None:
                            logging.critical('If show_img = False you need to define "self.ends_in_sec" [secs].')
                            break
                        else:
                            if (end_ends_in_sec - start_ends_in_sec) >= self.ends_in_sec:
                                logging.info(f'Stopping... The time exeeds the defined excecution time of {self.ends_in_sec} [seconds]...')
                                self.stopped = True
                                break
                                        

            else:
                logging.info('The video has finished.')
                self.stopped = False
                break

            if self.save_vid:
                logging.info('Writting the results...')
                result.write(self.out_frame)
                logging.info('The result video has been written.')

        # Stopped
        logging.info(f'Stopped manually: {self.stopped}')

        # Close the videocapture
        logging.info('Releasing capture...')
        cap.release()
        logging.info('The capture has been released.')

        # To save the video
        if self.save_vid:
            logging.info('Releasing results capture...')
            result.release()
            logging.info('The results capture has been released.')

        # Avg fps
        if frame_count > 0:
            logging.info('Calculating the average of fps...')
            self.avg_fps = total_fps / frame_count
            logging.info(f'The average of fps is: {round(self.avg_fps,2)} [fps]')

        # Close all windows
        if self.show_img:
            logging.info('Destroying all windows...')
            cv2.destroyAllWindows()
            logging.info('All windows has been destroyed.')

    def __str__(self):


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

