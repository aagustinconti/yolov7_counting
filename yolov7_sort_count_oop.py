# https://github.com/WongKinYiu/yolov7/

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

# YouTube videos

import pafy

# Classes
from detection_oop import YoloDetect
from deepsort_oop import DeepSortTrack
from count_oop import Count




class YoloSortCount():

    def __init__(self):

        # Init variables
        self.video_path = 0

        self.show_img = True
        self.auto_load_roi = True
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

        self.max_width = 720

        # Pre defined
        self.names = None

        self.orig_w = None
        self.orig_h = None
        self.orig_ratio = None
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
        logging.basicConfig(
            format='%(asctime)s | %(levelname)s: %(message)s', level=logging.INFO)

        self.show_configs = False
        self.show_detection = False
        self.show_tracking = False
        self.show_count = False

        # Plots

        self.plot_xmin = 10
        self.plot_ymin = 10
        self.plot_padding = 2
        self.plot_text_color = (255, 255, 255)
        self.plot_bgr_color = (0, 0, 0)

    def load_device(self, graphic_card):
        """
        WHAT IT DOES:
            - Load the torch device.

        """

        try:
            device = torch.device("cuda:"+str(graphic_card))
            return device

        except Exception as err:
            raise SystemError(
                'Error while trying to use Graphic Card. Please check that it is available.')

    def load_video_capture(self, video_path):
        """
        WHAT IT DOES:
            - Load the video capture.
            - Resize the frames.

        """

        try:

            logging.info('Loading the video capture...')

            if "https://www.youtube.com/" in str(video_path):

                logging.info('YouTube video detected as source.')
                video = pafy.new(video_path)
                logging.info(f'YouTube video name: {video.title}')

                logging.info(
                    'Getting the video, remember thats maybe take a while...')
                best = video.getbest(preftype="mp4")
                cap = cv2.VideoCapture(best.url)
                logging.info('YouTube video capture has been loaded.')

            else:
                if video_path == 0:
                    logging.info('Source of the video: WebCamera')
                else:
                    logging.info(f'Source of the video: {video_path}')

                cap = cv2.VideoCapture(video_path)
                
                # To discard delayed frames
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                logging.info('The video capture has been loaded.')

            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_ratio = orig_w / orig_h
            orig_fps = cap.get(cv2.CAP_PROP_FPS) % 100

            return cap, orig_w, orig_h, orig_ratio, orig_fps

        except Exception as err:
            raise ImportError(
                'Error while trying read the video. Please check that.')

    def load_save_vid(self, save_loc, orig_w, orig_h):
        """
        WHAT IT DOES:
            - Load the result writer.

        """

        try:

            result = cv2.VideoWriter(save_loc+'.mp4',
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     10, (orig_w, orig_h))
            return result

        except Exception as err:
            raise ImportError(
                'Error while trying write the results. Please check that.')

    def load_detection_model(self, model_path, device):
        """
        WHAT IT DOES:
            - Load the detection model and extracting the names of the classes and the model.

        """

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
        """
        WHAT IT DOES:
            - To load the tracking model

        """

        try:
            logging.info('This step may take a while...')
            deepsort = DeepSort(deep_sort_model,
                                max_dist=max_dist,
                                max_iou_distance=max_iou_distance,
                                max_age=max_age, n_init=n_init, nn_budget=nn_budget,
                                use_cuda=True)
            return deepsort

        except Exception as err:
            raise ImportError(
                'Error while trying to load the tracking model. Please check that.')

    def load_roi(self):
        """
        WHAT IT DOES:
            - To select the ROI, interactive way.

        """
        cap_roi, _, _,_,_= self.load_video_capture(self.video_path)
        
        orig_w_roi = int(cap_roi.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h_roi = int(cap_roi.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_ratio_roi = orig_w_roi / orig_h_roi
        
        ret, select_roi_frame = cap_roi.read()
      
        # To avoid the camera delay
        if not self.hold_img:
            frame_count_roi = 0
            while frame_count_roi <= 3 and ret:

                ret, select_roi_frame = cap_roi.read()
                frame_count_roi += 1

        
        # To adjust to the max width
        if (self.max_width !=None) and (orig_w_roi != 0) and (orig_w_roi > self.max_width):                
                select_roi_frame = cv2.resize(select_roi_frame,(int(self.max_width), int(self.max_width/orig_ratio_roi)))
        

        # To show image correctly (IE: web camera)
        if self.inv_h_frame:
            select_roi_frame = cv2.flip(select_roi_frame, 1)

        roi = cv2.selectROI("Load ROI", select_roi_frame)

        if roi != (0, 0, 0, 0):
            roi = [roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]]
        else:
            roi = None

        cap_roi.release()
        cv2.destroyAllWindows()

        return roi

    def plot_text(self, frame, frame_w, fps, plot_xmin, plot_ymin, padding, counter_text, plot_text_color, plot_bgr_color):
        """
        WHAT IT DOES:
            - Plot text into the output frame

        """

        # Save the first xmin
        aux_xmin = plot_xmin

        # FPS counter
        label = f"FPS (YOLOv7x + SORT): {fps:.3f}"
        # min required space for the text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        frame = cv2.rectangle(frame, (plot_xmin-padding, plot_ymin - padding), (plot_xmin +
                              w + padding, plot_ymin + h + padding * 4), plot_text_color, cv2.FILLED)
        frame = cv2.putText(frame, label, (plot_xmin + padding, plot_ymin + padding + h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, plot_bgr_color, 1)  # bottom-left align

        plot_ymin = plot_ymin + h + 7 * padding

        # Detection elements
        for elem in counter_text:  # [[0, 'person', 1]]

            # format text -> person (ID: 0): 1
            label = f"{elem[1]} (ID: {elem[0]}): {elem[2]}"
            # min required space for the text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # check if is not sobrepassing the frame width
            if (plot_xmin + w + padding * 2) < frame_w:

                frame = cv2.rectangle(frame, (plot_xmin-padding, plot_ymin - padding),
                                      (plot_xmin + w + padding, plot_ymin + h + padding * 4), plot_bgr_color, cv2.FILLED)
                frame = cv2.putText(frame, label, (plot_xmin + padding, plot_ymin + padding + h),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, plot_text_color, 1)  # bottom-left align

                plot_xmin = plot_xmin + w + padding * 4
            else:
                plot_xmin = aux_xmin
                plot_ymin = plot_ymin + h + 7 * padding

                frame = cv2.rectangle(frame, (plot_xmin-padding, plot_ymin - padding),
                                      (plot_xmin + w + padding, plot_ymin + h + padding * 4), plot_bgr_color, cv2.FILLED)
                frame = cv2.putText(frame, label, (plot_xmin + padding, plot_ymin + padding + h),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, plot_text_color, 1)  # bottom-left align

        return frame


    def run(self):
        """
        WHAT IT DOES:
            - Run the entire process of detection > tracking > count.

        """

        # Debug
        if self.show_configs:
            logging.info(f'\n{self.__str__()}')

        logging.info('Setting up the device...')
        device = self.load_device(self.graphic_card)
        logging.info('Device has been setted up.')

        logging.info('Loading the detection model...')
        detection_model, self.names = self.load_detection_model(
            self.model_path, device)
        logging.info('The detection model has been loaded.')

        logging.info('Loading the tracking model...')
        tracking_model = self.load_tracking_model(
            self.deep_sort_model, self.ds_max_dist, self.ds_max_iou_distance, self.ds_max_age, self.ds_n_init, self.ds_nn_budget)
        logging.info('The tracking model has been loaded.')

        if self.show_img:
            if self.auto_load_roi:
                logging.info('Loading ROI...')
                self.roi = self.load_roi()
                logging.info('ROI has been loaded.')

        cap, self.orig_w, self.orig_h, self.orig_ratio, self.orig_fps = self.load_video_capture(
            self.video_path)

        if not self.roi:
            logging.info('ROI has not loaded. Setting it up as full image...')
            self.roi = [0, 0, self.orig_w, self.orig_h]
            logging.info('ROI has been loaded as full image area.')

        if self.save_vid:
            logging.info('Loading the results capture...')
            result = self.load_save_vid(
                self.save_loc, self.orig_w, self.orig_h)
            logging.info('The results capture has been laoded.')

        frame_count = 0
        total_fps = 0

        start_ends_in_sec = time.time()


        # Run detection
        while (cap.isOpened()):
            
            # Get frame (The slow streaming video capture 
            # can be resolved by:
            # https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync )
            
            ret, self.frame = cap.read()

            # To resize frames
            if (self.max_width !=None) and (self.orig_w != 0) and (self.orig_w > self.max_width):                
                self.frame = cv2.resize(self.frame,(int(self.max_width), int(self.max_width/self.orig_ratio)))

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

                        # draw fps and detections
                        self.out_frame = self.plot_text(self.out_frame, self.orig_w, fps, self.plot_xmin, self.plot_ymin,
                                                        self.plot_padding, self.count.counter_text, self.plot_text_color, self.plot_bgr_color)

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
                            logging.critical(
                                'If show_img = False you need to define "self.ends_in_sec" [secs].')
                            break
                        else:
                            if (end_ends_in_sec - start_ends_in_sec) >= self.ends_in_sec:
                                logging.info(
                                    f'Stopping... The time exeeds the defined excecution time of {self.ends_in_sec} [seconds].')
                                self.stopped = True
                                break

            else:
                logging.info('The video has finished.')
                self.stopped = False
                break

            if self.save_vid:
                result.write(self.out_frame)

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
            logging.info('Calculating the average of fps of the models...')
            self.avg_fps = total_fps / frame_count
            logging.info(
                f'The average of fps of the models is: {round(self.avg_fps,2)} [fps]')

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
