from yolov7_sort_count_oop import YoloSortCount

#################### TEST ####################

# INSTANCIATE

test = YoloSortCount()


"""
###### AVAILABLE SOURCES ###### 
WebCamera: 0 ---> DEFAULT
Youtube Video or stream: "https://www.youtube.com/watch?v=qP1y7Tdab7Y"
Stream URL:  "http://IP/hls/stream_src.m3u8"
RSTP Stream: 
Local video: "img_bank/cows_for_sale.mp4"
Local image: "img_bank/img.jpg" | "img_bank/img.png" 
"""
test.video_path = 0


"""
###### FRAME PROPERTIES ###### 

- Set the max size of the frame (width)
- Set the max fps of the video
- Invert the image (In case of your WebCamera is mirrored, IE)	
"""
test.max_width = 720
test.max_fps = 25  # Max 1000
test.inv_h_frame = True


"""
###### SHOWING RESULTS ###### 

- Show the results in your display (Interactive ROI, imshow of the out frame)
- In case of you are not showing the results, set the timer to stop the execution.
- Stop the frame with hold_image method in case you are using image as a source.	
"""
test.show_img = True
test.ends_in_sec = 10
test.hold_img = False


"""
###### ROI ###### 

- Load the ROI manually.
- 
- Load the ROI color.	
"""
#test.roi = [0,0,0,0]
test.auto_load_roi = True
test.roi_color = (255, 255, 255)


"""
###### DETECTION MODEL ###### 

- Specify the path of the model.
- Select the ID of your Graphic Card (nvidia-smi)
- Select the classes to detect	
- Set the image size (Check if the YOLO model allows that --> IE: yolov7.pt 640, yolov7-w6.pt 1280 or 640)
- Set the bounding box color
- Set the minimum confidence to detect.
- Set the minimum overlap of a predicted versus actual bounding box for an object.
"""
test.model_path = 'pretrained_weights/yolov7.pt'
test.graphic_card = 0
test.class_ids = [0]
test.img_sz = 640
test.color = (0, 255, 0)
test.conf_thres = 0.5
test.iou_thres = 0.65

"""
###### TRACKING MODEL ###### 

- Specify the path of the model.
- Set the max distance between two points to consider a tracking object.
- Set the max overlap of a predicted versus actual bounding box for an object.
- Set the image size (Check if the YOLO model allows that --> IE: yolov7.pt 640, yolov7-w6.pt 1280 or 640)
- Set max_age to consider a lost of a tracking object that get out of the seen area.
- Set the minimum frames to start to track objects.
- Set the value that indicates how many previous frames of feature vectors should be retained for distance calculation for each track.
- Set the color of the centroid and label of a tracking object.
"""
test.deep_sort_model = "osnet_x1_0"
test.ds_max_dist = 0.1
test.ds_max_iou_distance = 0.7
test.ds_max_age = 30
test.ds_n_init = 3
test.ds_nn_budget = 100
test.ds_color = (0, 0, 255)

"""
###### PLOT RESULTS ###### 

- Specify the min x (left to right) to plot the draws
- Specify the min y (top to bottom) to plot the draws
- Specify padding between rectangles and text
- Specify the text color.
- Specify the rectangles color.
"""
test.plot_xmin = 10
test.plot_ymin = 10
test.plot_padding = 2
test.plot_text_color = (255, 255, 255)
test.plot_bgr_color = (0, 0, 0)


"""
###### DEBUG TEXT ###### 

- Show the configs
- Show the detection output variables
- Show the tracking output variables
- Show counting output variables
"""
test.show_configs = False
test.show_detection = False
test.show_tracking = False
test.show_count = False

"""
###### SAVING RESULTS ###### 

- Select if you want to save the results
- Select a location to save the results
"""
test.save_vid = False
test.save_loc = "results/result"


# Run
test.run()
