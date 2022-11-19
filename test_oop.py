from yolov7_sort_count_oop import YoloSortCount
import time

# Test
test = YoloSortCount()

# Show results
test.show_img = True
test.ends_in_sec = 10

# Debug
test.show_configs = False
test.show_detection = False
test.show_tracking = False
test.show_count = True

# Detection model
test.class_ids = [0]
test.conf_thres = 0.5

# Frame
test.inv_h_frame = True
test.roi = [200,200,400,300]


# Run
test.run()
