from yolov7_sort_count_oop import YoloSortCount

# Test
test = YoloSortCount()

# Debug
test.show_configs = False
test.show_detection = False
test.show_tracking = False
test.show_count = False

# Detection model
test.class_ids = []
test.conf_thres = 0.5

# Frame
test.inv_h_frame = True
test.roi = [200,200,400,300]


# Run
test.run()
