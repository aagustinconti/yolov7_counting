from yolov7_sort_count_oop import YoloSortCount

# Test
test = YoloSortCount()

test.show_configs = True
test.class_ids = []

test.inv_h_frame = True
test.roi = [200,200,400,300]

test.conf_thres = 0.5



print(test.run())
