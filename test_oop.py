from yolov7_sort_count_oop import YoloSortCount

# Test
test = YoloSortCount()

test.show_configs = True
test.show_detections = True
test.show_tracking = True
test.show_count = True

test.class_ids = [0]

print(test)

test.run()
