from yolov7_sort_count_oop import YoloSortCount

# Test
test = YoloSortCount()

test.show_configs = True
test.class_ids = []
test.roi = [0,0,100,100]

print(test)
print(test.detection)
print(test.tracking)
print(test.count)

test.run()
