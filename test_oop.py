from yolov7_sort_count_oop import YoloSortCount
import time
import cv2

# Test
test = YoloSortCount()

# Source
test.video_path = 0#"https://www.youtube.com/watch?v=qP1y7Tdab7Y" | "http://IP/hls/stream_src.m3u8" | 0 | "img_bank/cows_for_sale.mp4"

test.max_fps = 1000 #Max 1000
test.max_width = 720

# Show results
test.show_img = True
test.hold_img = False

test.auto_load_roi = True

test.ends_in_sec = 10

# Debug
test.show_configs = False
test.show_detection = False
test.show_tracking = False
test.show_count = False

# Detection model
test.class_ids = [0]
test.conf_thres = 0.5

# Frame
test.inv_h_frame = False

# Save
test.save_loc = "results/test_test"
test.save_vid = True

# Run
test.run()