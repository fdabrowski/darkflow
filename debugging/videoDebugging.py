import os

import cv2

# vidcap = cv2.VideoCapture('../samples/traffic_3sec.mp4')
# success,image = vidcap.read()

os.system("flow --model cfg/yolo.cfg --load bin/yolo.weights --demo ../samples/traffic_3sec.mp4 --saveVideo --savepb")