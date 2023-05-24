
import cvzone
import math
import time
import numpy as np

import numpy as np
from ultralytics import YOLO
import cv2

def color_detection():
    model = YOLO('C:/Users/admin/OneDrive/Documents/College/Final Project/6th Sem Project/Yolo-Weights/best.pt')
    color=0
    # image = cv2.imread('Detected Images/traffic_light.jpg')
    results = model('Detected Images/traffic_light.jpg', show=False)
    # height, width, number of channels in image
    classNames = ['bus_stop', 'do_not_enter','do_not_stop','do_not_turn_l','do_not_turn_r','do_not_u_turn', 'enter_left_lane','green_light','left_right_lane', 'no_parking', 'parking', 'ped_crossing','yellow_light','ped_zebra_cross', 'railway_crossing',  'red_light', 'stop','traffic_light','u_turn', 'warning']
    # names: {'bus_stop', 'do_not_enter','do_not_stop','do_not_turn_l','do_not_turn_r','do_not_u_turn', 'enter_left_lane','green_light','left_right_lane', 'no_parking', 'parking', 'ped_crossing', 'ped_zebra_cross', 'railway_crossing',  'red_light', 'stop','traffic_light','u_turn', 'warning','yellow_light'}

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            print(currentClass)
            # cvzone.putTextRect(image, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            if currentClass=="stop" or currentClass=="red_light" or currentClass=="yellow_light":
                color=1
            if currentClass=="green_light":
                color=2

    print(color)
    return color
