import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
from p_crossing import color_detect
from p_crossing import zebra_crossing

def detection():

    cap = cv2.VideoCapture("../Videos/caught.mp4")  # For Video

    model = YOLO("../Yolo-Weights/yolov8n.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    #mask = cv2.imread("mask.png")

    # Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    flag= False
    # limits = [300, 497, 973, 497]
    # point1=[300,497]
    # point2=[973,497]
    success, img = cap.read()
    list=zebra_crossing.zebra(img)
    print(list)

    limits = list
    point1=limits[0:2]
    print("point 1",point1)
    point2=limits[2:]
    print("point 2",point2)
    totalCount = []
    dcnt=1
    while True:
        success, img = cap.read()
        #imgRegion = cv2.bitwise_and(img,mask)
        car_cascade = cv2.CascadeClassifier('cars.xml')

        if point1 and point2:

            # Rectangle marker
            r1 = cv2.rectangle(img, point1, point2, (100, 50, 200), 5)
            frame_ROI = img[point1[1]:point2[1], point1[0]:point2[0]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cimg = img


        results = model(img, stream=True)

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

                if currentClass == "traffic light" and conf > 0.3:
                    # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                    #                    scale=0.6, thickness=1, offset=3)
                    cvzone.cornerRect(img, (x1, y1, w, h), l=0, rt=0)
                    cimg = img[y1:y2, x1:x2]

                    # print("cimg", cimg)
                    cv2.imwrite("Detected Images/traffic_light.jpg", cimg)
                    a = color_detect.color_detection()
                    if a==1:
                        flag=True
                        cv2.putText(img, "RED", (20, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
                    if a==2:
                        flag=False
                        cv2.putText(img, "GREEN", (105, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 8)
                    print(a)


                if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                        or currentClass == "motorbike"  and conf > 0.3:
                    # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                    #                    scale=0.6, thickness=1, offset=3)
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            print(flag)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] - 5 < cy < limits[1] + 5 and flag==True:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))
                    cimg = img[y1:y2,x1:x2]
                    print("cimg",cimg)
                    cv2.imwrite("Detected Images/violation_" + str(dcnt) + ".jpg", cimg)
                    dcnt = dcnt + 1
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
        cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),8)

        cv2.imshow("Image", img)
        # cv2.imshow("ImageRegion", imgRegion)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
