import cv2
import numpy as np

def zebra(img):
    # image = cv2.imread('zebra.jpg', -1)
    image=img
    # height, width, number of channels in image
    height = image.shape[0]
    width = image.shape[1]
    paper = cv2.resize(image, (width,height))
    ret, thresh_gray = cv2.threshold(cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Erase small contours, and contours which small aspect ratio (close to a square)
    for c in contours:
        area = cv2.contourArea(c)

        # Fill very small contours with zero (erase small contours).
        if area < 10:
            cv2.fillPoly(thresh_gray, pts=[c], color=0)
            continue

        # https://stackoverflow.com/questions/52247821/find-width-and-height-of-rotatedrect
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)

        # Assume zebra line must be long and narrow (long part must be at lease 1.5 times the narrow part).
        if (aspect_ratio < 1.5):
            cv2.fillPoly(thresh_gray, pts=[c], color=0)
            continue


    # Use "close" morphological operation to close the gaps between contours
    # https://stackoverflow.com/questions/18339988/implementing-imcloseim-se-in-opencv
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51,51)));

    # Find contours in thresh_gray after closing the gaps
    contours, hier = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        area = cv2.contourArea(c)

        # Small contours are ignored.
        if area < 20000:
            cv2.fillPoly(thresh_gray, pts=[c], color=0)
            continue

        rect = cv2.minAreaRect(c)
        print(rect)
        box = cv2.boxPoints(rect)
        print(box)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        list=box.tolist()
        list=list[0]+list[2]
        print(list)
        cv2.drawContours(paper, [box], 0, (0, 255, 0),1)

    return list

