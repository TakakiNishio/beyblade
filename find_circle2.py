import cv2
import cv2.cv as cv
import numpy as np
import argparse

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    original_image = cv2.imread(args["image"])

    gray_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

    blur_image = cv2.medianBlur(gray_image,3)

    circles = cv2.HoughCircles(blur_image,cv.CV_HOUGH_GRADIENT,1,40,
                            param1=60,param2=35,minRadius=10,maxRadius=40)

    circles = np.uint16(np.around(circles))
    print circles
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(original_image,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(original_image,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',original_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
