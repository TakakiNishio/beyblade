import cv2
import cv2.cv as cv
import numpy as np
import colorsys
from PIL import Image
import copy
import argparse


def get_dominant_color(image):
    """
    Find a PIL image's dominant color, returning an (r, g, b) tuple.
    """
    image = image.convert('RGBA')
    # Shrink the image, so we don't spend too long analysing color
    # frequencies. We're not interpolating so should be quick.
    image.thumbnail((200, 200))
    max_score = None
    dominant_color = None

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # Skip 100% transparent pixels
        if a == 0:
            continue
        # Get color saturation, 0-1
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        # Calculate luminance - integer YUV conversion from
        # http://en.wikipedia.org/wiki/YUV
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        # Rescale luminance from 16-235 to 0-1
        y = (y - 16.0) / (235 - 16)
        # Ignore the brightest colors
        if y > 0.9:
            continue
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = [b, g, r]

    return dominant_color


def bgr_to_hsv(bgr_color):
    hsv = cv2.cvtColor(np.array([[[bgr_color[0], bgr_color[1], bgr_color[2]]]],
                                dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    return [hsv[0], hsv[1], hsv[2]]


def hsv_to_bgr(hsv_color):
    bgr = cv2.cvtColor(np.array([[[hsv_color[0], hsv_color[1], hsv_color[2]]]],
                                dtype=np.uint8),cv2.COLOR_HSV2BGR)[0][0]
    return [bgr[0], bgr[1], bgr[2]]


if __name__ == '__main__':

    cap = cv2.VideoCapture('videos/test/bey4.avi')

    while True:

        ret, frame = cap.read()
        print frame.shape

        original_image = frame

        circle_image = copy.deepcopy(original_image)

        gray_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

        blur_image = cv2.medianBlur(gray_image,5)

        circles = cv2.HoughCircles(blur_image,cv.CV_HOUGH_GRADIENT,1,40,
                                   param1=60,param2=35,minRadius=10,maxRadius=40)

        if circles == None:
            continue

        circles = np.uint16(np.around(circles))

        print len(circles[0,:])

        for i in circles[0,:]:

            # draw the outer circle
            cv2.circle(circle_image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(circle_image,(i[0],i[1]),2,(0,0,255),3)

        # crop image
        left = circles[0][0][0] - circles[0][0][2]
        right = circles[0][0][0] + circles[0][0][2]

        top = circles[0][0][1] - circles[0][0][2]
        bottom = circles[0][0][1] + circles[0][0][2]

        cropped_image = original_image[top:bottom, left:right]

        # convert the image into PIL image format
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # detect the dominant color
        dominant_bgr = get_dominant_color(pil_image)

        # convert BGR to HSV
        dominant_hsv = bgr_to_hsv(dominant_bgr)

        print "dominant bgr"
        print dominant_bgr

        print "dominant hsv"
        print dominant_hsv

        low_brightness = copy.deepcopy(dominant_hsv)
        high_brightness = copy.deepcopy(dominant_hsv)

        # modify V value
        v_range = 50
        low_brightness[2] = low_brightness[2] - v_range
        high_brightness[2] = high_brightness[2] + v_range
        low_brightness = hsv_to_bgr(low_brightness)
        high_brightness = hsv_to_bgr(high_brightness)

        # display doninant color
        size = 200, 200, 3
        dominant_color_display = np.zeros(size, dtype=np.uint8)
        dominant_color_display[:] = dominant_bgr

        # display modified dominant color
        low_brightness_display = np.zeros(size, dtype=np.uint8)
        low_brightness_display[:] = low_brightness

        # display modified dominant color
        high_brightness_display = np.zeros(size, dtype=np.uint8)
        high_brightness_display[:] = high_brightness

        # cv2.imshow("original image", original_image)
        cv2.imshow('detected circles',circle_image)
        cv2.imshow("cropped image", cropped_image)
        cv2.imshow("dominant", dominant_color_display)
        cv2.imshow("low", low_brightness_display)
        cv2.imshow("high", high_brightness_display)

        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            break

    cap.release()
