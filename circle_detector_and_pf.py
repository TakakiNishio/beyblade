import cv2
import cv2.cv as cv
import numpy as np
import colorsys
from PIL import Image
import copy
import argparse
from collections import deque


class ParticleFilter:

    def __init__(self,particle_N, image_size):

        self.SAMPLEMAX = particle_N
        self.height = image_size[0]
        self.width = image_size[1]

    def initialize(self):
        self.Y = np.random.random(self.SAMPLEMAX) * self.height
        self.X = np.random.random(self.SAMPLEMAX) * self.width

    # Need adjustment for tracking object velocity
    def modeling(self):
        self.Y += np.random.random(self.SAMPLEMAX) * 200 - 100 # 2:1
        self.X += np.random.random(self.SAMPLEMAX) * 200 - 100

    def normalize(self, weight):
        return weight / np.sum(weight)

    def resampling(self, weight):
        index = np.arange(self.SAMPLEMAX)
        sample = []

        # choice by weight
        for i in range(self.SAMPLEMAX):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)
        return sample

    def calcLikelihood(self, image):
        # white space tracking 
        mean, std = 250.0, 10.0
        intensity = []

        for i in range(self.SAMPLEMAX):
            y, x = self.Y[i], self.X[i]
            if y >= 0 and y < self.height and x >= 0 and x < self.width:
                intensity.append(image[int(y),int(x)])
            else:
                intensity.append(-1)

        # normal distribution
        weights = 1.0 / np.sqrt(2 * np.pi * std) * np.exp(-(np.array(intensity) - mean)**2 /(2 * std**2))
        weights[intensity == -1] = 0
        weights = self.normalize(weights)
        return weights

    def filtering(self, image):
        self.modeling()
        weights = self.calcLikelihood(image)
        index = self.resampling(weights)
        self.Y = self.Y[index]
        self.X = self.X[index]

        # return COG
        return np.sum(self.Y) / float(len(self.Y)), np.sum(self.X) / float(len(self.X))


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
    return (int(hsv[0]), int(hsv[1]), int(hsv[2]))



def hsv_to_bgr(hsv_color):
    bgr = cv2.cvtColor(np.array([[[hsv_color[0], hsv_color[1], hsv_color[2]]]],
                                dtype=np.uint8),cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]),int(bgr[2]))


if __name__ == '__main__':

    # camera
    # cap = cv2.VideoCapture(1)
    # particle_N = 1000
    # image_size = (480, 640)

    # video
    cap = cv2.VideoCapture('videos/test/bey4.avi')
    particle_N = 200
    image_size = (230, 320)

    pf1 = ParticleFilter(particle_N, image_size)
    pf1.initialize()

    trajectory_length = 20
    object_size = 300
    trajectory_points1 = deque(maxlen=trajectory_length)

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


        for i in circles[0,:]:

            # draw the outer circle
            cv2.circle(circle_image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(circle_image,(i[0],i[1]),2,(0,0,255),3)

        # crop image
        left = circles[0][0][0] - circles[0][0][2]/2
        right = circles[0][0][0] + circles[0][0][2]/2

        top = circles[0][0][1] - circles[0][0][2]/2
        bottom = circles[0][0][1] + circles[0][0][2]/2

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
        print

        s_range = 10
        if dominant_hsv[0] < s_range:
            # low_s = dominant_hsv[0]
            low_s = 0
        else:
            low_s = dominant_hsv[0]-s_range

        if dominant_hsv[0]+s_range > 179 :
            high_s = dominant_hsv[0]
        else:
            high_s = dominant_hsv[0] + s_range

        _LOWER_COLOR1 = np.array([low_s,80,80])
        _UPPER_COLOR1 = np.array([high_s,255,255])

        low_bgr = hsv_to_bgr(_LOWER_COLOR1)
        high_bgr = hsv_to_bgr(_UPPER_COLOR1)

        print "lower hsv"
        print _LOWER_COLOR1
        print "upper hsv"
        print _UPPER_COLOR1
        print

        # display doninant color
        size = 200, 200, 3
        dominant_color_display = np.zeros(size, dtype=np.uint8)
        dominant_color_display[:] = dominant_bgr

        # display modified dominant color
        low_bgr_display = np.zeros(size, dtype=np.uint8)
        low_bgr_display[:] = low_bgr

        # display modified dominant color
        high_bgr_display = np.zeros(size, dtype=np.uint8)
        high_bgr_display[:] = high_bgr

        # cv2.imshow("original image", original_image)
        cv2.imshow('detected circles',circle_image)
        cv2.imshow("cropped image", cropped_image)
        cv2.imshow("dominant", dominant_color_display)
        cv2.imshow("low", low_bgr_display)
        cv2.imshow("high", high_bgr_display)


        while True:

            ret, frame = cap.read()
            result_frame = copy.deepcopy(frame)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Threshold the HSV image to get only a color
            mask1 = cv2.inRange(hsv, _LOWER_COLOR1, _UPPER_COLOR1)
            # mask2 = cv2.inRange(hsv, _LOWER_COLOR2, _UPPER_COLOR2)

            # Start Tracking
            y1, x1 = pf1.filtering(mask1)
            # y2, x2 = pf2.filtering(mask2)

            frame_size = frame.shape
            p_range_x1 = np.max(pf1.X)-np.min(pf1.X)
            p_range_y1 = np.max(pf1.Y)-np.min(pf1.Y)
            # p_range_x2 = np.max(pf2.X)-np.min(pf2.X)
            # p_range_y2 = np.max(pf2.Y)-np.min(pf2.Y)

            for i in range(pf1.SAMPLEMAX):
                cv2.circle(result_frame, (int(pf1.X[i]), int(pf1.Y[i])), 2, dominant_bgr, -1)

            # for j in range(pf2.SAMPLEMAX):
            #     cv2.circle(result_frame, (int(pf2.X[j]), int(pf2.Y[j])), 2, (99, 101, 211), -1)

            if p_range_x1 < object_size and p_range_y1 < object_size:

                center1 = (int(x1), int(y1))
                cv2.circle(result_frame, center1, 10, (0, 255, 255), -1)
                trajectory_points1.appendleft(center1)

                for m in range(1, len(trajectory_points1)):
                    if trajectory_points1[m - 1] is None or trajectory_points1[m] is None:
                        continue
                    cv2.line(result_frame, trajectory_points1[m-1], trajectory_points1[m], high_bgr, thickness=3)
            else:
                trajectory_points1 = deque(maxlen=trajectory_length)


            # if p_range_x2 < object_size and p_range_y2 < object_size:

            #     center2 = (int(x2), int(y2))
            #     cv2.circle(result_frame, center2, 10, (0, 255, 255), -1)
            #     trajectory_points2.appendleft(center2)

            #     for n in range(1, len(trajectory_points2)):
            #         if trajectory_points2[n - 1] is None or trajectory_points2[n] is None:
            #             continue
            #         cv2.line(result_frame, trajectory_points2[n-1], trajectory_points2[n], (25, 28, 209), thickness=3)
            # else:
            #     trajectory_points2 = deque(maxlen=trajectory_length)


            cv2.imshow("tracking result", result_frame)

            if cv2.waitKey(20) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
