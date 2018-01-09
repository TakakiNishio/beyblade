import numpy as np
import cv2
import copy
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


def tracking():

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

    pf2 = ParticleFilter(particle_N, image_size)
    pf2.initialize()

    trajectory_length = 20
    object_size = 300
    trajectory_points1 = deque(maxlen=trajectory_length)
    trajectory_points2 = deque(maxlen=trajectory_length)

    #blue
    _LOWER_COLOR1 = np.array([80, 50, 50])
    _UPPER_COLOR1 = np.array([110, 255, 255])

    #red
    _LOWER_COLOR2 = np.array([0, 50, 50])
    _UPPER_COLOR2 = np.array([10, 255, 255])

    while True:

        ret, frame = cap.read()
        result_frame = copy.deepcopy(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only a color
        mask1 = cv2.inRange(hsv, _LOWER_COLOR1, _UPPER_COLOR1)
        mask2 = cv2.inRange(hsv, _LOWER_COLOR2, _UPPER_COLOR2)

        # Start Tracking
        y1, x1 = pf1.filtering(mask1)
        y2, x2 = pf2.filtering(mask2)

        frame_size = frame.shape
        p_range_x1 = np.max(pf1.X)-np.min(pf1.X)
        p_range_y1 = np.max(pf1.Y)-np.min(pf1.Y)
        p_range_x2 = np.max(pf2.X)-np.min(pf2.X)
        p_range_y2 = np.max(pf2.Y)-np.min(pf2.Y)

        for i in range(pf1.SAMPLEMAX):
            cv2.circle(result_frame, (int(pf1.X[i]), int(pf1.Y[i])), 2, (216, 174, 47), -1)

        for j in range(pf2.SAMPLEMAX):
            cv2.circle(result_frame, (int(pf2.X[j]), int(pf2.Y[j])), 2, (99, 101, 211), -1)

        if p_range_x1 < object_size and p_range_y1 < object_size:

            center1 = (int(x1), int(y1))
            cv2.circle(result_frame, center1, 10, (0, 255, 255), -1)
            trajectory_points1.appendleft(center1)

            for m in range(1, len(trajectory_points1)):
                if trajectory_points1[m - 1] is None or trajectory_points1[m] is None:
                    continue
                cv2.line(result_frame, trajectory_points1[m-1], trajectory_points1[m], (219, 71, 48), thickness=3)
        else:
            trajectory_points1 = deque(maxlen=trajectory_length)


        if p_range_x2 < object_size and p_range_y2 < object_size:

            center2 = (int(x2), int(y2))
            cv2.circle(result_frame, center2, 10, (0, 255, 255), -1)
            trajectory_points2.appendleft(center2)

            for n in range(1, len(trajectory_points2)):
                if trajectory_points2[n - 1] is None or trajectory_points2[n] is None:
                    continue
                cv2.line(result_frame, trajectory_points2[n-1], trajectory_points2[n], (25, 28, 209), thickness=3)
        else:
            trajectory_points2 = deque(maxlen=trajectory_length)


        cv2.imshow("tracking result", result_frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tracking()
