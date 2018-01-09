import numpy as np
import cv2
import copy

#blue
_LOWER_COLOR = np.array([80, 50, 50])
_UPPER_COLOR = np.array([110, 255, 255])

#red
# _LOWER_COLOR = np.array([0, 50, 50])
# _UPPER_COLOR = np.array([10, 255, 255])


class ParticleFilter:
    def __init__(self):
        self.SAMPLEMAX = 1000
        # frame.shape
        self.height, self.width = 480, 640

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
    cap = cv2.VideoCapture(1)

    pf = ParticleFilter()
    pf.initialize()

    trajectory_points = []

    while True:
        ret, frame = cap.read()
        # cv2.imshow("frame", frame)
        result_frame = copy.deepcopy(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only a color
        mask = cv2.inRange(hsv, _LOWER_COLOR, _UPPER_COLOR)

        # Start Tracking
        y, x = pf.filtering(mask)

        # origin is upper left
        frame_size = frame.shape
        p_range_x = np.max(pf.X)-np.min(pf.X)
        p_range_y = np.max(pf.Y)-np.min(pf.Y)
        # print "position_x_rate"
        # print x/frame_size[1]
        # print "position_y_rate"
        # print y/frame_size[0]

        for i in range(pf.SAMPLEMAX):
            cv2.circle(result_frame, (int(pf.X[i]), int(pf.Y[i])), 2, (0, 255, 0), -1)

        if p_range_x < 300 and p_range_y < 300:

            cv2.circle(result_frame, (int(x), int(y)), 10, (255, 0, 0), -1)
            trajectory_points.append((int(x), int(y)))

            for i in range(1, len(trajectory_points)):
                if trajectory_points[i - 1] is None or trajectory_points[i] is None:
                    continue
                cv2.line(result_frame, trajectory_points[i-1], trajectory_points[i], (0, 255, 255), thickness=2)
        else:
            trajectory_points = []

        cv2.imshow("result", result_frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tracking()
