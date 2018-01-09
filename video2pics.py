import numpy as np
import matplotlib.pyplot as plt

import cv2


def video_to_images(video_path, image_path):

    cap = cv2.VideoCapture(video_path)

    im_cnt = 0

    cv2.namedWindow('video_frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('saved_frame', cv2.WINDOW_NORMAL)

    while(True):
        ret, frame = cap.read()
        key = cv2.waitKey(25) & 0xFF

        if ret is not True:
            break

        cv2.imshow('video_frame', frame)

        if key == ord('a'):
            print cnt

        if key == ord('s'):
            cv2.imshow('saved_frame', frame)
            cv2.imwrite(image_path+str(im_cnt)+'.jpg', frame)
            im_cnt += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    video_path = 'videos/test/bey4.avi'
    image_path = 'images/'

    print video_path

    video_to_images(video_path, image_path)
