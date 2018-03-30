import cv2
import numpy as np
from os.path import join
from matplotlib import pyplot as plt


######################################################################
## Hoa: 30.11.2017 Version 1 : optflow2.py
######################################################################
# Uses Optical Flow to estimate the motion direction of clouds
# Source: https://github.com/npinto/opencv/blob/master/samples/python2/opt_flow.py
#
#
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 30.11.2017 : New
#
#
######################################################################

global input_path
global output_path
input_path  = r'C:\Hoa_Python_Projects\optflow\input'  # @ home
output_path = r'C:\Hoa_Python_Projects\optflow\output'  # @ home
#output_path = r'C:\Users\tahorvat\PycharmProjects\python_scripts\optflow\output' #@ Lab
#input_path  = r'C:\Users\tahorvat\PycharmProjects\python_scripts\optflow\input' # @Lab


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    fx, fy = flow[y.astype(np.int64), x.astype(np.int64)].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def main():
    try:
        print("Start optflow2.py")

        e1 = cv2.getTickCount()
        frame1 = cv2.imread(join(input_path, 'raw_img4.jpg'), 1)
        frame1= cv2.resize(frame1, None, fx=0.25, fy=0.25)
        frame2 = cv2.imread(join(input_path, 'raw_img9.jpg'), 1)
        frame2 = cv2.resize(frame2, None, fx=0.25, fy=0.25)
        prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        show_hsv = False
        show_glitch = True

        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        imgWithFlow = draw_flow(next, flow)

        fig1 = plt.figure(1)
        plt.title('Optical Flow')
        plt.imshow(imgWithFlow)

        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
        print('Time needed to calculate: {}'.format(str(time)))

        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(prev, flow)
            cv2.imshow('glitch', cur_glitch)

        cv2.imshow('flow', imgWithFlow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(join(output_path, 'optflow2.jpg'), imgWithFlow)

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()