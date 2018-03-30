import cv2
import glob, os
from os.path import join
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

global input_path
global output_path

rpicam_path  = r'C:\Hoa_Python_Projects\optflow\input\RPICam 2017.10.27'  # @ home
input_path  = r'C:\Hoa_Python_Projects\optflow\input'   # @ home
output_path = r'C:\Hoa_Python_Projects\optflow\output'  # @ home

#input_path  = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\input'  # @ Lab
#output_path = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\output' # @ Lab


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

def readImages():
    list_names = []

    for file in sorted(glob.glob(os.path.join(rpicam_path,'*.jpg'))):
          list_names.append(file)

    return list_names

def main():
    try:
        print("Start animoptflow.py")

        fig = plt.figure(1)
        plt.title('Optical Flow')

        list_names = readImages()

        frame1 = cv2.imread(join(input_path,list_names[0]), 1)
        frame1 = cv2.resize(frame1, None, fx=0.25, fy=0.25)
        prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

       #LOESUNG : https://stackoverflow.com/questions/33602185/speed-up-plotting-images-in-matplotlib

        counter = 1

        anims = []

        while counter < 500: #len(list_names):

            print('{} Frame'.format(counter))

            frame2 = cv2.imread(list_names[counter],1)
            frame2 = cv2.resize(frame2, None, fx=0.25, fy=0.25)
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            imgWithFlow = draw_flow(next, flow)

            plt.imshow(prev)

            plt.pause(.01)
            plt.draw()

            # Make next frame previous frame
            prev = next.copy()
            counter += 1




        print('done !')

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()




