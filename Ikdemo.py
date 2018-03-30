#! /usr/bin/env python

#Quelle: https://github.com/npinto/opencv/blob/master/samples/python/lkdemo.py

global input_path
global output_path
#input_path  = r'C:\Hoa_Python_Projects\python_scripts\WAHRSIS\input'  # @ home
#output_path = r'C:\Hoa_Python_Projects\python_scripts\optflow\output'  # @ home
output_path = r'C:\Users\tahorvat\PycharmProjects\python_scripts\optflow\output' # @ Lab
input_path  = r'C:\Users\tahorvat\PycharmProjects\python_scripts\optflow\input'  # @ Lab

from os.path import join
import numpy as np
import cv2


#############################################################################
# some "constants"

win_size = 10
MAX_COUNT = 500

#############################################################################
# some "global" variables

image = None
pt = None
add_remove_pt = False
flags = 0
night_mode = False
need_to_init = False


#############################################################################
# the mouse callback

# the callback on the trackbar
def on_mouse(event, x, y, flags, param):
    # we will use the global pt and add_remove_pt
    global pt
    global add_remove_pt

    if image is None:
        # not initialized, so skip
        return

    if image.origin != 0:
        # different origin
        y = image.height - y

    if event == cv2.CV_EVENT_LBUTTONDOWN:
        # user has click, so memorize it
        pt = (x, y)
        add_remove_pt = True

def getSize(img):
    im = np.asarray(img)
    width, height, colors = im.shape
    return width,height


#############################################################################

def main():
    try:
        image = None

        print("OpenCV Python version of lkdemo")
        frame1 = cv2.imread(join(input_path, 'raw_img4.jpg'), 1)
        frame2 = cv2.imread(join(input_path, 'raw_img9.jpg'), 1)

        frames = [frame1,frame2]


        # first, create the necessary windows
        cv2.namedWindow('LkDemo', cv2.WINDOW_NORMAL)

        # register the mouse callback
      #  cv2.setMouseCallback('LkDemo', on_mouse(0,0,None,None),None)

        fc = 0
        while 1:
            # do forever

            frame = frames[fc]

            if image is None:
                # create the images we need
                print(getSize(frame))

                image = cv2.cv.resize(cv2.GetSize(frame), 8, 3)
                image.origin = frame.origin
                grey = cv2.resize(cv2.GetSize(frame), 8, 1)
                prev_grey = cv2.resize(cv2.GetSize(frame), 8, 1)
                pyramid = cv2.resize(cv2.GetSize(frame), 8, 1)
                prev_pyramid = cv2.resize(cv2.GetSize(frame), 8, 1)
                features = []

            # copy the frame, so we can draw on it
            cv2.Copy(frame, image)

            # create a grey version of the image
            cv2.CvtColor(image, grey, cv2.CV_BGR2GRAY)

            if night_mode:
                # night mode: only display the points
                cv2.SetZero(image)

            if need_to_init:
                # we want to search all the good points

                # create the wanted images
                eig = cv2.CreateImage(cv2.GetSize(grey), 32, 1)
                temp = cv2.CreateImage(cv2.GetSize(grey), 32, 1)

                # the default parameters
                quality = 0.01
                min_distance = 10

                # search the good points
                features = cv2.GoodFeaturesToTrack(
                    grey, eig, temp,
                    MAX_COUNT,
                    quality, min_distance, None, 3, 0, 0.04)

                # refine the corner locations
                features = cv2.FindCornerSubPix(
                    grey,
                    features,
                    (win_size, win_size), (-1, -1),
                    (cv2.CV_TERMCRIT_ITER | cv2.CV_TERMCRIT_EPS, 20, 0.03))

            elif features != []:
                # we have points, so display them

                # calculate the optical flow
                features, status, track_error = cv2.CalcOpticalFlowPyrLK(
                    prev_grey, grey, prev_pyramid, pyramid,
                    features,
                    (win_size, win_size), 3,
                    (cv2.CV_TERMCRIT_ITER | cv2.CV_TERMCRIT_EPS, 20, 0.03),
                    flags)

                # set back the points we keep
                features = [p for (st, p) in zip(status, features) if st]

                if add_remove_pt:
                    # we have a point to add, so see if it is close to
                    # another one. If yes, don't use it
                    def ptptdist(p0, p1):
                        dx = p0[0] - p1[0]
                        dy = p0[1] - p1[1]
                        return dx ** 2 + dy ** 2


                    if min([ptptdist(pt, p) for p in features]) < 25:
                        # too close
                        add_remove_pt = 0

                # draw the points as green circles
                for the_point in features:
                    cv2.Circle(image, (int(the_point[0]), int(the_point[1])), 3, (0, 255, 0, 0), -1, 8, 0)

            if add_remove_pt:
                # we want to add a point
                # refine this corner location and append it to 'features'

                features += cv2.FindCornerSubPix(
                    grey,
                    [pt],
                    (win_size, win_size), (-1, -1),
                    (cv2.CV_TERMCRIT_ITER | cv2.CV_TERMCRIT_EPS,
                     20, 0.03))
                # we are no longer in "add_remove_pt" mode
                add_remove_pt = False

            # swapping
            prev_grey, grey = grey, prev_grey
            prev_pyramid, pyramid = pyramid, prev_pyramid
            need_to_init = False

            # we can now display the image
            cv2.ShowImage('LkDemo', image)

            # handle events
            c = cv2.WaitKey(10) % 0x100

            if c == 27:
                # user has press the ESC key, so exit
                break

            # processing depending on the character
            if 32 <= c and c < 128:
                cc = chr(c).lower()
                if cc == 'r':
                    need_to_init = True
                elif cc == 'c':
                    features = []
                elif cc == 'n':
                    night_mode = not night_mode
                elif cc == ' ':
                    fc = (fc + 1) % len(frames)
            cv2.DestroyAllWindows()

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))

if __name__ == '__main__':
    main()