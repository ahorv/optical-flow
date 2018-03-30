import cv2
import numpy as np
from os.path import join


######################################################################
## Hoa: 30.11.2017 Version 1 : opticalflow1.py
######################################################################
# Uses Optical Flow to estimate the motion direction of clouds
# Source: https://github.com/LeeChongkai/-Cloud-motion-estimation/blob/master/Cloud%20motion%20estimation(algo1).py
#
#
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 22.11.2017 : New
#
#
######################################################################


global input_path
global output_path
input_path  = r'C:\Hoa_Python_Projects\optflow\input'   # @ home
output_path = r'C:\Hoa_Python_Projects\optflow\output'  # @ home

#input_path  = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\input'  # @ Lab
#output_path = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\output' # @ Lab

print('Start optflow1.py')

e1 = cv2.getTickCount()
frame1 = cv2.imread(join(input_path ,'raw_img6.jpg'),1)
frame2 = cv2.imread(join(input_path ,'raw_img9.jpg'),1)

'''
cv2.imshow('Frame 1', frame1)
cv2.imshow('Frame 2', frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

frame3 = np.zeros_like(frame1)  # normalize an empty array

flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

flowY = flow[..., 0]
flowX = flow[..., 1]

n = np.shape(frame1)[0]
m = np.shape(frame1)[1]

integer = 5

# predict third image
for x in range(0, n):
    for y in range(0, m):
        value = frame2[x, y]
        shiftX = flowX[x, y]
        shiftY = flowY[x, y]
        xn = max((0, min((n - 1, (x + int(np.rint(integer * shiftX)))))))
        yn = max((0, min((m - 1, (y + int(np.rint(integer * shiftY)))))))
        frame3[xn, yn] = value

# Using neighbour pixel value to fill up the empty pixel
for a in range(0, n):
    for b in range(0, m):
        k = frame3[a, b]

        if np.all(k == 0):
            if ((a + 1) > (n - 1)):
                a1 = frame3[a, b]
            else:
                a1 = frame3[(a + 1), b]

            if ((a - 1) < 0):
                a2 = frame3[a, b]
            else:
                a2 = frame3[(a - 1), b]

            if ((b + 1) > (n - 1)):
                b1 = frame3[a, b]
            else:

                b1 = frame3[a, (b + 1)]

            if ((b - 1) < 0):
                b2 = frame3[a, b]
            else:
                b2 = frame3[a, (b - 1)]

            a1 = a1.astype('uint16')
            a2 = a2.astype('uint16')
            b1 = b1.astype('uint16')
            b2 = b2.astype('uint16')

            avg_value = (a1 + a2 + b1 + b2) / np.max(
                [np.count_nonzero([np.sum(a1), np.sum(a2), np.sum(b1), np.sum(b2)]), 1])

            frame3[a, b] = avg_value.astype('uint8')

cv2.imshow('Opticalflow',frame3)
cv2.imwrite(join(output_path,'OptFlowFrame3.jpg'), frame3)



e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
print ('Time needed to calculate: {}'.format(str(time)))


cv2.imwrite('optflow1.jpg', frame3)
cv2.imshow('Opticalflow', frame3)
cv2.waitKey(0)
cv2.destroyAllWindows()
