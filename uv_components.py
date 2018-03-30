import cv2
import numpy as np
from os.path import join

global input_path
global output_path
#input_path  = r'C:\Hoa_Python_Projects\python_scripts\optflow\input'  # @ home
#output_path = r'C:\Hoa_Python_Projects\python_scripts\optflow\output'  # @ home

input_path  = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\input'  #@ Lab
output_path = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\output' #@ Lab

path_frame1 = join(input_path, 'raw_img6.jpg')
path_frame2 = join(input_path, 'raw_img9.jpg')


frame1 = cv2.imread(path_frame1)
frame2 = cv2.imread(path_frame2)

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Change here
horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
horz = horz.astype('uint8')
vert = vert.astype('uint8')

# Change here too
cv2.imshow('Horizontal Component', horz)
cv2.imshow('Vertical Component', vert)

cv2.waitKey(0)
cv2.destroyAllWindows()