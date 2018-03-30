import numpy as np
import cv2
from matplotlib import pyplot as plt
#import matplotlib.pylab as plt (Alternative) # https://solarianprogrammer.com/2017/02/25/install-numpy-scipy-matplotlib-python-3-windows/
from os.path import join
from sklearn.cluster import KMeans

#Quelle: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

# @ LAB use interpreter: C:\Anaconda3\envs\py34\python.exe

global input_path
global output_path
#input_path  = r'C:\Hoa_Python_Projects\python_scripts\WAHRSIS\input'  # @ home
#output_path = r'C:\Hoa_Python_Projects\python_scripts\optflow\output'  # @ home
output_path = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\output' #@ Lab
input_path  = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\input' # @Lab


def cmask(index, radius, array):
    """Generates the mask for a given input image.
    The generated mask is needed to remove occlusions during post-processing steps.

    Args:
        index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
        radius (float): Radius of the circular mask.
        array (numpy array): Input sky/cloud image for which the mask is generated.

    Returns:
        numpy array: Generated mask image."""

    a, b = index
    is_rgb = len(array.shape)

    if is_rgb == 3:
        ash = array.shape
        nx = ash[0]
        ny = ash[1]
    else:
        nx, ny = array.shape

    s = (nx, ny)
    image_mask = np.zeros(s)
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= radius * radius
    image_mask[mask] = 1

    return (image_mask)

def getBRchannel(input_image, mask_image):
    """Extracts the ratio of red and blue blue channel from an input sky/cloud image.
    It is used in the clustering step to generate the binary sky/cloud image.

    Args:
        input_image (numpy array): Input sky/cloud image.
        mask_image (numpy array): Mask to remove occlusions from the input image.
        This mask contains boolean values indicating the allowable pixels from an image.

    Returns:
        numpy array: Ratio image using red and blue color channels, normalized to [0,255]."""

    red = input_image[:, :, 2]
    green = input_image[:, :, 1]
    blue = input_image[:, :, 0]

    # RGB images for transfer
    red_image = red.astype(float) * mask_image
    green_image = green.astype(float) * mask_image
    blue_image = blue.astype(float) * mask_image

    BR = (blue_image - red_image) / (blue_image + red_image)
    BR[np.isnan(BR)] = 0

    return showasImage(BR)

def showasImage(input_matrix):
    """Normalizes an input matrix to the range [0,255]. It is useful in displaying the matrix as an image.

    Args:
        input_matrix (numpy array): Input matrix that needs to be normalized.

    Returns:
        numpy array: Returns the normalized matrix."""


    return (input_matrix - np.amin(input_matrix)) / (np.amax(input_matrix) - np.amin(input_matrix)) * 255

def make_cluster_mask(input_matrix, mask_image):
    """Clusters an input sky/cloud image to generate the binary image and the coverage ratio value.

    Args:
        input_matrix (numpy array): Input matrix that needs to be normalized.
        mask_image (numpy array): Mask to remove occlusions from the input image. This mask contains boolean values indicating the allowable pixels from an image.

    Returns:
        numpy array: Binary output image, where white denotes cloud pixels and black denotes sky pixels.
        float: Cloud coverage ratio in the input sky/cloud image.
        float: The first (out of two) cluster center.
        float: The second (out of two) cluster center."""

    [rows, cols] = mask_image.shape

    im_mask_flt = mask_image.flatten()
    find_loc = np.where(im_mask_flt == 1)
    find_loc = list(find_loc)

    input_vector = input_matrix.flatten()

    input_select = input_vector[list(find_loc)]

    X = input_select.reshape(-1, 1)
    k_means = KMeans(init='k-means++', n_clusters=2)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    center1 = k_means_cluster_centers[0]
    center2 = k_means_cluster_centers[1]

    if center1 < center2:
        # Interchange the levels.
        temp = center1
        center1 = center2
        center2 = temp

        k_means_labels[k_means_labels == 0] = 99
        k_means_labels[k_means_labels == 1] = 0
        k_means_labels[k_means_labels == 99] = 1

    cent_diff = np.abs(center1 - center2)
    if cent_diff < 20:
        # Segmentation not necessary.
        if center1 > 120 and center2 > 120:
            # All cloud image
            k_means_labels[k_means_labels == 0] = 1
        else:
            # All sky image
            k_means_labels[k_means_labels == 1] = 0

    # 0 is sky and 1 is cloud
    cloud_pixels = np.count_nonzero(k_means_labels == 1)
    sky_pixels = np.count_nonzero(k_means_labels == 0)
    total_pixels = cloud_pixels + sky_pixels

    # print (cloud_pixels,total_pixels)
    cloud_coverage = float(cloud_pixels) / float(total_pixels)

    # Final threshold image for transfer
    index = 0
    Th_image = np.zeros([rows, cols])
    for i in range(0, rows):
        for j in range(0, cols):

            if mask_image[i, j] == 1:
                # print (i,j)
                # print (index)
                Th_image[i, j] = k_means_labels[index]
                index = index + 1


    return (Th_image, cloud_coverage, center1, center2)

def segment(th_img, coverage, center1, center2, img, mask):

    cent_diff = np.abs(center1 - center2)

    if cent_diff < 18:
        # Segmentation not necessary.
        if center1 > 120 and center2 > 120:
            # All cloud image
            th_img = np.zeros(np.shape(mask))
        else:
            # All sky image
            th_img = np.ones(np.shape(mask))
    else:
        th_img = img < center1 + (center2 - center1) / 2
    th_img[mask == 0] = 0


    return th_img

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

def main():
    try:
        global input_path
        print("Start goodfeauter to track:")

        print("input path: {}".format(input_path))

        #img = cv2.imread(join(input_path, '2015-10-29-12-58-01-wahrsis3-high.jpg'), 1)  # 1=color, 0=grayscale, -1= color including alpha channel

        print('joined path: {}'.format(join(input_path, 'raw_img4.jpg')))

        frame1 = cv2.imread(join(input_path, 'raw_img4.jpg'), 1)
        frame2 = cv2.imread(join(input_path, 'raw_img9.jpg'), 1)

        # good features on threshold image 1
        fig1 = plt.figure(1)
        plt.title('original image 1')
        #plt.imshow(frame1,cmap ='gray')
        plt.imshow(frame1)

        # good features on threshold image 2
        fig2 = plt.figure(2)
        plt.title('original image 2')
        plt.imshow(frame2,cmap ='gray')


        #image_mask = cmask([1724, 2592], 1470, img) # orginal
        image_mask = cmask([972, 1296], 930, frame1)  # ([972, 1296], 735, img)
        image_mask_resized = cv2.resize(image_mask, None, fx=0.25, fy=0.25)  #resize image to 1/4th of original size


        # Extract the color channels frame1
        frame1_resized = cv2.resize(frame1, None, fx=0.25, fy=0.25)
        BR_frame1 = getBRchannel(frame1_resized, image_mask_resized)
        BR_frame1[np.isnan(BR_frame1)] = 0

        # Extract the color channels frame2
        frame2_resized = cv2.resize(frame2, None, fx=0.25, fy=0.25)
        BR_frame2 = getBRchannel(frame2_resized, image_mask_resized)
        BR_frame2[np.isnan(BR_frame2)] = 0

        # segment frame1
        (th_img1, coverage1, center1_1, center1_2) = make_cluster_mask(BR_frame1, image_mask_resized)
        th_img1 = segment(th_img1, coverage1, center1_1, center1_2,BR_frame1,image_mask_resized)

        # segment frame2
        (th_img2, coverage1, center2_1, center2_2) = make_cluster_mask(BR_frame1, image_mask_resized)
        th_img2 = segment(th_img2, coverage1, center2_1, center2_2,BR_frame1,image_mask_resized)


        th_img1_uint8 = np.array(th_img1 * 255, dtype=np.uint8)
        gray1 = cv2.cvtColor(th_img1_uint8, cv2.COLOR_GRAY2RGB)
        img_grey_1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)

        th_img2_uint8 = np.array(th_img2 * 255, dtype=np.uint8)
        gray2 = cv2.cvtColor(th_img2_uint8, cv2.COLOR_GRAY2RGB)
        img_grey_2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(img_grey_1, 25, 0.01, 10)
        #corners = np.int0(corners)


        gray_to_rgb_1 = cv2.cvtColor(th_img1_uint8, cv2.COLOR_GRAY2RGB)
        gray_to_rgb_2 = cv2.cvtColor(th_img2_uint8, cv2.COLOR_GRAY2RGB)

        corners = np.int0(p0)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(gray_to_rgb_1, (x, y), 5, 255, -1)
            cv2.circle(gray_to_rgb_2, (x, y), 5, 255, -1)

        # good features on threshold image 1
        fig3 = plt.figure(3)
        plt.title('good features on threshold image 1')
        plt.imshow(gray_to_rgb_1,cmap ='gray')

        # good features on threshold image 2
        fig4 = plt.figure(4)
        plt.title('good features on threshold image 2')
        plt.imshow(gray_to_rgb_2,cmap ='gray')

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        print('img_grey_1.shape: {}'.format(img_grey_1.shape))
        print('img_grey_1.dtype: {}'.format(img_grey_1.dtype))
        print('img_grey_1.size:  {}'.format(img_grey_1.size))
        print('\n')

        # calculate optical flow
        try:
            p0 = corners
            p1, st, err = cv2.calcOpticalFlowPyrLK(img_grey_1, img_grey_2, p0, None, **lk_params)

        except Exception as e:
            print('calcOpticalFlowPyrLK(): ' + str(e))

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        flow_img = cv2.add(frame, mask)

        # good features on threshold image 2
        fig5 = plt.figure(5)
        plt.title('optical flow')
        plt.imshow(flow_img, cmap='gray')

        plt.show()



    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()