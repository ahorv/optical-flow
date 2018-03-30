import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import join
from sklearn.cluster import KMeans

# @ LAB use interpreter: C:\Anaconda3\envs\py34\python.exe

global input_path
global output_path
#input_path  = r'C:\Hoa_Python_Projects\python_scripts\WAHRSIS\input'  # @ home
#output_path = r'C:\Hoa_Python_Projects\python_scripts\optflow\output'  # @ home
output_path = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\output' # @ Lab
input_path  = r'C:\Users\tahorvat\PycharmProjects\opt_flow_python_scripts\optflow\input'  # @ Lab


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

def main():
    try:
        global input_path
        print("Start goodfeauter to track:")

        #img = cv2.imread(join(input_path, '2015-10-29-12-58-01-wahrsis3-high.jpg'), 1)  # 1=color, 0=grayscale, -1= color including alpha channel
        img = cv2.imread(join(input_path, 'raw_img9.jpg'), 1)

        fig1 = plt.figure(1)
        plt.title('original image')
        plt.imshow(img, cmap='gray')

        #print('img.shape: {}'.format(img.shape))
        #print('img.dtype: {}'.format(img.dtype))
        #print('img.size:  {}'.format(img.size))
        #print('\n')

        #image_mask = cmask([1724, 2592], 1470, img) # orginal
        image_mask = cmask([972, 1296], 930, img)  # ([972, 1296], 735, img)
        image_mask_resized = cv2.resize(image_mask, None, fx=0.25, fy=0.25)  #resize image to 1/4th of original size

        fig2 = plt.figure(2)
        plt.title('mask (972/1296) r = 930 ')
        plt.imshow(image_mask_resized,cmap ='gray')


        # Extract the color channels
        img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)
        BR_image = getBRchannel(img_resized, image_mask_resized)
        BR_image[np.isnan(BR_image)] = 0

        fig3 = plt.figure(3)
        plt.title('blue/red ratio image')
        plt.imshow(BR_image)


        (th_img, coverage, center1, center2) = make_cluster_mask(BR_image, image_mask_resized)
        th_img = segment(th_img, coverage, center1, center2,BR_image,image_mask_resized)

        #print('th_img.shape: {}'.format(th_img.shape))
        #print('th_img.dtype: {}'.format(th_img.dtype))
        #print('th_img.size:  {}'.format(th_img.size))
        #print('\n')

        '''
        plt.imshow(th_img,cmap ='gray')
        plt.show()
        return
        '''

        th_img_uint8 = np.array(th_img * 255, dtype=np.uint8)


        print('th_img_uint8.shape: {}'.format(th_img_uint8.shape))
        print('th_img_uint8.dtype: {}'.format(th_img_uint8.dtype))
        print('th_img_uint8.size:  {}'.format(th_img_uint8.size))
        print('\n')

        gray = cv2.cvtColor(th_img_uint8,cv2.COLOR_GRAY2RGB)
        img_grey = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)


        corners = cv2.goodFeaturesToTrack(img_grey, 25, 0.01, 10)
        corners = np.int0(corners)

        gray_to_rgb = cv2.cvtColor(th_img_uint8, cv2.COLOR_GRAY2RGB)

        # good features on threshold image
        for i in corners:
            x, y = i.ravel()
            cv2.circle(gray_to_rgb, (x, y), 5, 255, -1)

        fig4 = plt.figure(4)
        plt.title('good features on threshold image')
        plt.imshow(gray_to_rgb,cmap ='gray')

        #good features to track on (resized) orginal image
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img_resized, (x, y), 3, 255, -1)

        fig5 = plt.figure(5)
        plt.title('good features to track on original image')
        plt.imshow(img_resized)
        plt.show()





    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()