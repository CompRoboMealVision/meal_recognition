#!/usr/bin/env python

""" This script detects plates in an image and isolates 
    just the plate and the food on it. """


import cv2
from matplotlib import pyplot as plt
import numpy as np
from SliderWindow import SliderWindow

def isolatePlate(image, canny_thresh1=100, canny_thresh2=200, num_contours=10, num_windows=20):
    """ Isolate a food plate from an image with extra data.
        Approach taken from Hsin-Chen Chen et al 2015 Meas. Sci. Technol. 26 025702
        http://iopscience.iop.org/article/10.1088/0957-0233/26/2/025702/pdf. """
    # Convert to greyscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get edges using canny edge detection. Params not tuned.
    edges = cv2.Canny(image_gray, canny_thresh1, canny_thresh2)
    kernel = np.ones((3,3),np.uint8)

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours according to their length
    sorted_data = sorted(contours, key=lambda curve: cv2.arcLength(curve, closed=False), reverse=True)
    
    # Create an image with only the longest contours
    contour_image = np.zeros(edges.shape)
    cv2.drawContours(contour_image, sorted_data[0:num_contours], -1, (255, 0, 0), 1)
    # import pdb
    # pdb.set_trace()
    window_xs, window_ys = generateWindowCoords(contour_image, num_windows)
    image_with_windows = drawWindows(np.copy(contour_image), 20, window_xs, window_ys)

    return image_with_windows

def generateWindowCoords(edges, num_windows):
    """ Generates random coordinates for the windows which
        will be used for ellipse detection. """
    # Converts the edge image to coordinate of points
    # Returns two ndarrays
    row_coords, col_coords = np.nonzero(edges)
    number_of_points = col_coords.size
    # Generate random numbers from 0 to number_of_points
    box_indices = np.random.choice(number_of_points, num_windows)
    selected_xs = col_coords[box_indices]
    selected_ys = row_coords[box_indices]
    
    return selected_xs, selected_ys

def drawWindows(image, width, window_xs, window_ys):
    """ Draw squares at the specified coordinates on an image. """
    color = (255, 255, 255)
    for (x,y) in zip(window_xs, window_ys):
        top_left = (x - width/2, y - width/2)
        bot_right = (x + width/2, y + width/2)
        cv2.rectangle(image, top_left, bot_right, color, 1)

    return image

if __name__ == '__main__':
    # slider_window = SliderWindow()

    image1 = cv2.imread('../images/Food_Plate_Captures/001.png', 1)
    image2 = cv2.imread('../images/Food_Plate_Captures/002.png', 1)
    image3 = cv2.imread('../images/Food_Plate_Captures/003.png', 1)
    image4 = cv2.imread('../images/Food_Plate_Captures/004.png', 1)
    image5 = cv2.imread('../images/Food_Plate_Captures/005.png', 1)

    images = [image1, image2, image3, image4, image5]

    num_images = len(images)

    for i, image in enumerate(images):
        
        isolated_image = isolatePlate(image)
        
        plt.subplot(num_images, 2, 2*i+1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])

        plt.subplot(num_images, 2, 2*i+2)
        plt.imshow(isolated_image, cmap='gray')
        plt.title('Isolated Image')
        plt.xticks([]), plt.yticks([])

    plt.show()


