#!/usr/bin/env python

""" This script detects plates in an image and isolates 
    just the plate and the food on it. """


import cv2
from matplotlib import pyplot as plt
import numpy as np

def isolatePlate(image, num_windows=20):
    # Convert to greyscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get edges using canny edge detection. Params not tuned.
    edges = cv2.Canny(image_gray,100, 200)
    
    window_xs, window_ys = generateWindowCoords(edges, num_windows)
    image_with_windows = drawWindows(np.copy(image), 20, window_xs, window_ys)

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
    image = cv2.imread('../images/Food_Plate_Captures/001.png', 1)
    isolated_image = isolatePlate(image)

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])



    plt.subplot(122)
    plt.imshow(isolated_image, cmap='gray')
    plt.title('Isolated Image')
    plt.xticks([]), plt.yticks([])

    plt.show()


