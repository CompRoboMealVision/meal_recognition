#!/usr/bin/env python

""" This script detects plates in an image and isolates 
    just the plate and the food on it. """


import cv2
from matplotlib import pyplot as plt
import numpy as np
from SliderWindow import SliderWindow
from ContourSplitter import splitContours


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
     # Split the contours up, in order to break erroneous connections
    split_contours = splitContours(contours)

    # Sort the contours according to their length
    sorted_data = sorted(split_contours, key=lambda curve: cv2.arcLength(curve, closed=False), reverse=True)

    # Create an image with only the longest contours
    contour_image = np.zeros(edges.shape)
    # NOTE: Still unsure why drawing all contors simultaneously results in bad lines.
    big_contours = sorted_data[0:num_contours]
    for i in range(num_contours):
        cv2.drawContours(contour_image, big_contours[i], -1, (255, 0, 0), 1)

    # Draw windows around random points
    window_xs, window_ys = generateWindowCoords(contour_image, num_windows)
    image_with_windows = drawWindows(image, window_xs, window_ys)

    gx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    gy = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)
    final_image = image_with_windows#final_image = drawNormals(image_with_windows, window_xs, window_ys, gx, gy)
    thetas = calcTangents(window_xs, window_ys, gx, gy)

    x1 = window_xs[0]
    y1 = window_ys[0]
    angle = thetas[0]

    left = -100
    x_left = np.cos(angle)*left + x1
    y_left = np.sin(angle)*left + y1
    right = 1000
    x_right = np.cos(angle)*right + x1
    y_right = np.sin(angle)*right + y1
    # import pdb; pdb.set_trace()

    m = (y_left - y_right)/(x_left - x_right)
    cv2.circle(final_image, (x1, y1), 3, (0, 255, 255), 3)

    cv2.line(final_image, (int(x_left), int(y_left)), (int(x_right), int(y_right)), color=(255, 0, 0), thickness=2)

    
    for (x2, y2) in zip(window_xs, window_ys):
        if inRegion(m, (x_left, y_left), (x2, y2)):
            # Draw a circle at that point
            # We pass in (y2, x2) because that corresponds to (row, col)
            cv2.circle(final_image, (x2, y2), 1, (0, 255, 0), 2)


    print thetas
    return final_image


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

def drawWindows(image, window_xs, window_ys, width=20):
    """ Draw squares at the specified coordinates on an image. """
    im = np.copy(image)
    color = (255, 255, 255)
    for (x,y) in zip(window_xs, window_ys):
        top_left = (x - width/2, y - width/2)
        bot_right = (x + width/2, y + width/2)
        cv2.rectangle(im, top_left, bot_right, color, 1)

    return im

def calcTangents(window_xs, window_ys, gx, gy):
    """ Calcs slope of tangent line at window coordinates. """
    grad_angle = np.arctan2(gy, gx)
    thetas = []
    for (col, row) in zip(window_xs, window_ys):
        theta = grad_angle[row, col] + np.pi/2
        thetas.append(theta)
    return thetas

def drawNormals(image, window_xs, window_ys, gx, gy):
    """ Draws normals at window coordinates. """
    im = np.copy(image)
    grad_angle = np.arctan2(gy, gx)
    grad_mag = np.sqrt(gx**2 + gy**2)
    for (col, row) in zip(window_xs, window_ys):
        theta = grad_angle[row, col]
        mag = grad_mag[row, col] * 0.005
        x2 = np.cos(theta)*mag + col
        y2 = np.sin(theta)*mag + row
        cv2.line(im, (col, row), (int(x2),int(y2)), color=(255, 0, 0), thickness=2)
    return im

def inRegion(m, (x1, y1), (x2, y2)):
    """ Detect if a point (x2, y2) is in a region bounded below 
        by the line defined by m, x1, and y1. """
    # Calculate the slope of our bounding line
    b = y1 - m*x1
    return y2 >= m*x2 + b



if __name__ == '__main__':
    slider_window = SliderWindow()

    image1 = cv2.imread('../images/Food_Plate_Captures/001.png', 1)
    image2 = cv2.imread('../images/Food_Plate_Captures/002.png', 1)
    image3 = cv2.imread('../images/Food_Plate_Captures/003.png', 1)
    image4 = cv2.imread('../images/Food_Plate_Captures/004.png', 1)
    image5 = cv2.imread('../images/Food_Plate_Captures/005.png', 1)

    images = [image1, image2, image3, image4, image5]
    # images = [image1]

    num_images = len(images)
    
    fig = plt.figure(figsize=(7, 20))


    for i, image in enumerate(images):  
        # num_image = slider_window.number_contours
        isolated_image = isolatePlate(image, num_contours=5, num_windows=20)
        # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # concated_images = np.concatenate((gray_image, isolated_image), axis=1)
        # cv2.imshow('Image ', isolated_image)
        
        fig.add_subplot(num_images, 2, 2*i+1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.title('Original Image')
        plt.xticks([]), plt.yticks([])

        fig.add_subplot(num_images, 2, 2*i+2)
        plt.imshow(isolated_image, cmap='gray')
        # plt.title('Isolated Image')
        plt.xticks([]), plt.yticks([])

        # cv2.waitKey(1)
    plt.tight_layout()
    plt.show()


