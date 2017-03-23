#!/usr/bin/env python

""" This script detects plates in an image and isolates 
    just the plate and the food on it. """

import cv2
from matplotlib import pyplot as plt
import numpy as np
from SliderWindow import SliderWindow
from ContourSplitter import splitContours
from CliqueFinder import findMaximalClique


# @profile
def isolatePlate(image, canny_thresh1=52, canny_thresh2=184, contour_thresh=0.36, num_windows=32, window_dist=0):
    """ Isolate a food plate from an image with extra data.
        Approach taken from Hsin-Chen Chen et al 2015 Meas. Sci. Technol. 26 025702
        http://iopscience.iop.org/article/10.1088/0957-0233/26/2/025702/pdf. """
    # Convert to greyscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Equalize the contrast
    image_equalized = cv2.equalizeHist(image_gray)
    # Get edges using canny edge detection. Params not tuned.
    edges = cv2.Canny(image_equalized, canny_thresh1, canny_thresh2)
    kernel = np.ones((3,3),np.uint8)

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
     # Split the contours up, in order to break erroneous connections
    split_contours = splitContours(contours)

    # Sort the contours according to their length
    sorted_data = sorted(split_contours, key=lambda curve: cv2.arcLength(curve, closed=False), reverse=True)

    # Create an image with only the longest contours
    longest_contour = cv2.arcLength(sorted_data[0], closed=False)
    # Create a list with all contours up to a certain threshold of the longest
    big_contours = []
    for contour in sorted_data:
        if cv2.arcLength(contour, closed=False) >= contour_thresh*longest_contour:
            big_contours.append(contour)

    contour_image = np.zeros(edges.shape)
    # NOTE: Still unsure why drawing all contors simultaneously results in bad lines.
    for contour in big_contours:
        cv2.drawContours(contour_image, contour, -1, (255, 0, 0), 1)

    drew_elipse = False
    # Sometimes drawEllipse doesn't draw an ellipse.
    # We iterate until it does.
    while not drew_elipse:
        # Draw windows around random points
        window_xs, window_ys = generateWindowCoords(contour_image, num_windows, min_dist=window_dist)
        sorted_args = np.argsort(window_xs)
        window_xs = window_xs[sorted_args]
        window_ys = window_ys[sorted_args]
        best_ellipse, drew_elipse, size_maximal_clique = drawEllipse(image, image_equalized, edges, window_xs, window_ys, 10)

    mask = np.zeros(edges.shape)
    cv2.ellipse(mask, best_ellipse, (255, 255, 255), -1)
    final_image = np.copy(image)
    final_image[mask[..., np.newaxis].repeat(3, 2) == 0] = 0
    # cv2.ellipse(contour_image, best_ellipse, (255, 255, 255), 2)

    return final_image, size_maximal_clique

def drawEllipse(image, image_equalized, edges, window_xs, window_ys, min_clique_size):
    """ Draws the best ellipse through the given windows. 
        Returns: image_with_ellipse, an image with an ellipse drawn on it
                 drew_elipse, indicates whether an ellipse was drawn successfully
                 size_maximal_clique, the size of the biggest clique."""
    
    
    image_with_windows = drawWindows(image, window_xs, window_ys)

    gx = cv2.Sobel(image_equalized, cv2.CV_64F,1,0,ksize=5)
    gy = cv2.Sobel(image_equalized, cv2.CV_64F,0,1,ksize=5)
    final_image = drawNormals(image_with_windows, window_xs, window_ys, gx, gy)

    # C is a conjunction matrix where
    # A_i = the region bounded by the tangent at point i
    # C[i, j] == True if the point j is in A_i
    C = np.zeros((window_xs.size, window_xs.size), dtype=bool)

    for i, (x1, y1) in enumerate(zip(window_xs, window_ys)):

        normal_x = gx[y1, x1]
        normal_y = gy[y1, x1]
        normal_vec = np.array([normal_x, normal_y])
        origin_vec = np.array([x1, y1])

        for j, (x2, y2) in enumerate(zip(window_xs, window_ys)):
            point_vec = np.array([x2, y2])
            if inDirection(normal_vec, origin_vec, point_vec):
                C[i, j] = True

    # connections is like C except:
    # connections[i, j] == True if the point j is in A_i 
    #                              and point i is in A_j
    connections = C * C.T

    # We want to group our points together, to find the ones
    # that fall on the same ellipse.
    # This is the same problem as finding the biggest clique
    # in an undirected graph.
    groups = findMaximalClique(connections)

    size_maximal_clique = len(groups[0])

    drew_elipse = False
    best_ellipse = 0
    if size_maximal_clique >= min_clique_size:
        best_fit = 0
        
        for group in groups:

        # Plot the points that are our best guesses for the ellipse
        # for i in group:
        #     cv2.circle(final_image, (window_xs[i], window_ys[i]), 3, (255, 255, 0), 3)
            group_xs = window_xs[group]
            group_ys = window_ys[group]
            # cv2.fitEllipse wants a 2xn numpy array
            points = np.vstack((group_xs, group_ys)).T
            
            num_points = points.shape[0]
            # We need a minimum of 5 points to draw an ellipse
            if num_points > 5:
                ellipse_points = cv2.fitEllipse(points)
                # Create an empty image with just the ellipse
                ellipse_image = np.zeros(edges.shape)
                cv2.ellipse(ellipse_image, ellipse_points, (255, 255, 255), 3)
                # We measure error by looking how well ellipses line up with the edges
                fitting_constant = np.sum(np.logical_and(ellipse_image, edges))
                # print fitting_constant
                if fitting_constant > best_fit:
                    best_fit = fitting_constant
                    best_ellipse = ellipse_points

    if best_ellipse:
        drew_elipse = True

    return best_ellipse, drew_elipse, size_maximal_clique

def generateWindowCoords(edges, num_windows, min_dist=0):
    """ Generates random coordinates for the windows which
        will be used for ellipse detection. """
    # Converts the edge image to coordinate of points
    # Returns two ndarrays
    row_coords, col_coords = np.nonzero(edges)
    number_of_points = col_coords.size
    # Generate random numbers from 0 to number_of_points

    selected_xs = np.array([], dtype=int)
    selected_ys = np.array([], dtype=int)

    while selected_xs.size < num_windows:
        box_indices = np.random.choice(number_of_points, 1)
        x = col_coords[box_indices]
        y = row_coords[box_indices]
        dists = np.sqrt((selected_xs - x)**2 + (selected_ys-y)**2)
        # Check that its a minimum distance from other points
        if not np.any(dists < min_dist):
            selected_xs = np.append(selected_xs, x)
            selected_ys = np.append(selected_ys, y)
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

def inDirection(normal, origin, point):
    """ Tells if point is in the direction of the normal to origin.
        nomral: Normal vector at origin
        origin: a 2x1 position vector 
        point: a 2x1 position vector. """
    diff = point - origin
    double_dot = np.dot(diff, normal)
    return double_dot > 0

def refineParams():
    """ Run main function while allowing user to manipulate parameters. """
    slider_window = SliderWindow()

    image1 = cv2.imread('../images/Food_Plate_Captures/001.png', 1)
    image2 = cv2.imread('../images/Food_Plate_Captures/002.png', 1)
    image3 = cv2.imread('../images/Food_Plate_Captures/003.png', 1)
    image4 = cv2.imread('../images/Food_Plate_Captures/004.png', 1)
    image5 = cv2.cvtColor(cv2.imread('../images/Food_Plate_Captures/006.jpeg', 1), cv2.COLOR_BGR2RGB)
    image5 = cv2.resize(image5, (400, 400)) 
    images = [image1, image2, image3, image4, image5]

    last_num_windows = 0
    last_canny_thresh1 = 0
    last_canny_thresh2 = 0

    while (True):
        num_windows = slider_window.num_windows
        canny_thresh1 = slider_window.canny_thresh1
        canny_thresh2 = slider_window.canny_thresh2
        points_dist = slider_window.points_dist
        contour_thresh = slider_window.contour_thresh
        param_has_changed = (num_windows != last_num_windows
                        or canny_thresh1 != last_canny_thresh1
                        or canny_thresh2 != last_canny_thresh2
                        or points_dist != last_points_dist
                        or contour_thresh != last_contour_thresh)
        try:
            for i, image in enumerate(images):
                isolated_image, size_maximal_clique = isolatePlate(image, contour_thresh=contour_thresh, num_windows = num_windows,
                                         canny_thresh1=canny_thresh1, canny_thresh2=canny_thresh2,
                                         window_dist=points_dist)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(isolated_image,str(size_maximal_clique),(10,40), font, 1,(0,255,255),2)
                cv2.imshow('Image ' + str(i), isolated_image)
        except Exception as e:
            print e
            
        
        cv2.waitKey(1)
        last_num_windows = num_windows
        last_canny_thresh1 = canny_thresh1
        last_canny_thresh2 = canny_thresh2
        last_points_dist = points_dist
        last_contour_thresh = contour_thresh

def run():
    # slider_window = SliderWindow()

    image1 = cv2.imread('../images/Food_Plate_Captures/001.png', 1)
    image2 = cv2.imread('../images/Food_Plate_Captures/002.png', 1)
    image3 = cv2.imread('../images/Food_Plate_Captures/003.png', 1)
    image4 = cv2.imread('../images/Food_Plate_Captures/004.png', 1)
    image5 = cv2.cvtColor(cv2.imread('../images/Food_Plate_Captures/006.jpeg', 1), cv2.COLOR_BGR2RGB)
    image5 = cv2.resize(image5, (400, 400)) 

    images = [image1, image2, image3, image4, image5]

    num_images = len(images)
    
    fig = plt.figure(figsize=(7, 20))

    for i, image in enumerate(images):  
        # num_image = slider_window.number_contours
        isolated_image, size_maximal_clique = isolatePlate(image)

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


if __name__ == '__main__':
    refineParams()
    # run()

