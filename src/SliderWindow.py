#!/usr/bin/env python
import cv2

class SliderWindow(object):
    """Creates a slider that allows selection of Food Recognition Params"""

    
    def __init__(self):
        """ Initialize the Slider Window"""
        self.window_name = 'threshold_image'

        cv2.namedWindow(self.window_name)
        self.number_contours = 0
        self.canny_thresh1 = 0
        self.canny_thresh2 = 0

        cv2.createTrackbar('Which Contour? ', self.window_name, 0, 255, self.set_contour)
        cv2.createTrackbar('Canny Threshold 1', self.window_name, 0, 255, self.set_thresh1)
        cv2.createTrackbar('Canny Threshold 2', self.window_name, 0, 255, self.set_thresh2)
    
    def set_contour(self, val):
        self.number_contours = val
    def set_thresh1(self, val):
        self.canny_thresh1 = val
    def set_thresh2(self, val):
        self.canny_thresh2 = val
    # def set_hue_upper_bound(self, val):
    #     self.hue_upper_bound = val
    # def set_sat_lower_bound(self, val):
    #     self.sat_lower_bound = val
    # def set_sat_upper_bound(self, val):
    #     self.sat_upper_bound = val
    # def set_val_lower_bound(self, val):
    #     self.val_lower_bound = val
    # def set_val_upper_bound(self, val):
    #     self.val_upper_bound = val