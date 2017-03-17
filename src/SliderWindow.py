#!/usr/bin/env python
import cv2

class SliderWindow(object):
    """Creates a slider that allows selection of Food Recognition Params"""

    
    def __init__(self):
        """ Initialize the Slider Window"""
        self.window_name = 'threshold_image'

        cv2.namedWindow(self.window_name)
        self.num_windows = 20
        self.canny_thresh1 = 50
        self.canny_thresh2 = 200

        cv2.createTrackbar('Number of windows', self.window_name, self.num_windows, 255, self.set_num_windows)
        cv2.createTrackbar('Canny Threshold 1', self.window_name, self.canny_thresh1, 255, self.set_thresh1)
        cv2.createTrackbar('Canny Threshold 2', self.window_name, self.canny_thresh2, 255, self.set_thresh2)
    
    def set_num_windows(self, val):
        self.num_windows = val
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