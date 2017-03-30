#!/usr/bin/env python

""" This labels all food in a directory using a trained neural net."""

from PlateIsolator import isolatePlate
import sys
sys.path.insert(0, '../../dcnns/testing')
from label_image import label_image, setup
from label_image_hist import plot_bar_graph
import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    crops_right = 0
    crops_wrong = 0
    full_right = 0
    full_wrong = 0

    folder = sys.argv[1]
    for file_name in os.listdir(folder):
        if '_test' in file_name:
            os.remove(temp_file_name)
        else:
            # Setup the neural net
            human_strings = setup()
            full_name = os.path.join(folder,file_name)
            print full_name
            original_image = cv2.imread(full_name, 1)
            try:
                original_image = cv2.resize(original_image, (400, 400))
            except:
                pass
            isolated_image, _ = isolatePlate(original_image, expansionFactor = 2, overlap_thresh=0.5)

            name, extension = os.path.splitext(full_name)
            temp_file_name = name + '_temp' + extension

            cv2.imwrite(temp_file_name, isolated_image)

            labels, scores = label_image(temp_file_name, human_strings)
            if labels[0] == 'apple pie':
                crops_right = crops_right + 1
            else:
                crops_wrong = crops_wrong + 1

            os.remove(temp_file_name)

            labels, scores = label_image(full_name, human_strings)
            if labels[0] == 'apple pie':
                full_right = full_right + 1
            else:
                full_wrong = full_wrong + 1

            print('Cropped Image: %.5f)' % (crops_right/float(crops_wrong + crops_right)))
            print('Full Image: %.5f)' % (full_right/float(full_wrong + full_right)))

