#!/usr/bin/env python

""" This labels a food using a trained neural net."""

from PlateIsolator import isolatePlate
import sys
sys.path.insert(0, '../../dcnns/testing')
from label_image import label_image, setup
from label_image_hist import plot_bar_graph
import cv2
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    image_path = sys.argv[1]
    human_strings = setup()
    original_image = cv2.imread(image_path, 1)
    isolated_image, _ = isolatePlate(original_image, expansionFactor = 2)
    file_name, file_extension = os.path.splitext(image_path)
    temp_file_name = file_name + '_temp' + file_extension

    cv2.imwrite(temp_file_name, isolated_image)

    labels, scores = label_image(temp_file_name, human_strings)
    plot_bar_graph(labels, scores, 5)

    os.remove(temp_file_name)
    plt.imshow(cv2.cvtColor(isolated_image, cv2.COLOR_BGR2RGB))
    plt.show()
