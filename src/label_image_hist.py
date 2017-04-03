#!/usr/bin/env python2.7

from label_image import label_image, setup
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_bar_graph(labels, scores, bins):

    fig, ax = plt.subplots()
    ind = np.arange(bins)
    width = 0.35
    labels = labels[0:bins]
    heights = scores[0:bins]
    rects = ax.bar(ind, heights, width)
    ax.set_xticks(ind)
    ax.set_xticklabels((labels))
    ax.set_ylabel('Probablity')
    ax.set_title('Top 5 food categories probablities')
    plt.show()

def plot_bar_graph_withfood(image_path, labels, scores, bins):
    ax1 = plt.subplot(211)
    img = mpimg.imread(image_path)
    fig = plt.imshow(img)

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    ax = plt.subplot(212)
    ind = np.arange(bins)
    width = 0.35
    labels = labels[0:bins]
    heights = scores[0:bins]
    rects = ax.bar(ind, heights, width)
    ax.set_xticks(ind)
    ax.set_xticklabels((labels))
    ax.set_ylabel('Probablity')
    plt.show()

if __name__ == '__main__':
    # change this as you see fit
    image_path = sys.argv[1]
    human_strings = setup()
    labels, scores = label_image(image_path, human_strings)
    plot_bar_graph_withfood(image_path, labels, scores, 5)
