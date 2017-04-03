# http://stackoverflow.com/questions/42463896/prediction-using-inceptionv3-in-tensorflow

import argparse
import cv2
import numpy as np
import tensorflow as tf

from label_image_hist import plot_bar_graph

import cv2
import os
import matplotlib.pyplot as plt

def create_test_batch(input_image):
    data = []
    img = cv2.imread(input_image) # Read the test image
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # Convert RGB image to YUV
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_resize = cv2.resize(img_output,(299,299)) # Resize the image acceptable by InceptionV3 model
    for i in range(0, 32):
        data.append(img_resize) # Create a batch of 64 images
    print np.shape(data)
    return data

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

def setup():
    pass

def label_image():
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="", type=str, help="Frozen model file to import")
    parser.add_argument("--label_path", default="", type=str, help="")
    parser.add_argument("--image_name",type=str,help="Image to test")
    args = parser.parse_args()

    # Create test batch
    image_data = create_test_batch(args.image_name)
    # We use our "load_graph" function
    graph = load_graph(args.model_path)

    label_lines = [line.rstrip() for line in tf.gfile.GFile(args.label_path)]

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/batch_processing/Reshape:0') # Input tensor
    y = graph.get_tensor_by_name('prefix/tower_0/logits/predictions:0') # Output tensor

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        y_out = sess.run(y, feed_dict={ x: image_data }) # < 45
        predictions = np.mean(y_out, axis = 0)

        print label_lines
        print predictions[1:-1]
        labeled_predictions = zip(label_lines, predictions[1:-1])
        labeled_predictions = sorted(labeled_predictions, key=lambda tup: tup[1], reverse = True)
        # print labeled_predictions

        return [label[0] for label in labeled_predictions], [label[1] for label in labeled_predictions]

if __name__ == "__main__":
    labels, scores = label_image()
    plot_bar_graph(labels, scores, 90)
