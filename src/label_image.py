#!/usr/bin/env python2.7

# Derived from https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#4

import tensorflow as tf, sys
import ModelPath

def setup():
    # Loads label file, strips off carriage return
    ModelPath.initialize()
    label_lines = [line.rstrip() for line
                       in tf.gfile.GFile(ModelPath.retrained_labels())]
    # Unpersists graph from file
    with tf.gfile.FastGFile(ModelPath.retrained_graph()) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    return label_lines

def label_image(image_path, labels):

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        guesses = {}
        human_strings = [labels[node_id] for node_id in top_k]
        scores = [predictions[0][node_id] for node_id in top_k]

    return human_strings, scores


if __name__ == '__main__':
    # change this as you see fit
    image_path = sys.argv[1]
    labels = setup()
    human_strings, scores = label_image(image_path, labels)
    for human_string, score in zip(human_strings, scores):
        print('%s (score = %.5f)' % (human_string, score))
