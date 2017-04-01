#!/usr/bin/env python2.7

import sys
import os
import json

"""Interface for getting and setting the path for the model."""

# Location of the file that contains the path to retrained_graph.pb
path_loc = 'model_path.json'

def initialize():
    """Query the user for the path to the model's folder"""

    if is_initialized():
        'Reading path from model_path.json ...'
    else:
        user_input = raw_input("Enter the path to the folder that contains retrained_graph.pb: ")

        if not os.path.exists(user_input):
            print "The following is not a valid path: " + str(user_input)
            # Retry
            initialize()
        elif os.path.isfile(user_input):
            print "The following is not a valid directory path: " + str(user_input)
            print "Please point to a directory, not a file."
            # Retry
            initialize()
        elif os.path.isdir(user_input):
            path = user_input
            # Check for the two necessary files
            retrained_graph_path = os.path.join(path, 'retrained_graph.pb')
            f = open(retrained_graph_path, 'r+')
            print 'retrained_graph.pb found...'
            f.close()
            retrained_labels_path = os.path.join(path, 'retrained_labels.txt')
            f = open(retrained_labels_path, 'r+')
            print 'retrained_labels.txt found...'
            f.close()

            with open(path_loc, 'w') as outfile:
                print 'Saving path to ' + path_loc
                info = {'model_path': path}
                json.dump(info, outfile)

def is_initialized():
    """Checks if we have initialized the local model_path file."""
    return os.path.isfile(path_loc)

def retrained_graph():
    with open(path_loc, 'r+') as f:
        data = json.load(f)
    return os.path.join(data['model_path'], 'retrained_graph.pb')

def retrained_labels():
    with open(path_loc, 'r+') as f:
        data = json.load(f)
    return os.path.join(data['model_path'], 'retrained_labels.txt')


