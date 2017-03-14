#!/usr/bin/env python

import numpy as np

def splitContours(contours):
    """Splits a list of contours up, in order to break erroneous connections"""
    split_contours = []
    for contour in contours:
        c = contour.reshape(-1, 2)
        line_segments = splitLine(c)
        for seg in line_segments:
            # Turn it back to its original shape, so we can add it back to contours
            new_contour = seg.reshape(-1,1,2)
            # Dont add empty contours to our new list of contours
            if new_contour.size != 0:
                split_contours.append(new_contour)

    return split_contours

def splitLine(line):
    """Splits a line on horizontal or vertical segments"""
    # Find a point where our line changes direction
    l = line
    change = l[2:] - l[:-2]
    # Create breaks where derivative equals 0
    break_indicies = np.unique(np.where(change == 0)[0])
    line_segments = []
    while break_indicies.size > 0:
        i = break_indicies[0]

        # Add the beginning of the line to our list
        new_line= l[0:i+1]
        line_segments.append(new_line)

        # The rest of the line becomes our new line
        l = l[i+1:]

        # Recalculate the change vector
        change = l[2:] - l[:-2]
        break_indicies = np.unique(np.where(change == 0)[0])

    line_segments.append(l)
    return line_segments

def testSplitLine_one_split():
    """Unit test for splitLine"""
    line = np.array([1, 2, 3, 3, 3, 4, 5])
    split_lines = splitLine(line)
    if np.all(np.concatenate(split_lines) == line):
        print(str(np.concatenate(split_lines)), " == ", str(line))
    else:
        print(str(np.concatenate(split_lines)), " != ", str(line))

def testSplitLine_two_splits():
    """Unit test for splitLine"""
    line = np.array([1, 2, 3, 3, 3, 4, 4, 4, 5])
    split_lines = splitLine(line)
    if np.all(np.concatenate(split_lines) == line):
        print(str(np.concatenate(split_lines)), " == ", str(line))
    else:
        print(str(np.concatenate(split_lines)), " != ", str(line))



if __name__ == '__main__':
    testSplitLine_one_split()
    testSplitLine_two_splits()