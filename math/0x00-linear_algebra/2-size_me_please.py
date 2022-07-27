#!/usr/bin/env python3
'''
module for matrix_shape
'''


def matrix_shape(matrix):
    '''
    calculates the shape of a matrix
    '''
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
