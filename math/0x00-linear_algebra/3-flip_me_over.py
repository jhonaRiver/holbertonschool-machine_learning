#!/usr/bin/env python3
'''
module for matrix_transpose
'''


def matrix_transpose(matrix):
    '''
    returns the transpose of a 2D matrix
    '''
    return [list(row) for row in zip(*matrix)]
