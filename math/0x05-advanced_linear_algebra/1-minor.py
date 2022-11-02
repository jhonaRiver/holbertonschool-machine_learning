#!/usr/bin/env python3
"""Module minor."""


def minor(matrix):
    """
    Calculate the minor matrix of a matrix.

    Args:
        matrix (list): list of lists whose minor matrix should be calculated
    Returns:
        minor matrix
    """
    mat_l = len(matrix)
    range_mat_l = range(mat_l)
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(mat) == list for mat in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all([len(mat) == mat_l for mat in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    if mat_l == 1:
        return [[1]]
    minor_values = []
    for row in range_mat_l:
        minor_r = []
        for col in range_mat_l:
            minor_c = minor_val(matrix, row, col)
            minor_r.append(minor_c)
        minor_values.append(minor_r)
    return minor_values


def minor_val(matrix, idx_r, idx_c):
    """
    Compute minor in each index of the matrix.

    Args:
        matrix (list): list of lists to calculate minor
        idx_r (int): row
        idx_c (int): column
    Returns:
        minor of matrix
    """
    minor_mat = [rows[:idx_c] + rows[idx_c + 1:] for rows in (matrix[:idx_r] +
                                                              matrix[idx_r +
                                                                     1:])]
    return determinant(minor_mat)


def multi_determinant(matrix):
    """
    Compute the determinant of a given matrix.

    Args:
        matrix: list of lists whose determinant should be calculated
    Returns:
        determinant of matrix
    """
    mat_l = len(matrix)
    if mat_l == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    deter = 0
    cols = list(range(len(matrix)))
    for c in cols:
        mat_cp = [r[:] for r in matrix]
        mat_cp = mat_cp[1:]
        rows = range(len(mat_cp))
        for r in rows:
            mat_cp[r] = mat_cp[r][0:c] + mat_cp[r][c + 1:]
        sign = (-1) ** (c % 2)
        sub_det = multi_determinant(mat_cp)
        deter += sign * matrix[0][c] * sub_det
    return deter


def determinant(matrix):
    """
    Calculate the determinant of a matrix.

    Args:
        matrix: list of lists whose determinant should be calculated
    Returns:
        determinant of matrix
    """
    mat_l = len(matrix)
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(mat) == list for mat in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix[0] and mat_l != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if matrix == [[]]:
        return 1
    if mat_l == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if not all(mat_l == len(col) for col in matrix):
        raise ValueError("matrix must be a square matrix")
    return multi_determinant(matrix)
