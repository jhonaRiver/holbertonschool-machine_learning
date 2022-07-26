#!/usr/bin/env python3
"""Module inverse."""


def inverse(matrix):
    """
    Calculate the inverse of a matrix.

    Args:
        matrix (list): list of lists whose inverse should be calculated
    Returns:
        inverse of matrix, or None if matrix is singular
    """
    mat_l = len(matrix)
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(m) == list for m in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(mat_l == len(col) for col in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if mat_l == 1 and len(matrix[0]) == 1:
        return [[1/(matrix[0][0])]]
    if mat_l == 1 and matrix[0][0] == 0:
        return None
    if determinant(matrix) == 0:
        return None
    det = determinant(matrix)
    adjugate_mat = adjugate(matrix)
    inversed = [[mat_minor / det for mat_minor in row] for row in adjugate_mat]
    return inversed


def determinant(matrix):
    """
    Compute the determinant of a given matrix.

    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Determinant of matrix
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
        sub_det = determinant(mat_cp)
        deter += sign * matrix[0][c] * sub_det
    return deter


def adjugate(matrix):
    """
    Compute the adjugate of a matrix.

    Args:
        matrix: list of lists whose adjugate matrix should be calculated
    Returns: the adjugate matrix of matrix
    """
    mat_l = len(matrix)
    range_mat_l = range(len(matrix))
    if mat_l == 1:
        return [[1]]
    if mat_l == 2 and len(matrix[0]) == 2:
        return [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]]
    minor_values = []
    for row in range_mat_l:
        minor_r = []
        for col in range_mat_l:
            minor_c = minor_val(matrix, row, col)
            sign = (-1) ** (row + col)
            minor_r.append(minor_c * sign)
        minor_values.append(minor_r)
    minor_mat_len = range(len(minor_values))
    minor_mat_len2 = range(len(minor_values[0]))
    swap = [[minor_values[c][r] for c in minor_mat_len]
            for r in minor_mat_len2]
    return swap


def minor_val(matrix, idx_r, idx_c):
    """
    Compute minor in each idx position of the given matrix.

    Args:
        matrix: given matrix
        idx_r: row skipped
        idx_c: col skipped
    Returns: determinant of the matrix with row and col skipped
    """
    minor_mat = [rows[:idx_c] + rows[idx_c + 1:]
                 for rows in (matrix[:idx_r] + matrix[idx_r + 1:])]
    return determinant(minor_mat)
