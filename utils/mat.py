import numpy as np


def get_top_id(mat_2d, top_n, col_id=None, row_id=None):

    if col_id is not None:
        top_idx = np.argpartition(mat_2d[:, col_id], -1 * top_n)[-1 * top_n:][::-1].tolist()

        return top_idx

    elif row_id is not None:
        top_idx = np.argpartition(mat_2d[row_id, :], -1 * top_n)[-1 * top_n:][::-1].tolist()

        return top_idx



