import copy
import random

import numpy as np
from scipy.spatial.distance import correlation
from skimage.feature import corner_harris, corner_peaks, corner_subpix


class Sequence:
    def __init__(self):
        self.path = np.array([])
        self.corners = np.array([])
        self.junctions = np.array([])
        self.stumps = np.array([])
        self.ends = np.array([])


def hog_gradient(image):
    """Compute unnormalized gradient image along `row` and `col` axes.
    Parameters
    ----------
    image : (M, N) ndarray
        Grayscale image or one of image channel.
    Returns
    -------
    g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
    """
    g_row = np.zeros(image.shape, dtype=np.double)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = image[2:, :] - image[:-2, :]
    g_col = np.zeros(image.shape, dtype=np.double)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]

    magnitude = np.hypot(g_col, g_row)

    orientation = np.rad2deg(np.arctan2(g_row, g_col)) % 180

    orientations = np.array([0., 45., 90., 135.])
    orientation_histogram = np.zeros(orientations.shape)
    for orie in orientations:
        h = np.sum(magnitude[orientation == orie])
        orientation_histogram[orientations == orie] = h
    orientation_histogram = orientation_histogram / np.sum(orientation_histogram)
    return orientation_histogram


def compare_histogram(a, b):
    return correlation(a, b)


def find_first_stump(skeleton):
    coordinates = list()
    direction = list()
    m, n = skeleton.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if skeleton[i, j]:
                foot = skeleton[i - 1:i + 2, j - 1:j + 2]
                foot = np.array(foot * 1)
                if np.sum(foot) == 2:
                    coordinates.append(np.array([i, j]))
                    foot[1, 1] = 0
                    p = np.array(np.where(foot)).flatten()
                    direction.append([i + p[0] - 1, j + p[1] - 1])
                    return np.array(coordinates), np.array(direction)
    return np.array([]), np.array([])


def find_corners(skeleton):
    repons_image = corner_harris(skeleton)
    coords = corner_peaks(repons_image, min_distance=4)
    coords_subpix = corner_subpix(skeleton, coords, window_size=13)
    return coords, coords_subpix, repons_image


def follow_line(skeleton, **kwargs):
    skeleton_local = copy.deepcopy(skeleton)
    window_size = kwargs.get('window_size', 3)
    corner_threshold = kwargs.get('corner_threshold', 0.1)
    rs = list()
    before = np.sum(skeleton_local * 1)
    after = 0
    loop = True
    while np.sum(skeleton_local * 1) != 0:
        l_before = np.sum(skeleton_local * 1)
        s = Sequence()
        stumps, directions = find_first_stump(skeleton_local)
        if stumps.size == 0:

            if before > 1 and not l_before == after and loop:
                loop = False
                cells = np.where(skeleton_local)
                r_id = random.randint(0, len(cells[0]))
                skeleton_local[cells[0][r_id], cells[1][r_id]] = False
                stumps, directions = find_first_stump(skeleton_local)
            else:
                after = np.sum(skeleton_local * 1)
                break
        start = stumps.flatten()  # [0, :]
        direction = directions.flatten()  # [0, :]

        path = list()
        starts = list()
        corners = list()
        junctions = list()
        ends = list()

        path.append(start)
        path.append(direction)
        last = start
        current = np.array(direction)
        old_histogram = np.array([])

        while True:
            skeleton_local[last[0], last[1]] = False
            foot = skeleton_local[current[0] - 1:current[0] + 2, current[1] - 1:current[1] + 2]
            # start index:end index-1
            window = skeleton[current[0] - window_size:current[0] + window_size + 1,
                     current[1] - window_size:current[1] + window_size + 1]  # start index:end index-1
            p = np.where(foot)
            current_histogram = hog_gradient(1 * window)
            if old_histogram.size != 0:
                sim = compare_histogram(old_histogram, current_histogram)
                if sim > corner_threshold:  # magic number denotes similarity between histograms...
                    corners.append(current)
            old_histogram = current_histogram
            col1 = np.array(current[0] + p[0] - 1)
            col2 = np.array(current[1] + p[1] - 1)
            future = np.concatenate((col1, col2)).reshape((2, -1)).transpose()
            test = np.vstack((future, current))
            lid = np.ravel_multi_index(test.T, test.max(0) + 1)
            _, idx, count = np.unique(lid, return_index=True, return_counts=True)
            future = test[idx[count == 1]].flatten().reshape(-1, 2)
            if future.size > 2:
                starts.extend(future[1:, :])
                future = future[0, :]
                junctions.append(current)
            last = current
            current = future.flatten()
            if future.size == 0:
                ends = np.array(last).reshape((1, 2))
                break
            path.append(future.flatten())
        s.path = np.array(path)
        s.corners = np.array(corners)
        s.junctions = np.array(junctions)
        s.stumps = stumps
        s.ends = ends
        rs.append(s)
        after = np.sum(skeleton_local * 1)

    if before > after:
        flag = 'regular'
    elif before == after and before == 1:
        flag = 'island'
        s = Sequence()
        s.path = np.where(skeleton_local)
        s.corners = np.where(skeleton_local)
        rs.append(s)

    return rs, flag
