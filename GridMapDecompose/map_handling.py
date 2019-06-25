import copy

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import io
from skimage.filters import threshold_yen
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

from GridMapDecompose import segment_handling as sgh, slice_handling as sh


class GridMapHandling:
    def __init__(self):
        self.grid_map = np.array([])
        self.map_file_name = []
        self.threshold_type = np.array([])
        self.binary_map = np.array([])
        self.processed_map = np.array([])
        self.slices = list([])
        self.segment_type = list([])
        self.labeled_map = np.array([])
        self.segments = list([])
        self.corner_segments = list([])
        self.__distance_map = np.array([])
        self.__remaining_segments = []

    def load_map_flat(self, file_name):
        self.map_file_name = file_name
        self.grid_map = img_as_ubyte(io.imread(self.map_file_name))
        if len(self.grid_map.shape) == 3:
            self.grid_map = self.grid_map[:, :, 1]
            return self.grid_map
        elif len(self.grid_map.shape) == 2:
            return self.grid_map
        return []

    def threshold_map(self, threshold_type):
        self.threshold_type = threshold_type
        if threshold_type == 'regular':
            thresh = threshold_yen(self.grid_map)
            self.binary_map = self.grid_map <= thresh
        elif threshold_type == 'electrolux':
            l_map = self.grid_map
            l_map[l_map >= 179] = 0
            self.binary_map = l_map > 0
        return 1 * self.binary_map

    def __compute_distances_map(self):
        dimensions = self.binary_map.shape
        inverted_map = np.ones(dimensions, dtype = int)
        inverted_map = inverted_map - self.binary_map
        self.__distance_map = ndimage.distance_transform_cdt(inverted_map)
        # return self.__distance_map

    def fill_gaps(self, step):
        self.__compute_distances_map()
        self.processed_map = np.zeros(self.__distance_map.shape, dtype = int)
        self.processed_map[self.__distance_map <= step] = int(1)
        return self.processed_map

    def slice_map(self):
        if self.processed_map.size == 0:
            print("no fill")
            self.processed_map = copy.deepcopy(self.binary_map * 1)
        labeled_array, _ = ndimage.measurements.label(self.processed_map, [[1, 1, 1],
                                                                           [1, 1, 1],
                                                                           [1, 1, 1]]
                                                      )
        self.slices = list()
        for label in range(1, labeled_array.max() + 1):
            m = np.zeros(labeled_array.shape, dtype = int)
            m[labeled_array == label] = 1
            self.slices.append(m)
        return self.slices

    @staticmethod
    def skeletonize_slice(map_slice):
        l_slice = map_slice
        l_slice[map_slice > 0] = 1
        skeleton = skeletonize(map_slice)
        return skeleton

    def get_segments(self, **kwargs):
        self.labeled_map = np.zeros(self.processed_map.shape, dtype = np.int)

        max_sequence = 0
        for local_slice in self.slices:
            skeleton = self.skeletonize_slice(local_slice)
            skeleton_segments, flag = sh.follow_line(skeleton, **kwargs)
            if flag == 'regular':
                all_labels = list([])
                for S in skeleton_segments:
                    s, f = self.follow_sequence(S)
                    self.segment_type.extend(['s'] * len(s))
                    self.segment_type.extend(['f'] * len(f))
                    all_labels.extend(s)
                    all_labels.extend(f)
                if all_labels:
                    local_labels = self.split_slice(all_labels, local_slice)
                    local_labels = local_labels + max_sequence
                    local_labels = np.where(local_labels == max_sequence, 0, local_labels)
                    max_sequence = np.max(local_labels)
                    self.labeled_map = self.labeled_map + local_labels
            if flag == 'island':
                local_labels = np.zeros(local_slice.shape, dtype = np.int)
                lf = np.array(skeleton_segments[0].path)
                max_sequence += 1
                local_labels[lf[0], lf[1]] = max_sequence
                self.labeled_map = self.labeled_map + local_labels
                self.segment_type.extend(['f'])

        return self.labeled_map, self.segment_type

    def evaluate_segments(self, **kwargs):
        feature_size = kwargs.get('feature_size', 20)

        corner_map = np.zeros(self.labeled_map.shape, dtype = np.int)
        for i in range(1, self.labeled_map.max()):
            local_segment = sgh.Segment()
            cluster = np.where(self.labeled_map == i)
            cluster_size = cluster[0].size
            cluster = np.column_stack((cluster[0], cluster[1]))
            if cluster_size < feature_size and self.segment_type[i - 1] is 's':
                self.segment_type[i - 1] = 'f'
            local_segment.add_cells(cluster)
            local_segment.compute_hull()
            local_segment.compute_mbb()

            self.segments.append(local_segment)

            if self.segment_type[i - 1] is 'f':
                corner_map += (self.labeled_map == i)
        labeled_corner_map, _ = ndimage.measurements.label(corner_map, [[1, 1, 1],
                                                                        [1, 1, 1],
                                                                        [1, 1, 1]]
                                                           )
        for i in range(0, labeled_corner_map.max()):
            local_segment = sgh.Segment()
            cluster = np.where(labeled_corner_map == i)

            cluster = np.column_stack((cluster[0], cluster[1]))

            local_segment.add_cells(cluster)
            local_segment.compute_hull()
            local_segment.compute_mbb()

            self.corner_segments.append(local_segment)

        return self.segments

    @staticmethod
    def follow_sequence(sequence):
        corners_iterator = iter(sequence.corners)
        junctions_iterator = iter(sequence.junctions)
        stumps_iterator = iter(sequence.stumps)
        ends_iterator = iter(sequence.ends)

        c_corner = next(corners_iterator, None)
        c_junction = next(junctions_iterator, None)
        c_stump = next(stumps_iterator, None)
        c_end = next(ends_iterator, None)

        segments = list([])
        features = list([])

        segment = False
        feature = True
        c_feature = list([])
        c_segment = list([])

        for point in sequence.path:
            if (np.array(point) == np.array(c_corner)).all():
                if not feature and not segment:
                    feature = True
                    c_feature.append(point)
                    c_corner = next(corners_iterator, None)
                elif not feature and segment:
                    feature = True
                    segment = False
                    segments.append(c_segment)
                    c_segment = []
                    c_feature.append(point)
                    c_corner = next(corners_iterator, None)
                elif feature:
                    c_feature.append(point)
                    c_corner = next(corners_iterator, None)

            elif (np.array(point) == np.array(c_junction)).all():
                if not feature and not segment:
                    feature = True
                    c_feature.append(point)
                    c_junction = next(junctions_iterator, None)
                elif not feature and segment:
                    feature = True
                    segment = False
                    segments.append(c_segment)
                    c_segment = []
                    c_feature.append(point)
                    c_junction = next(junctions_iterator, None)
                elif feature:
                    c_feature.append(point)
                    c_junction = next(junctions_iterator, None)

            elif (np.array(point) == np.array(c_stump)).all():
                if not feature and not segment:
                    feature = True
                    c_feature.append(point)
                    c_stump = next(stumps_iterator, None)
                elif not feature and segment:
                    feature = True
                    segment = False
                    segments.append(c_segment)
                    c_segment = []
                    c_feature.append(point)
                    c_stump = next(stumps_iterator, None)
                elif feature:
                    c_feature.append(point)
                    c_stump = next(stumps_iterator, None)
            elif (np.array(point) == np.array(c_end)).all():
                if not feature and not segment:
                    feature = True
                    c_feature.append(point)
                    c_end = next(ends_iterator, None)
                elif not feature and segment:
                    feature = True
                    segment = False
                    segments.append(c_segment)
                    c_segment = []
                    c_feature.append(point)
                    c_end = next(ends_iterator, None)
                elif feature:
                    c_feature.append(point)
                    c_end = next(ends_iterator, None)
            else:
                if not feature and not segment:
                    segment = True
                    c_segment.append(point)
                elif not segment and feature:
                    segment = True
                    feature = False
                    features.append(c_feature)
                    c_feature = list([])
                    c_segment.append(point)
                elif segment:
                    c_segment.append(point)
        return segments, features

    @staticmethod
    def split_slice(segments, local_slice):
        labels_matrix = np.zeros(local_slice.shape, dtype = np.int)
        it = 0
        seeds = list([])
        labels = list([])
        for segment in segments:
            lf = np.array(segment)
            it += 1
            labels_matrix[lf[:, 0], lf[:, 1]] = it
            local_slice[lf[:, 0], lf[:, 1]] = 0
            seeds.extend(lf)
            elem = len(lf)
            labels.extend(np.ones((elem, 1)) * it)
        pixels = np.array(np.where(local_slice == 1))
        pixels = np.transpose(pixels.reshape([2, -1]))
        seeds = np.array(seeds)
        dist = cdist(seeds, pixels)

        ps = pixels.shape

        for i in range(0, ps[0]):
            local_distances = dist[:, i]
            local_index = np.argmin(local_distances)
            labels_matrix[pixels[i, 0], pixels[i, 1]] = labels[local_index]
        return labels_matrix
