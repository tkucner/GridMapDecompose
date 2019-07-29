import copy

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import io
from skimage.filters import threshold_yen
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

from GridMapDecompose import segment_handling as sgh, graph_handling as gh

import pickle


def save(graph_map, name):
    pickle_out = open(name + ".pickle", "wb")
    pickle.dump(graph_map, pickle_out)
    pickle_out.close()


def load(name):
    return pickle.load(open(name + ".pickle", "rb"))



class GridMapHandling:
    def __init__(self):
        self.map_file_name = []

        self.grid_map = np.array([])
        self.occupancy_map = np.array([])
        self.threshold_type = np.array([])
        self.binary_map = np.array([])
        self.processed_map = np.array([])
        self.labeled_map = np.array([])

        self.__distance_map = np.array([])

        self.skeleton = []
        self.graph = []

        self.segment_type = list([])
        self.segments = list([])

        # ------------------ #
        # Graph for elements #
        # ------------------ #
        self.adjacency_matrix_segments = []

    def load_map_flat(self, file_name):
        self.map_file_name = file_name
        self.grid_map = img_as_ubyte(io.imread(self.map_file_name))
        self.occupancy_map = (np.full(self.grid_map.shape, np.max(self.grid_map)) - self.grid_map) / np.max(
            self.grid_map)

        if len(self.grid_map.shape) == 3:
            self.grid_map = self.grid_map[:, :, 1]
            self.occupancy_map = self.occupancy_map[:, :, 1]
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

    def build_graphs(self):
        if self.processed_map.size == 0:
            self.processed_map = copy.deepcopy(self.binary_map * 1)
        self.skeleton = skeletonize(self.processed_map)
        self.graph = gh.Graph()
        self.graph.build_graph(self.skeleton)
        self.graph.label_nodes()
        self.graph.split_to_subgraphs()

    def evaluate_segments(self):

        for i in range(1, self.labeled_map.max() + 1):
            local_segment = sgh.Segment()
            cluster = np.where(self.labeled_map == i)
            # cluster_size = cluster[0].size
            cluster = np.column_stack((cluster[0], cluster[1]))

            local_segment.add_cells(cluster)
            local_segment.compute_hull()
            local_segment.compute_mbb()
            local_segment.id = i

            self.segments.append(local_segment)

        return self.segments

    def label_map(self):
        self.labeled_map = np.zeros(self.binary_map.shape, dtype=np.int)
        local_binary_map = self.binary_map.copy()
        labels = list([])
        it = 0
        seeds = list([])

        for sg in self.graph.W:
            self.segment_type.append('w')
            it = it +1
            for node_id in sg.node:
                a = self.graph.nodes[node_id]

                ac = a['coordinates']
                self.labeled_map[ac[0], ac[1]] = it
                local_binary_map[ac[0], ac[1]] = 0
                seeds.append(a['coordinates'])
                labels.append(it)
        for sg in self.graph.C:
            self.segment_type.append('f')
            it = it + 1
            for node_id in sg.node:
                a = self.graph.nodes[node_id]
                ac = a['coordinates']
                self.labeled_map[ac[0], ac[1]] = it
                local_binary_map[ac[0], ac[1]] = 0
                seeds.append(a['coordinates'])
                labels.append(it)
        pixels = np.array(np.where(local_binary_map == 1))
        pixels = np.transpose(pixels.reshape([2, -1]))
        seeds = np.array(seeds)
        dist = cdist(seeds, pixels)
        ps = pixels.shape
        for i in range(0, ps[0]):
            local_distances = dist[:, i]
            local_index = np.argmin(local_distances)
            self.labeled_map[pixels[i, 0], pixels[i, 1]] = labels[local_index]
        self.adjacency_matrix_segments = np.zeros((max(labels) + 1, max(labels) + 1))
        local_walls_coord = np.array(np.where(self.labeled_map > 0)).transpose()

        for coord in local_walls_coord:
            neighbours = np.array(self.check_neighbour(coord, self.labeled_map))
            if not neighbours.size == 0:
                for neighbour in neighbours:
                    self.adjacency_matrix_segments[self.labeled_map[coord[0], coord[1]], neighbour] = 1
                    self.adjacency_matrix_segments[neighbour, self.labeled_map[coord[0], coord[1]]] = 1

    @staticmethod
    def skeletonize(map_slice):
        l_slice = map_slice
        l_slice[map_slice > 0] = 1
        skeleton = skeletonize(map_slice)
        return skeleton

    @staticmethod
    def check_neighbour(coord, labeled_map):
        my_label = labeled_map[coord[0], coord[1]]
        foot = labeled_map[coord[0] - 1:coord[0] + 2, coord[1] - 1:coord[1] + 2]
        foot = foot.ravel()
        foot = np.unique(np.delete(foot, np.where((foot == my_label) | (foot == 0))))
        return foot
