import copy

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import io
from skimage.filters import threshold_yen
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

from GridMapDecompose import segment_handling as sgh, graph_handling as gh

class GridMapHandling:
    def __init__(self):
        self.map_file_name = []

        self.grid_map = np.array([])
        self.threshold_type = np.array([])
        self.binary_map = np.array([])
        self.processed_map = np.array([])

        self.__distance_map = np.array([])

        self.skeleton = []
        self.graph = []

        self.segment_type = list([])
        self.segments = list([])


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

    def build_graphs(self):
        if self.processed_map.size == 0:
            self.processed_map = copy.deepcopy(self.binary_map * 1)
        self.skeleton = skeletonize(self.processed_map)
        self.graph = gh.Graph()
        self.graph.build_graph(self.skeleton)
        self.graph.label_nodes()
        self.graph.split_to_subgraphs()

    def evaluate_segments(self):
        for sg in self.graph.W:
            coord = []
            local_segment = sgh.Segment()
            for node_id in sg.node:
                a = self.graph.nodes[node_id]
                coord.append(a['coordinates'])
            coord = np.array(coord)
            local_segment.add_cells(coord)
            local_segment.compute_hull()
            local_segment.compute_mbb()
            self.segment_type.append('w')
            self.segments.append(local_segment)
        for sg in self.graph.C:
            coord = []
            local_segment = sgh.Segment()
            for node_id in sg.node:
                a = self.graph.nodes[node_id]
                coord.append(a['coordinates'])
            coord = np.array(coord)
            local_segment.add_cells(coord)
            local_segment.compute_hull()
            local_segment.compute_mbb()
            self.segment_type.append('f')
            self.segments.append(local_segment)
        return self.segments



    @staticmethod
    def skeletonize(map_slice):
        l_slice = map_slice
        l_slice[map_slice > 0] = 1
        skeleton = skeletonize(map_slice)
        return skeleton

