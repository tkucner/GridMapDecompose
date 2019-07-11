import numpy as np
from scipy.spatial.distance import correlation

# from scipy import ndimage

import networkx as nx


def hog_gradient(image):
    """Compute unnormalized gradient image along `row` and `col` axes.
    Parameters

    :param image:(M, N) ndarray
        Grayscale image or one of image channel.
    :return:
    g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
    """

    g_row = np.zeros(image.shape, dtype=np.double)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = image[2:, :] - image[:-2, :]
    # g_row = ndimage.sobel(image, axis=0, mode='constant')
    g_col = np.zeros(image.shape, dtype=np.double)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]
    # g_col = ndimage.sobel(image, axis=1, mode='constant')
    magnitude = np.hypot(g_col, g_row)
    orientation = np.rad2deg(np.arctan2(g_row, g_col)) % 180
    orientations = np.array([0., 45., 90., 135.])
    orientation_histogram = np.zeros(orientations.shape)
    for orie in orientations:
        h = np.sum(magnitude[orientation == orie])
        orientation_histogram[orientations == orie] = h
    orientation_histogram = orientation_histogram / np.sum(orientation_histogram)
    return orientation_histogram


class Graph:
    def __init__(self):
        self.window_size = 2
        self.map = []
        self.similarity_threshold = 0.09
        self.adjacency_matrix = []
        self.nodes = list([])
        self.node_labels = list([])
        self.walls = []
        self.features = []
        self.WA = []
        self.CA = []
        self.W = []
        self.C = []

    def compute_point_features(self, coordinates, point_id):
        """Compute the features of an occupied cell in skeletonised map.

        :param coordinates: ndarray with a position of a cell in the map
        :param point_id: unique ide of the point
        :return:
        """
        window = self.map[coordinates[0] - self.window_size:coordinates[0] + self.window_size + 1,
                 coordinates[1] - self.window_size:coordinates[1] + self.window_size + 1] * 1

        return {
            'id': point_id,
            'coordinates': coordinates,
            'hog': hog_gradient(window)
        }

    def find_neighbours(self, coordinates):
        foot = self.map[coordinates[0] - 1:coordinates[0] + 2, coordinates[1] - 1:coordinates[1] + 2]
        p = np.where(foot)
        col1 = np.array(coordinates[0] + p[0] - 1)
        col2 = np.array(coordinates[1] + p[1] - 1)
        neighbours = np.concatenate((col1, col2)).reshape((2, -1)).transpose()
        f = (neighbours == coordinates).all(axis=1).nonzero()
        neighbours = np.delete(neighbours, f, axis=0)
        return neighbours

    def set_window_size(self, window_size):
        self.window_size = window_size

    def score_edge(self, id_1, id_2):

        node_a = self.nodes[id_1]
        node_b = self.nodes[id_2]

        return correlation(node_a['hog'], node_b['hog'])

    def build_graph(self, input_map):
        self.map = input_map
        cells = np.nonzero(input_map)

        self.adjacency_matrix = np.full((cells[0].size, cells[0].size), np.nan)

        id = 0
        cells_a = np.concatenate((cells[0], cells[1])).reshape((2, -1)).transpose()
        for cell in zip(cells[0], cells[1]):
            cell = np.array(cell)

            self.nodes.append(self.compute_point_features(cell, id))
            neigbours = self.find_neighbours(cell)
            for neighbour in neigbours:
                f = (cells_a == neighbour).all(axis=1).nonzero()

                self.adjacency_matrix[f, id] = 1
                self.adjacency_matrix[id, f] = 1
            id += 1
        # find all edges id_1->id_2
        for node in self.nodes:
            adjacency_row = self.adjacency_matrix[node['id']]
            adjacent_cells = np.flatnonzero(np.isfinite(adjacency_row) * 1)

            for a_cell in adjacent_cells:
                score = self.score_edge(node['id'], a_cell)
                self.adjacency_matrix[node['id'], a_cell] = score
                self.adjacency_matrix[a_cell, node['id']] = score

    def label_nodes(self):
        for row in self.adjacency_matrix:
            self.node_labels.append(np.any(row > self.similarity_threshold))

    def split_to_subgraphs(self):
        self.walls = np.array(self.adjacency_matrix.copy())
        self.walls[self.walls >= self.similarity_threshold] = np.nan

        result = np.where(~np.isnan(self.walls))
        list_of_coordinates = list(zip(result[1], result[0]))

        self.WA = nx.Graph()

        self.WA.add_edges_from(list_of_coordinates)
        self.W = list(nx.connected_component_subgraphs(self.WA))

        self.features = np.array(self.adjacency_matrix.copy())
        self.features[self.features < self.similarity_threshold] = np.nan
        result = np.where(~np.isnan(self.features))
        list_of_coordinates = np.array(list(zip(result[0], result[1])))
        self.CA = nx.Graph()
        self.CA.add_edges_from(list_of_coordinates)
        self.C = list(nx.connected_component_subgraphs(self.CA))

        for sg in self.W:
            if sg.number_of_nodes() == 2:
                n = list(sg.nodes)
                self.WA.remove_node(n[0])
                self.WA.remove_node(n[1])
                self.CA.add_edge(n[0], n[1])

        local_wa_nodes = list(self.WA.nodes)
        local_ca_nodes = list(self.CA.nodes)

        # for wa_node in local_wa_nodes:
        #     if wa_node in local_ca_nodes:
        #         self.CA.remove_node(wa_node)

        for ca_node in local_ca_nodes:
            if ca_node in local_wa_nodes:
                self.WA.remove_node(ca_node)

        self.W = list(nx.connected_component_subgraphs(self.WA))
        self.C = list(nx.connected_component_subgraphs(self.CA))
