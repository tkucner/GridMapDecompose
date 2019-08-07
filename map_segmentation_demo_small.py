import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from GridMapDecompose import map_handling as mh
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file', help='Path to the map')
    parser.add_argument('threshold_type', help='Threshold type')
    args = parser.parse_args()

    test_map = mh.GridMapHandling()
    test_map.load_map_flat_file(args.img_file)
    test_map.threshold_map(args.threshold_type)
    test_map.fill_gaps(1)

    test_map.build_graphs()
    test_map.label_map()
    test_map.evaluate_segments()

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

    ax[0].imshow(test_map.labeled_map, cmap="nipy_spectral")
    ax[0].axis('off')

    ax[1].imshow(test_map.binary_map, cmap="nipy_spectral")
    for local_segment, local_segment_type in zip(test_map.segments, test_map.segment_type):
        if local_segment_type is 'w':
            ax[1].plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'g')
        if local_segment_type is 'f':
            ax[1].plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'r')
    ax[1].axis('off')

    ax[2].imshow(test_map.binary_map, cmap="nipy_spectral")
    for local_segment, local_segment_type in zip(test_map.segments, test_map.segment_type):
        if local_segment_type is 'w':
            ax[2].plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'g')
        if local_segment_type is 'f':
            ax[2].plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'r')

    # quickly find edges
    LU_adjacency_matrix = np.triu(test_map.adjacency_matrix_segments)
    edges = np.column_stack(np.nonzero(LU_adjacency_matrix))
    for edge in edges:
        x = (test_map.segments[edge[0] - 1].center[1], test_map.segments[edge[1] - 1].center[1])
        y = (test_map.segments[edge[0] - 1].center[0], test_map.segments[edge[1] - 1].center[0])
        ax[2].plot(x, y, 'b')

    ax[2].axis('off')

    plt.show()
