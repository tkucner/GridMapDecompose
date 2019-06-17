import argparse

import matplotlib.pyplot as plt

from GridMapDecompose import map_handling as mh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file', help = 'Path to the map')
    parser.add_argument('threshold_type', help = 'Threshold type')
    args = parser.parse_args()

    test_map = mh.GridMapHandling()
    test_map.load_map_flat(args.img_file)
    test_map.threshold_map(args.threshold_type)
    test_map.fill_gaps(1)
    test_map.slice_map()
    test_map.get_segments()
    test_map.evaluate_segments()

    fig, ax = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True)

    ax[0].imshow(test_map.binary_map, cmap = "nipy_spectral")

    for local_segment, local_segment_type in zip(test_map.segments, test_map.segment_type):

        if local_segment_type is 's':
            ax[0].plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'g')

    ax[0].axis('off')

    ax[1].imshow(test_map.binary_map, cmap = "nipy_spectral")

    for local_segment, local_segment_type in zip(test_map.segments, test_map.segment_type):

        if local_segment_type is 'f':
            ax[1].plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'r')
    ax[1].axis('off')

    ax[2].imshow(test_map.binary_map, cmap = "nipy_spectral")
    for local_corner in test_map.corner_segments:
        ax[2].plot(local_corner.minimal_bounding_box[:, 1], local_corner.minimal_bounding_box[:, 0], 'y')
    ax[2].axis('off')

    plt.show()
