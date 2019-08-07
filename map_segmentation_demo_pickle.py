import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from GridMapDecompose import map_handling as mh
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

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
    mh.save(test_map, "test_map")
