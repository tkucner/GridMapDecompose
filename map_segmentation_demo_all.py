import argparse

from GridMapDecompose import map_handling as mh

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
    visualisation = {"adjacency matrix": False,
                     "nodes": False,
                     "edges": False,
                     "corners and walls": False,
                     "corners and walls mbb": True,
                     "labeled map": True,
                     "graph and mbb": True}

    test_map.show(visualisation)
    # test_map.save()

    # visualize = [False, False, False, True, True, True, True]
    #
    # if visualize[0]:
    #     ################################
    #     fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #     im2 = ax.imshow(test_map.graph.adjacency_matrix)
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     plt.colorbar(im2, cax=cax)
    # if visualize[1]:
    #     ################################
    #     fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #     min_score = np.nanmin(test_map.graph.adjacency_matrix)
    #     max_score = np.nanmax(test_map.graph.adjacency_matrix)
    #     norm = plt.Normalize(min_score, max_score)
    #     cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    #     cmap.set_array([])
    #     ax1.imshow(test_map.skeleton, cmap="nipy_spectral")
    #     for it in range(0, test_map.graph.adjacency_matrix.shape[0]):
    #         if not test_map.graph.node_labels[it]:
    #             ax1.plot(test_map.graph.nodes[it]['coordinates'][1], test_map.graph.nodes[it]['coordinates'][0], 'g+')
    #         else:
    #             ax1.plot(test_map.graph.nodes[it]['coordinates'][1], test_map.graph.nodes[it]['coordinates'][0], 'r.')
    #     for it in range(0, test_map.graph.adjacency_matrix.shape[0]):
    #         for jt in range(it, test_map.graph.adjacency_matrix.shape[0]):
    #             if not np.isnan(test_map.graph.adjacency_matrix[it, jt]):
    #                 a = test_map.graph.nodes[it]
    #                 b = test_map.graph.nodes[jt]
    #                 a = a['coordinates']
    #                 b = b['coordinates']
    #                 y = [a[0], b[0]]
    #                 x = [a[1], b[1]]
    #                 im = ax1.plot(x, y, color=cm.jet(norm(test_map.graph.adjacency_matrix[it, jt])))
    #     fig1.subplots_adjust(right=0.8)
    #     cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
    #     fig1.colorbar(cmap, cax=cbar_ax)
    #     ax1.axis('off')
    #     fig1.tight_layout()
    # if visualize[2]:
    #     #############################
    #     fig2, ax2 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #     min_score = np.nanmin(test_map.graph.walls)
    #     max_score = np.nanmax(test_map.graph.walls)
    #     norm = plt.Normalize(min_score, max_score)
    #     cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    #     cmap.set_array([])
    #     ax2.imshow(test_map.skeleton, cmap="nipy_spectral")
    #     for it in range(0, test_map.graph.walls.shape[0]):
    #         for jt in range(it, test_map.graph.walls.shape[0]):
    #             if not test_map.graph.walls[it, jt] == 0.0:
    #                 a = test_map.graph.nodes[it]
    #                 b = test_map.graph.nodes[jt]
    #                 a = a['coordinates']
    #                 b = b['coordinates']
    #                 y = [a[0], b[0]]
    #                 x = [a[1], b[1]]
    #                 im = ax2.plot(x, y, color=cm.jet(norm(test_map.graph.walls[it, jt])))
    #     fig2.subplots_adjust(right=0.8)
    #     cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
    #     fig2.colorbar(cmap, cax=cbar_ax)
    #     fig2.tight_layout()
    #     ax2.axis('off')
    #
    # if visualize[3]:
    #     #############################
    #     fig3, ax3 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #     ax3.imshow(test_map.skeleton, cmap="nipy_spectral")
    #     for sg in test_map.graph.C:
    #         coord = []
    #         for node_id in sg.node:
    #             a = test_map.graph.nodes[node_id]
    #             coord.append(a['coordinates'])
    #         coord = np.array(coord)
    #         ax3.plot(coord[:, 1], coord[:, 0], 's', markerfacecolor='none')
    #     for sg in test_map.graph.W:
    #         coord = []
    #         for node_id in sg.node:
    #             a = test_map.graph.nodes[node_id]
    #             coord.append(a['coordinates'])
    #         coord = np.array(coord)
    #
    #         ax3.plot(coord[:, 1], coord[:, 0], '.')
    #
    #     fig3.tight_layout()
    #     ax3.axis('off')
    #
    # if visualize[4]:
    #     #############################
    #     fig4, ax4 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #     ax4.imshow(test_map.binary_map, cmap="nipy_spectral")
    #     for local_segment, local_segment_type in zip(test_map.segments, test_map.segment_type):
    #         if local_segment_type is 'w':
    #             ax4.plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'g')
    #         if local_segment_type is 'f':
    #             ax4.plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'r')
    #
    # ax4.axis('off')
    #
    # if visualize[5]:
    #     #############################
    #     fig5, ax5 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #     ax5.imshow(test_map.labeled_map, cmap="nipy_spectral")
    #
    # ax5.axis('off')
    #
    # if visualize[6]:
    #     #############################
    #     fig6, ax6 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #     ax6.imshow(test_map.binary_map, cmap="nipy_spectral")
    #     for local_segment, local_segment_type in zip(test_map.segments, test_map.segment_type):
    #         if local_segment_type is 'w':
    #             ax6.plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'g')
    #         if local_segment_type is 'f':
    #             ax6.plot(local_segment.minimal_bounding_box[:, 1], local_segment.minimal_bounding_box[:, 0], 'r')
    #
    #     # quickly find edges
    #     LU_adjacency_matrix = np.triu(test_map.adjacency_matrix_segments)
    #     edges = np.column_stack(np.nonzero(LU_adjacency_matrix))
    #     for edge in edges:
    #         x = (test_map.segments[edge[0] - 1].center[1], test_map.segments[edge[1] - 1].center[1])
    #         y = (test_map.segments[edge[0] - 1].center[0], test_map.segments[edge[1] - 1].center[0])
    #         ax6.plot(x, y, 'b')
    #
    # ax6.axis('off')
    #
    # plt.show()
