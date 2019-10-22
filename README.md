# GridMapDecompose
Simple code used for decomposing occupancy grid map into set of corners and walls. The idea is to utilize the pattern in a structured environment, which is a combination of straight walls and corners. 

The code outputs a labeled map with corresponding minimal bounding boxes for each element in the environment and label them as walls or corners. The code also provides adjacency matrix connecting neighbouring segments of the map.


 
 ![Example of labeling](https://github.com/tkucner/GridMapDecompose/blob/master/result.png)
 
## Installation 

```bash
pip install -i https://test.pypi.org/simple/GridMapDecompose
```
## Usage

An example of usage van be found in map_segmentation_test.py:
```bash
python3 map_segmentation_test.py example.pgm regular
```