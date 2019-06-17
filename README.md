# GridMapDecompose
Simple code used for decomposing occupancy grid map into set of corners and walls. The idea is to utilize the pattern in a structured environment, which is a combination of straight walls and corners. 

The code outputs a labeled map with corresponding minimal bounding boxes for each element in the environment. There are three types of labels:

* wall
* corner
* clustered corner
 
 ![Example of labeling](https://github.com/tkucner/GridMapDecompose/blob/master/example.png)
 
## Installation 

```bash
pip install -i https://test.pypi.org/simple/GridMapDecompose
```
## Usage

An example of usage van be found in map_segmentation_test.py:
```bash
python3 map_segmentation_test.py map.png regular
```