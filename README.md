# Fast-Global-Registration
This repository contains an implementation of the Fast-Global Registration (FGR) Algorithm, which is commonly used for point cloud registration tasks. The algorithm is designed to efficiently register 3D point clouds by minimizing a robust global objective function.

## Introduction
Fast-Global Registration (FGR) is an advanced method used in computer vision and robotics to align two or more 3D point clouds. It efficiently handles large datasets and provides accurate alignments with a fast computational speed, making it suitable for real-time applications.

The implementation here is based on the original paper: [Fast Global Registration](https://vladlen.info/publications/fast-global-registration/) by Zhou et al., 2016.

## Available Demo Notebooks
The repository includes demo notebooks that allow you to visualize the FGR algorithm:
* `optimization.ipynb`: provides a visualization of the optimization loop on a simple 2-dimensional spline.
* `correspondence.ipynb`: provides a visualization of the point correspondence creation process step by step.
* `registration.ipynb`: provides a comparison between our implementation and the implementation in the open3d package.

## How to Run the Registration
The registration can be run on any point-clouds by the functions `normalize_point_clouds` and `fast_global_registration` like so:
```python
import open3d as o3d

from src.logic.normalization import normalize_point_clouds
from src.logic.registration import fast_global_registration

pcd_p = o3d.io.read_point_cloud("path/to/first/point_cloud.ply")
pcd_q = o3d.io.read_point_cloud("path/to/second/point_cloud.ply")

pcd_p, pcd_q = normalize_point_clouds(pcd_p, pcd_q)
transformation = fast_global_registration(pcd_p, pcd_q)
# returns a transformation matrix that aligns pcd_q to the pcd_p
```

## Setup and Installation
### Python Version
The code is compatible with Python 3.11 or earlier.

### Requirements
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```
### Installation
#### 1) Clone the repository:

```bash
git clone https://github.com/karinsifri/Fast-Global-Registration.git
cd Fast-Global-Registration
```
#### 2) Install the required packages:
```bash
pip install -r requirements.txt
```
#### 3) Run the demo notebooks:
```bash
jupyter notebook
```
## References
* Zhou, Q.-Y., Park, J., & Koltun, V. (2016). Fast Global Registration. In Proceedings of the European Conference on Computer Vision (ECCV).