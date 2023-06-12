# Multi-View-PatchMatch
Python implementation of Multi View PatchMatch Algorithm

The following project was part of the Optional End of Course Projects in CS 532 - 3D Computer Vision at Stevens Institute of Technology. The problem statement was as follows:

- Select three images from each of the datasets/scenes and generate for each a depth map. 
- Show the resulting depth maps after each iteration.
- Report the accuracy of each generated depth map compared to the available ground truth by reporting the average pixel error and generation of an error map for each resulting depth map.

The dataset used was the Stretcha MVS dataset of which the images and their respective camera poses and calibration infomation was used. The output is not up to the mark as my implementation was taking a long time to run at full resolution and 1.00x scaling. Reducing the scale and tweaking the window size would speed up the process but will lead to poor results. However, with the output, the algorithm is performing what it is supposed to. It is generating the depth at random and then trying to create the depth map.