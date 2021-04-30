#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install open3d')


# In[3]:


import open3d as o3d


# In[2]:


get_ipython().system('pip3 install numpy')
import numpy as np


# In[4]:


# Problem 1
lidar_bin_name = '002_00000000.bin'
scan = np.fromfile(lidar_bin_name, dtype=np.float32)
scan


# In[5]:


points = scan.reshape((-1,4))[:, :4]
intensity = points[:,3]
x_points_raw = points[:,0]
y_points_raw = points[:,1]
z_points_raw = points[:,2]


# In[6]:


color = np.zeros([len(intensity), 3])
color[:, 0] = intensity/255
color[:, 1] = intensity/255
color[:, 2] = intensity/255
pcd = o3d.geometry.PointCloud()
# Lidar points are visualized here
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])
pcd


# In[7]:


# Problem 2
# Voxel filter was performed and the points were dwonsampled and displayed
pcd_down = pcd.voxel_down_sample(0.5)
o3d.visualization.draw_geometries([pcd_down])
pcd_down


# In[64]:


# Problem 3
# Apply RANSAC and find the ground model
plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.2,
                                         ransac_n=3,
                                         num_iterations=10000)
[a, b, c, d] = plane_model
# To print the ground plane equation
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
a,b,c,d


# In[65]:


inlier_cloud = pcd_down.select_by_index(inliers)
outlier_cloud = pcd_down.select_by_index(inliers, invert=True)
# To convert pcd to array
xyz = np.asarray(pcd_down.points)
x_points = xyz[:, 0]
y_points = xyz[:, 1]
z_points = xyz[:, 2]
xyz


# In[66]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# to plot the plane
x = np.linspace(-80,80,100)
y = np.linspace(-80,80,100)

X,Y = np.meshgrid(x,y)
Z = (-d- a*X - b*Y) / c


# In[67]:


inlier_points = np.asarray(inlier_cloud.points)
inlier_x_points = inlier_points[:, 0]
inlier_y_points = inlier_points[:, 1]
inlier_z_points = inlier_points[:, 2]
inlier_points


# In[68]:


# To plot the points
# Create the figure
fig = plt.figure(figsize = (20, 20))

# Add an axes
ax = fig.add_subplot(111,projection='3d')

# plot the surface
ax.plot_surface(X,Y,Z, cmap='gray')

# and plot the point 
ax.scatter(inlier_x_points , inlier_y_points , inlier_z_points, color = 'red')


# In[70]:


# To visualize the off-ground points
o3d.visualization.draw_geometries([outlier_cloud])


# In[62]:


# Problem 4
# To make the 3D points into 2D points
outlier_points = np.asarray(outlier_cloud.points)
x = outlier_points[:,0]
y = outlier_points[:,1]
array = np.vstack((x,y)).T

plt.scatter(x,y,s=0.1,cmap= 'plasma')
# plt.imshow(array,cmap = 'plasma')
plt.show()

