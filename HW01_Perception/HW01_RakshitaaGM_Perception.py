#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#Question 2
get_ipython().system('pip3 install nuscenes-devkit')


# In[2]:


get_ipython().system('pip3 install open3d')
from nuimages import NuImages
from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import NuScenesExplorer
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
nuim = NuScenes(dataroot='/data/sets/nuscenes', version='v1.0-mini', verbose=True)
nexp = NuScenesExplorer(nuim)


# In[3]:


#Question 3
my_scene = nuim.scene[0]


# In[4]:


first_sample_token = my_scene['first_sample_token']
my_sample = nuim.get('sample', first_sample_token)


# In[5]:


#Visualizing camera data
camera = 'CAM_FRONT'
cam_front_data = nuim.get('sample_data', my_sample['data'][camera])
nuim.render_sample_data(cam_front_data['token'])


# 

# In[6]:


#Visualizing radar data
radar = 'RADAR_FRONT'
radar_front_data = nuim.get('sample_data', my_sample['data'][radar])
nuim.render_sample_data(radar_front_data['token'])


# In[7]:


#Visualizing lidar data
lidar = 'LIDAR_TOP'
lidar_front_data = nuim.get('sample_data', my_sample['data'][lidar])
nuim.render_sample_data(lidar_front_data['token'])


# In[8]:


#Question 4
#(1)Visualizing image --Not using Nuscenes methods
path="/data/sets/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"
display(Image.open(path))


# In[9]:


#4.(2).(a)Visualizing lidar data
seg_name='/data/sets/nuscenes/lidarseg/v1.0-mini/4484110755904050a880043268149497_lidarseg.bin'
seg=np.fromfile(seg_name, dtype=np.uint8)
color = np.zeros([len(seg), 3])
pcd_name='/data/sets/nuscenes/samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489296047917.pcd.bin'
scan=np.fromfile(pcd_name, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]
color[:, 0] = seg/32
color[:, 1] = seg/32
color[:, 2] = seg/32
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])


# In[32]:


#4.2.(b)Colorizing Lidar PCD by height, intensity and semantic label respectively --CHANGED
pcd_name='/data/sets/nuscenes/samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489296047917.pcd.bin'
scan=np.fromfile(pcd_name, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]
height = points[:,2]
intensity = points[:,3]
color[:, 0] = height/32
color[:, 1] = 0.5
color[:, 2] = 0.5
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])


# In[34]:


color[:, 0] = intensity/32
color[:, 1] = 0.5
color[:, 2] = 0.5
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])


# In[35]:


color[:, 0] = seg/32
color[:, 1] = 0.5
color[:, 2] = 0.5
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])


# In[30]:


#4.3.(a)Visualizing radar data
radar_pcd_name='/data/sets/nuscenes/samples/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178.pcd'
radar_scan = np.fromfile(radar_pcd_name, dtype=np.float32)
radar_obj = RadarPointCloud.from_file(radar_pcd_name)
radar_points = radar_obj.points
radar_points = radar_obj.points.T
color_radar = np.zeros([len(radar_points), 3])
pcd.points = o3d.utility.Vector3dVector(radar_points[:, :3])
color_radar[:, 0] = radar_points[:,0]/32
color_radar[:, 1] = radar_points[:,1]/32
pcd.colors = o3d.utility.Vector3dVector(color_radar)
o3d.visualization.draw_geometries([pcd])


# In[14]:


#4.3.(b)Colorizing points by height and velocity
height_radar =  radar_points[:,2]
vel = radar_points[:,6]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(radar_points[:, :3])
color_radar[:, 0] = height_radar/32
color_radar[:, 1] = vel/32
pcd.colors = o3d.utility.Vector3dVector(color_radar)
o3d.visualization.draw_geometries([pcd])


# In[15]:


#5.(1)Visualize radar data projection on image
radar_points, radar_coloring, radar_image = nexp.map_pointcloud_to_image(radar_front_data['token'],cam_front_data['token'])
plt.figure(figsize=(9, 16))
plt.imshow(radar_image)
plt.scatter(radar_points[0, :], radar_points[1, :], c=radar_coloring, s=5)
plt.axis('off')


# In[16]:


#5.(1).a. Calibration info betweem radar and camera
#First Step
cs_record_radar = nuim.get('calibrated_sensor', radar_front_data['calibrated_sensor_token'])
radar_points[:3, :] = np.dot(Quaternion(cs_record_radar['rotation']).rotation_matrix,radar_points[:3,:])
x = np.array(cs_record_radar['translation'])
for i in range(3):
    radar_points[i,:]=radar_points[i,:]+x[i]
cs_record_radar


# In[17]:


#Second step
poserecord_radar = nuim.get('ego_pose', radar_front_data['ego_pose_token'])
radar_points[:3,:] = np.dot(Quaternion(poserecord_radar['rotation']).rotation_matrix, radar_points[:3,:])
x = np.array(poserecord_radar['translation'])
for i in range(3):
    radar_points[i,:]=radar_points[i,:]+x[i]
poserecord_radar


# In[18]:


#Third Step
poserecord_radar = nuim.get('ego_pose', cam_front_data['ego_pose_token'])
x = -np.array(poserecord_radar['translation'])
for i in range(3):
    radar_points[i,:]=radar_points[i,:]+x[i]
radar_points[:3,:] = np.dot(Quaternion(poserecord_radar['rotation']).rotation_matrix.T, radar_points[:3,:])
poserecord_radar


# In[19]:


#Fourth step
cs_record_radar = nuim.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
x = -np.array(cs_record_radar['translation'])
for i in range(3):
    radar_points[i,:]=radar_points[i,:]+x[i]
radar_points[:3,:] = np.dot(Quaternion(cs_record_radar['rotation']).rotation_matrix.T, radar_points[:3,:])
cs_record_radar


# In[20]:


#Fifth step
depths_radar = radar_points[2, :]
depths_radar


# In[21]:


#5.1.(b)Calibration info --Explanation -- Changed
#The pointclouds are obtained with respect to the radar frame. These are first transformed to the ego vehicle's 
#frame at the same time as that of the recorded data.The points are rotated and the translated to vehcile's frame
#The second step is to transform the points from the ego vehicle frame to the global frame. This is obtained from 
#the ego pose field corresponding to the radar location.
#Third step is to transform the points in the global frame to the ego vehicle frame at the timestamp of the image.
#The ego vehicle pose corresponding to the camera frame is taken and the rotation nad translation matrixes are multiplied. 
#Fourth step is to transform the points from ego vehicle vehicle frame to the camera. the points from 
#the ego vehicle frame corresponding to the camera is translated adnd rotated to the camera frame/ 
#Fifth step is to store the transformed points in depths 


# In[22]:


lidar_points, lidar_coloring, lidar_image = nexp.map_pointcloud_to_image(lidar_front_data['token'],cam_front_data['token'])
plt.figure(figsize=(9, 16))
plt.imshow(lidar_image)
plt.scatter(lidar_points[0, :], lidar_points[1, :], c=lidar_coloring, s=5)
plt.axis('off')


# In[36]:


#5.2.(a)Calibration info between Lidar and Camera
#First step
cs_record_lidar = nuim.get('calibrated_sensor', lidar_front_data['calibrated_sensor_token'])
lidar_points[:3, :] = np.dot(Quaternion(cs_record_lidar['rotation']).rotation_matrix,lidar_points[:3,:])
x = np.array(cs_record_lidar['translation'])
for i in range(3):
    lidar_points[i,:]=lidar_points[i,:]+x[i]
cs_record_lidar


# In[37]:


#Second step
poserecord_lidar = nuim.get('ego_pose', lidar_front_data['ego_pose_token'])
lidar_points[:3,:] = np.dot(Quaternion(poserecord_lidar['rotation']).rotation_matrix, lidar_points[:3,:])
x = np.array(poserecord_lidar['translation'])
for i in range(3):
    lidar_points[i,:]=lidar_points[i,:]+x[i]
poserecord_lidar


# In[38]:


#Third Step
poserecord_lidar = nuim.get('ego_pose', cam_front_data['ego_pose_token'])
x = -np.array(poserecord_lidar['translation'])
for i in range(3):
    lidar_points[i,:]=lidar_points[i,:]+x[i]
lidar_points[:3,:] = np.dot(Quaternion(poserecord_lidar['rotation']).rotation_matrix.T, lidar_points[:3,:])
poserecord_lidar


# In[40]:


#Fourth step
cs_record_lidar = nuim.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
x = -np.array(cs_record_lidar['translation'])
for i in range(3):
    lidar_points[i,:]=lidar_points[i,:]+x[i]
lidar_points[:3,:] = np.dot(Quaternion(cs_record_lidar['rotation']).rotation_matrix.T, lidar_points[:3,:])
cs_record_lidar 


# In[41]:


#Fifth step
depths_lidar = lidar_points[2, :]
depths_lidar

