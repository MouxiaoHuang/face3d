''' Simple example of pipeline
3D obj(process) --> 2d image
通过3d人脸数据重建或者说得到相应的2d图片
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh_numpy

# ------------------------------ 1. load mesh_numpy data 【mesh数据的一些格式和规则】
# -- mesh_numpy data consists of: vertices, triangles, color(optinal), texture(optional)
# -- here use colors to represent the texture of face surface
C = sio.loadmat('Data/example1.mat')                                                    # Step1: load samples's data
vertices = C['vertices']; colors = C['colors']; triangles = C['triangles']
colors = colors/np.max(colors)                                                          # 相当于均一化，设置数值范围

# ------------------------------ 2. modify vertices(transformation. change position of obj)
# -- change the position of mesh_numpy object in world space
# scale. target size=180 for example    【缩放因子】
s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
# rotate 30 degree for example  【相当于旋转矩阵】
R = mesh_numpy.transform.angle2matrix([0, 30, 0])
# no translation. center of obj:[0,0]   【即平易矩阵】
t = [0, 0, 0]
transformed_vertices = mesh_numpy.transform.similarity_transform(vertices, s, R, t)

# ------------------------------ 3. modify colors/texture(add light)
# -- add point lights. light positions are defined in world space
# set lights
light_positions = np.array([[-128, -128, 300]])
light_intensities = np.array([[1, 1, 1]])
lit_colors = mesh_numpy.light.add_light(transformed_vertices, triangles, colors, light_positions, light_intensities)

# ------------------------------ 4. modify vertices(projection. change position of camera)
# -- transform object from world space to camera space(what the world is in the eye of observer). 
# -- omit if using standard camera
camera_vertices = mesh_numpy.transform.lookat_camera(transformed_vertices, eye = [0, 0, 200], at = np.array([0, 0, 0]), up = None)
# -- project object from 3d world space into 2d image plane. orthographic or perspective projection 【3d投影至2d】
projected_vertices = mesh_numpy.transform.orthographic_project(camera_vertices)

'''
    至此，3d投影至2d的步骤完成，接下来是对投影得到的2d图进行处理以显示
'''

# ------------------------------ 5. render(to 2d image)
# set h, w of rendering
h = w = 256
# change to image coords for rendering
image_vertices = mesh_numpy.transform.to_image(projected_vertices, h, w)
# render 
rendering =  mesh_numpy.render.render_colors(image_vertices, triangles, lit_colors, h, w)

# ---- show rendering
#plt.imshow(rendering)
plt.imshow(rendering)
plt.show()
save_folder = 'results/pipeline'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
io.imsave('{}/rendering.jpg'.format(save_folder), rendering)

# ---- show mesh_numpy
#mesh_numpy.vis.plot_mesh(camera_vertices, triangles)
#plt.show()