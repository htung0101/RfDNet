# Visualization functions
# author: ynie
# date: Feb, 2020

import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
import matplotlib.patches as patches

def visualize_voxels(voxels, out_file=None, show=False):
    '''
    Visualizes voxel data.
    :param voxels (tensor): voxel data
    :param out_file (string): output file
    :param show (bool): whether the plot should be shown
    :return:
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False):
    '''
    Visualizes point cloud data.
    :param points (tensor): point data
    :param normals (tensor): normal data (if existing)
    :param out_file (string): output file
    :param show (bool): whether the plot should be shown
    :return:
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    if normals is not None:
        ax.quiver(
            points[:, 0], points[:, 1], points[:, 2],
            normals[:, 0], normals[:, 1], normals[:, 2],
            length=0.1, color='k'
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)



def visualize_pointcloud_boxes(points, bboxes, box_label_mask=None, normals=None,
                         out_file=None, show=False):
    '''
    Visualizes point cloud data.
    :param points (tensor, Npoints x 3): point data
    :param bboes (Nobjects x 6): normal data (if existing), xyz, hwd
    :param box_label_mask (Nobjects): whether object exists or not
    :param out_file (string): output file
    :param show (bool): whether the plot should be shown
    :return:
    '''
    # Use numpy

    #for object_id in range(nobjects):

    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(pts_world[:, 0], pts_world[:, 1], pts_world[:, 2], s=0.01)
    if box_label_mask is None:
        nobjects = bboxes.shape[0]
        box_label_mask = np.ones((nobjects))
    # xy plane
    ax1.scatter(points[:, 0], points[:, 2], s=0.01) #top view xz
    ax2.scatter(points[:, 0], points[:, 1], s=0.01) #side view xy
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    #ax1.invert_yaxis()

    nobjects = bboxes.shape[0]
    for obj_id in range(nobjects):
        if box_label_mask[obj_id] > 0.5:
            obj_center = bboxes[obj_id][:3]
            size = bboxes[obj_id][3:6]
            rect = patches.Rectangle((obj_center[0] - size[0] * 0.5, obj_center[2] - size[2] * 0.5), size[0], size[2], linewidth=1, edgecolor='r', facecolor='none')
            rect2 = patches.Rectangle((obj_center[0] - size[0] * 0.5, obj_center[1] - size[1] * 0.5), size[0], size[1], linewidth=1, edgecolor='r', facecolor='none')

            ax1.add_patch(rect)
            ax2.add_patch(rect2)

    plt.savefig(out_file)
    plt.close(fig)

def visualize_data(data, data_type, out_file):
    '''
    Visualizes the data with regard to its type.
    :param data (tensor): batch of data
    :param data_type (string): data type (img, voxels or pointcloud)
    :param out_file (string): output file
    :return:
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)