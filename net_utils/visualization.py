# Visualization functions
# author: ynie
# date: Feb, 2020

import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
import matplotlib.patches as patches
COLORS = np.array([
        [1., 1., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [1., 0., 1., 1.],
        [0., 1., 1., 1.],
        [1., 0., 0., 1.],
        [1., 0.5, 0.5, 1.],
        [0.5, 0.5, 1., 1.],
        [0.5, 1., 0.5, 1.],
        [1., 0.25, 0.25, 1.],
        [0.25, 1., 0.25, 1.],
        [0.25, 0.25, 1., 1.],
        [0.25, 0.1, 1., 1.],
        [0.25, 0.1, 0.1, 1.],
        [0.1, 0.1, 1., 1.],
        [0.1,   1, 0.1, 1.],
        [0.1, 0.1, 1., 1.],
        [  1, 0.1, 0.1, 1.],
        [  1, 0.1,   1, 1.],
        [  1,   1, 0.1, 1.],
        [1., 0.8, 0.8, 1.],
        [0.8, 0.8, 1., 1.],
        [0.8, 1., 0.8, 1.],
        [1.,  1, 0.5, 1.],
        [1,  0.5, 1., 1.],
        [0.5, 1., 1, 1.],

        [1.,   1, 0.8, 1.],
        [1, 0.8, 1., 1.],
        [0.8, 1., 1., 1.],
        ])
Ncolors = COLORS.shape[0]



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


def visualize_voxels2(voxels, out_file=None, show=False):
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
    #voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=225)
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


def visualize_mesh(points, faces,
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
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=faces)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=225)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_pointcloud_birdeye(points, color, normals=None,
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
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 15))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(pts_world[:, 0], pts_world[:, 1], pts_world[:, 2], s=0.01)
    # xy plane
    ax1.scatter(points[:, 0], points[:, 1], s=0.3, edgecolors=color) #top view xz
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    #ax1.invert_yaxis()

    plt.savefig(out_file)
    plt.close(fig)




def visualize_pointcloud_boxes(points, bboxes, box_label_mask=None, point_instance_labels=None, normals=None, object_point_label=None,
                         out_file=None, show=False, additional_points=None, draw_line=False, vote_xyz=None, debug=False):
    '''
    Visualizes point cloud data.
    :param points (tensor, Npoints x 3): point data
    :param bboes (Nobjects x 6): normal data (if existing), xyz, hwd
    :param box_label_mask (Nobjects): whether object exists or not
    :param out_file (string): output file
    :param show (bool): whether the plot should be shown
    :param point_label (Nobjects x Npoints)
    :param additional_points: background or dense point cloud to plot in the back
    :return:
    '''
    # Use numpy


    npoints = points.shape[0]
    nobjects = bboxes.shape[0]

    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(pts_world[:, 0], pts_world[:, 1], pts_world[:, 2], s=0.01)
    if box_label_mask is None:
        nobjects = bboxes.shape[0]
        box_label_mask = np.ones((nobjects))

    #for object_id in range(nobjects):
    if object_point_label is not None:
        assert(object_point_label.shape[0] == nobjects)
        assert(point_instance_labels == None)
        # build point_instance_labels
        point_instance_labels = (-1) * np.ones((npoints), dtype=np.int32)
        for obj_id in range(nobjects):
            if box_label_mask[obj_id] > 0.5:
                point_instance_labels[object_point_label[obj_id]] = obj_id

    # xy plane

    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    #ax1.invert_yaxis()
    from matplotlib import collections as mc
    colors = np.ones((npoints, 4))
    colors[:,:3] = 0

    output_obj_id = 0
    lines1 = []
    lines2 = []
    line_colors = []
    vote_points = []
    for obj_id in range(nobjects):
        if box_label_mask[obj_id] > 0.5:
            obj_center = bboxes[obj_id][:3]
            size = bboxes[obj_id][3:6]

            if point_instance_labels is not None:
                colors[obj_id == point_instance_labels, :] = COLORS[output_obj_id % Ncolors]
                if draw_line:

                    if object_point_label is not None:
                        selected_ids = object_point_label[obj_id]
                    else:
                        selected_ids = obj_id == point_instance_labels

                    selected_points = points[selected_ids, :]
                    if vote_xyz is None:
                        lines1 += [[(obj_center[0], obj_center[2]), (x, z)] for x, y, z in selected_points]
                        lines2 += [[(obj_center[0], obj_center[1]), (x, y)] for x, y, z in selected_points]
                    else:
                        votexyz_points = np.concatenate([vote_xyz[:,:3], points[:,:3]], axis=1)[selected_ids, :]

                        #lines1 += [[(vx, vz), (x, z), COLORS[output_obj_id % Ncolors]] for vx, vy, vz, x, y, z in votexyz_points]
                        #lines2 += [[(vx, vy), (x, y), COLORS[output_obj_id % Ncolors]] for vx, vy, vz, x, y, z in votexyz_points]
                        lines1 += [[(vx, vz), (x, z)] for vx, vy, vz, x, y, z in votexyz_points]
                        lines2 += [[(vx, vy), (x, y)] for vx, vy, vz, x, y, z in votexyz_points]
                        vote_points += [votexyz_points[:, :3]]
                    line_colors += [COLORS[output_obj_id % Ncolors]] * selected_points.shape[0]

            rect = patches.Rectangle((obj_center[0] - size[0] * 0.5, obj_center[2] - size[2] * 0.5), size[0], size[2], linewidth=1, edgecolor=COLORS[output_obj_id % Ncolors], facecolor='none')
            rect2 = patches.Rectangle((obj_center[0] - size[0] * 0.5, obj_center[1] - size[1] * 0.5), size[0], size[1], linewidth=1, edgecolor=COLORS[output_obj_id % Ncolors], facecolor='none')
            output_obj_id += 1
            ax1.add_patch(rect)
            ax2.add_patch(rect2)

    if draw_line and len(line_colors) > 0:
        line_colors = np.stack(line_colors, axis=0)
    if vote_xyz is not None and len(line_colors) > 0:
        vote_points = np.concatenate(vote_points, axis=0)

    # if draw_line:
    #     import ipdb; ipdb.set_trace()
    # draw points
    if additional_points is not None:
        ax1.scatter(additional_points[:, 0], additional_points[:, 2], s=0.01, edgecolor='k') #top view xz
        ax2.scatter(additional_points[:, 0], additional_points[:, 1], s=0.01, edgecolor='k') #side view xy
    if point_instance_labels is not None:
        ax1.scatter(points[:, 0], points[:, 2], s=0.1 if npoints > 2000 else 1.0, edgecolor=colors) #top view xz
        ax2.scatter(points[:, 0], points[:, 1], s=0.1 if npoints > 2000 else 1.0, edgecolor=colors) #side view xy
        if draw_line and len(line_colors) > 0:
            # for pt1, pt2, color in lines1:

            #     ax1.arrow(pt1[0], pt1[1], pt2[0] - pt1[0], pt2[1] - pt1[1], facecolor=color, linewidth=1)
            #ax2.plot(*sum(lines2, []))
            lc = mc.LineCollection(lines1, color=line_colors, linewidths=0.5)
            ax1.add_collection(lc)
            lc2 = mc.LineCollection(lines2, color=line_colors, linewidths=0.5)
            ax2.add_collection(lc2)
            if vote_xyz is not None:
                ax1.scatter(vote_points[:, 0], vote_points[:, 2], s=2.0, edgecolor='r') #top view xz
                ax2.scatter(vote_points[:, 0], vote_points[:, 1], s=2.0, edgecolor='r') #side view xy


    else:
        ax1.scatter(points[:, 0], points[:, 2], s=0.01) #top view xz
        ax2.scatter(points[:, 0], points[:, 1], s=0.01) #side view xy


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