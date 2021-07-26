import sys
sys.path.append('.')
import argparse
import os
from glob import glob
import trimesh
import numpy as np
from scipy import ndimage
import math

import pickle
#from external.librender import pyrender
import pyrender
from external.libmesh import check_mesh_contains
from external import binvox_rw, voxels
from multiprocessing import Pool
from functools import partial
import mcubes
from external import pyfusion



focal_length_x, focal_length_y = 640., 640.
principal_point_x, principal_point_y = 320., 320.
image_height, image_width = 640, 640
n_camera_views = 200
render_intrinsics = np.array([focal_length_x, focal_length_y, principal_point_x, principal_point_y])
image_size = np.array([image_height, image_width], dtype=np.int32)
znf = np.array([1 - 0.95, 1 + 0.95], dtype=float)
fusion_intrisics = np.array([
    [focal_length_x, 0, principal_point_x],
    [0, focal_length_y, principal_point_y],
    [0, 0, 1]
])
resolution = 256
truncation_factor = 10
voxel_size = 1. / resolution
truncation = truncation_factor*voxel_size
padding = 0.1
GT_N_Points = 100000

def parse_args():
    '''Parameters'''
    parser = argparse.ArgumentParser('Prepare tdw_physics Data.')
    parser.add_argument('--scenario', type=str, default='dominoes',
                        help='scenario to parse.')
    parser.add_argument('--mode', type=str, default='train',
                        help='scenario to parse.')
    parser.add_argument('--in_folder', type=str, default='datasets/ShapeNetv2_data/watertight',
                        help='Path to input watertight meshes.')
    parser.add_argument('--n_proc', type=int, default=12,
                        help='Number of processes to use.')
    parser.add_argument('--resize', action='store_true',
                        help='When active, resizes the mesh to bounding box.')
    parser.add_argument('--bbox_in_folder', type=str, default='datasets/ShapeNetCore.v2',
                        help='Path to other input folder to extract'
                             'bounding boxes.')
    parser.add_argument('--pointcloud_folder', type=str, default='datasets/ShapeNetv2_data/pointcloud',
                        help='Output path for point cloud.')
    parser.add_argument('--pointcloud_size', type=int, default=100000,
                        help='Size of point cloud.')
    parser.add_argument('--voxels_folder', type=str, default='datasets/ShapeNetv2_data/voxel',
                        help='Output path for voxelization.')
    parser.add_argument('--voxels_res', type=int, default=[16],
                        help='Resolution for voxelization.')
    parser.add_argument('--points_folder', type=str, default='/media/htung/Extreme SSD/fish/RfDNet/data_with_depth',
                        help='Output path for points.')
    parser.add_argument('--points_size', type=int, default=100000,
                        help='Size of points.')
    parser.add_argument('--points_uniform_ratio', type=float, default=0.5,
                        help='Ratio of points to sample uniformly'
                             'in bounding box.')
    parser.add_argument('--points_sigma', type=float, default=0.01,
                        help='Standard deviation of gaussian noise added to points'
                             'samples on the surfaces.')
    parser.add_argument('--points_padding', type=float, default=0.1,
                        help='Additional padding applied to the uniformly'
                             'sampled points on both sides (in total).')
    parser.add_argument('--mesh_folder', type=str, default='datasets/ShapeNetv2_data/watertight_scaled',
                        help='Output path for mesh.')
    parser.add_argument('--visualization', action='store_true',
                        help='Whether to overwrite output.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Whether to overwrite output.')
    parser.add_argument('--float16', action='store_true',
                        help='Whether to use half precision.')
    parser.add_argument('--packbits', action='store_true',
                        help='Whether to save truth values as bit array.')
    return parser.parse_args()

def export_pointcloud(mesh, filename, loc, scale, args):
    if not args.overwrite and os.path.exists(filename):
        print('Pointcloud already exist: %s' % filename)
        return
    points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    normals = mesh.face_normals[face_idx]

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)
    normals = normals.astype(dtype)

    print('Writing pointcloud: %s' % filename)
    np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)


import pymeshfix

def export_voxels(mesh, dirname, clsname, modelname, loc, scale, args):
    if not mesh.is_watertight:



        print('Warning: mesh %s/%s is not watertight!'
              'Cannot create voxelization.' % (clsname, modelname))
        import ipdb; idpb.set_trace()
        return

    for res in args.voxels_res:
        filename = os.path.join(dirname, str(res), clsname, modelname + '.binvox')
        if not args.overwrite and os.path.exists(filename):
            print('Voxels already exist: %s' % filename)
            return
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        voxels_occ = voxels.voxelize_ray(mesh, res)
        voxels_out = binvox_rw.Voxels(voxels_occ, (res,) * 3,
                                      translate=loc, scale=scale,
                                      axis_order='xyz')
        print('Writing voxels: %s' % filename)
        with open(filename, 'bw') as f:
            voxels_out.write(f)

def get_points(n_views=100):
    """
    See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

    :param n_points: number of points
    :type n_points: int
    :return: list of points
    :rtype: numpy.ndarray
    """

    rnd = 1.
    points = []
    offset = 2. / n_views
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    # visualization.plot_point_cloud(np.array(points))
    return np.array(points)

def get_views(n_camera_views):
    """
    Generate a set of views to generate depth maps from.

    :param n_views: number of views per axis
    :type n_views: int
    :return: rotation matrices
    :rtype: [numpy.ndarray]
    """

    Rs = []
    points = get_points(n_camera_views)

    for i in range(points.shape[0]):
        # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(latitude), -math.sin(latitude)],
                        [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array([[math.cos(longitude), 0, math.sin(longitude)],
                        [0, 1, 0],
                        [-math.sin(longitude), 0, math.cos(longitude)]])

        R = R_y.dot(R_x)
        Rs.append(R)

    return Rs


def render(vertices, faces, camera_Rs):

    depth_maps = []
    for camera_R in camera_Rs:
        np_vertices = camera_R.dot(vertices.astype(np.float64).T)
        np_vertices[2, :] += 1
        np_faces = faces.astype(np.float64)

        #depthmap, mask, img = pyrender.render(np_vertices.copy(), np_faces.T.copy(), render_intrinsics, znf, image_size)
        pose= np.eye(4)
        pose[1,1] = -1
        pose[2,2] = -1
        import trimesh
        mesh = trimesh.Trimesh(vertices=np_vertices.T, faces=np_faces)

        # camera_node = scene.get_nodes(obj=camera).pop()
        #scene = pyrender.Scene.from_trimesh_scene(mesh)

        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        camera = pyrender.PerspectiveCamera(yfov=2*np.arctan (image_size[0] / (2 * focal_length_x)), aspectRatio=1.0)

        scene.add(camera, pose=pose)
        r = pyrender.OffscreenRenderer(image_size[0], image_size[1])
        image, depthmap = r.render(scene)

        import ipdb; ipdb.set_trace()
        #import imageio
        #imageio.imwrite("depth1.png", depthmap)
        # colors = np.ones((np_vertices.shape[1], 4), dtype=np.float32)
        # mesh2 = trimesh.Trimesh(vertices=np_vertices.T, faces=np_faces, colors=colors)
        # axis = trimesh.creation.axis(axis_length=1)
        # (mesh2 + axis).show()
        # import ipdb; ipdb.set_trace()
        depthmap -= 1.5 * voxel_size
        depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))
        depth_maps.append(depthmap)
    return depth_maps

def fuse_depthmaps(depthmaps, Rs):
    Ks = fusion_intrisics.reshape((1, 3, 3))
    Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

    Ts = []
    for i in range(len(Rs)):
        Rs[i] = Rs[i]
        Ts.append(np.array([0, 0, 1]))

    Ts = np.array(Ts).astype(np.float32)
    Rs = np.array(Rs).astype(np.float32)

    depthmaps = np.array(depthmaps).astype(np.float32)
    views = pyfusion.PyViews(depthmaps, Ks, Rs, Ts)
    tsdf = pyfusion.tsdf_gpu(views, resolution, resolution, resolution, voxel_size, truncation, False)
    mask_grid = pyfusion.tsdfmask_gpu(views, resolution, resolution, resolution, voxel_size, truncation, False)
    tsdf[mask_grid==0.] = truncation
    tsdf = np.transpose(tsdf[0], [2, 1, 0])
    return tsdf

def make_watertight(mesh):
    n_camera_views = 200

    vertices = mesh.vertices
    faces = mesh.faces
    #faces = np.array([[int(face_id.split('/')[0]) for face_id in item] for item in faces])

    '''Scale to [-0.5, 0.5], center at 0.'''
    center = (vertices.max(0) + vertices.min(0)) / 2.
    scale = max(vertices.max(0) - vertices.min(0))
    scale = scale / (1 - padding)
    vertices_normalized = (vertices - center) / scale

    '''Render depth maps'''
    camera_Rs = get_views(n_camera_views)
    depths = render(vertices=vertices_normalized, faces=faces, camera_Rs=camera_Rs)


    import imageio
    imageio.imwrite("depth.png", np.concatenate(depths[:200:10], axis=1))

    #import ipdb; ipdb.set_trace()

    '''Fuse depth maps'''
    tsdf = fuse_depthmaps(depths, camera_Rs)
    # To ensure that the final mesh is indeed watertight
    tsdf = np.pad(tsdf, 1, 'constant', constant_values=1e6)
    vertices, triangles = mcubes.marching_cubes(-tsdf, 0)
    # Remove padding offset
    vertices -= 1
    # Normalize to [-0.5, 0.5]^3 cube
    vertices /= resolution
    vertices -= 0.5

    '''scale back'''
    vertices = vertices * scale + center
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return new_mesh
    #import ipdb; ipdb.set_trace()
    #mcubes.export_off(vertices, triangles, "tmp.obj")




def export_points(mesh, filename, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points. Converting it...' % filename)
        mesh2 = make_watertight(mesh)
        # meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)

        # meshfix.repair()
        # meshfix.plot()
        # meshfix.v
        # meshfix.f

        #axis = trimesh.creation.axis(axis_length=1)
        #(mesh + axis).show()
        #import ipdb; ipdb.set_trace()

        axis = trimesh.creation.axis(axis_length=1)
        (mesh2 + axis).show()

        import ipdb; ipdb.set_trace()


        #return

    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform


    extents = mesh.bounds

    boxsize = extents[1, :] - extents[0,:]
    center = (extents[1, :] + extents[0,:])/2
    boxsize = np.maximum(boxsize, 1) + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5) + center

    points_surface = mesh.sample(n_points_surface)
    points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    occupancies = check_mesh_contains(mesh, points)

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)

    if args.packbits:
        occupancies = np.packbits(occupancies)
    if args.visualization:
        npoints = points.shape[0]

        colors = np.zeros((npoints, 4))
        colors[occupancies, :] = np.array([[0,1,1,1]])
        colors[~occupancies, :] = np.array([[1,0,1,1]])

        pts_cam = trimesh.PointCloud(points, colors)
        pts_cam.show()
    print(points.shape)
    print(occupancies.shape)
    return points, occupancies
    #print('Writing points: %s' % filename)
    #np.savez(filename, points=points, occupancies=occupancies,
    #         loc=loc, scale=scale)

def export_mesh(mesh, filename, loc, scale, args):
    if not args.overwrite and os.path.exists(filename):
        print('Mesh already exist: %s' % filename)
        return
    print('Writing mesh: %s' % filename)
    mesh.export(filename)

def process_path(in_path, data_folder, output_root, args):

    pkl_path = os.path.join(data_folder, in_path)
    output_path = os.path.join(output_root, in_path.split("/")[0])
    with open(pkl_path, "rb") as f:
        phases_dict = pickle.load(f)

    assert(phases_dict["n_objects"] == len(phases_dict["vertices_faces"]))

    n_objects = phases_dict["n_objects"]
    filename = os.path.join(output_path, f'points.npz')
    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s, delete exisitng fodler first' % filename)
        assert(1==2)

    obj_points = []
    occs = []
    locs = []

    scales = []
    for object_id in range(n_objects):
        vertices, faces = phases_dict["vertices_faces"][object_id]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if not args.resize:
            loc = np.zeros(3)
            scale = 1.
        else:
            bbox = mesh.bounding_box.bounds

            # Compute location and scale
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max()

            # Transform input mesh
            mesh.apply_translation(-loc)
            mesh.apply_scale(1 / scale)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if args.visualization:
            mesh.visual.face_colors = [200, 200, 250, 100]
            axis = trimesh.creation.axis(axis_length=1)
            (mesh + axis).show()

        points, occupancies = export_points(mesh, filename + f"_{object_id}", args)

        obj_points.append(points)
        occs.append(occupancies)
        locs.append(loc)
        scales.append(scale)
    obj_points = np.stack(obj_points, axis=0)
    occs = np.stack(occs, axis=0)
    locs = np.stack(locs, axis=0)
    scales = np.stack(scales, axis=0)

    #import ipdb; ipdb.set_trace()
    np.savez(filename, points=points, occupancies=occupancies,
             loc=loc, scale=scale)

    #print("hello")


def main(args):
    data_root = "/media/htung/Extreme SSD/fish/DPI-Net/data_with_depth_small"
    scenario_folder = os.path.join(data_root, args.scenario)


    for arg_name in os.listdir(scenario_folder):
        data_folder = os.path.join(scenario_folder, arg_name, args.mode)
        output_folder = os.path.join(args.points_folder, args.scenario, arg_name, args.mode)
        folder_ndata = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

        pkl_files = [os.path.join(path, "phases_dict.pkl")  for path in folder_ndata]

        #input_files = glob(os.path.join(data_folder, '*', '*.off'))
        for file in pkl_files:
            process_path(file, data_folder, output_folder, args=args)

    # p = Pool(processes=args.n_proc)
    # p.map(partial(process_path, args=args), input_files)
    # p.close()
    # p.join()

if __name__ == '__main__':
    args = parse_args()
    main(args)