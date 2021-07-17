import sys
sys.path.append('.')
import argparse
import os
from glob import glob
import trimesh
import numpy as np
import pickle
from external.libmesh import check_mesh_contains
from external import binvox_rw, voxels
from multiprocessing import Pool
from functools import partial

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

def export_voxels(mesh, dirname, clsname, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s/%s is not watertight!'
              'Cannot create voxelization.' % (clsname, modelname))
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

def export_points(mesh, filename, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % filename)
        return
    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        assert(1==2)

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

    print('Writing points: %s' % filename)
    np.savez(filename, points=points, occupancies=occupancies,
             loc=loc, scale=scale)

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


    for object_id in range(n_objects):
        vertices, faces = phases_dict["vertices_faces"][object_id]
        loc = np.zeros(3)
        scale = 1.
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
        filename = os.path.join(output_path, f'{object_id}.npz')


        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if args.visualization:
            axis = trimesh.creation.axis(axis_length=1)
            (mesh + axis).show()

        export_points(mesh, filename, loc, scale, args)


def main(args):
    data_root = "/media/htung/Extreme SSD/fish/DPI-Net/data_with_depth"
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