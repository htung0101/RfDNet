import numpy as np
import os
import trimesh
import copy
import matplotlib.pyplot as plt
import binvox_rw as binvox_rw
import math
import utils_basic

import ipdb
st = ipdb.set_trace

def calc_rigid_transform(XX, YY):
    X = XX.copy().T
    Y = YY.copy().T

    mean_X = np.mean(X, 1, keepdims=True)
    mean_Y = np.mean(Y, 1, keepdims=True)
    X = X - mean_X
    Y = Y - mean_Y
    C = np.dot(X, Y.T)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
    R = np.dot(Vt.T, np.dot(D, U.T))
    T = mean_Y - np.dot(R, mean_X)

    '''
    YY_fitted = (np.dot(R, XX.T) + T).T
    print("MSE fit", np.mean(np.square(YY_fitted - YY)))
    '''

    return R, T


def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0],v[1],v[2]))
        for face in faces:
            f.write("f")
            for vertex in face:
                f.write(" %d" % (vertex + 1))
            f.write("\n")


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


#filename = "/home/htung/Documents/2021/example_meshes/0000_obj1_1.binvox"

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def apply_4x4(RT, XYZ):
    """
    RT: B x 4 x 4
    XYZ: B x N x 3
    """
    B, N, _ = XYZ.shape
    ones = np.ones([B, N, 1])
    XYZ1 = np.concatenate([XYZ, ones], 2)
    XYZ1_t = np.transpose(XYZ1, (0, 2, 1))
    # this is B x 4 x N

    XYZ2_t = np.matmul(RT, XYZ1_t)
    XYZ2 = np.transpose(XYZ2_t, (0, 2, 1))
    XYZ2 = XYZ2[:,:,:3]
    return XYZ2


def Pixels2CameraSingle(x, y, z, fx, fy, x0, y0):
    # x, y are N,
    # z is corresponding depth so it is N,
    # fx, fy, x0, y0 are all N,
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)
    xyz = np.stack([x,y,z], axis=1)
    return xyz


def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    # there is no randomness here

    B, H, W = list(z.shape)

    fx = np.reshape(fx, [B,1,1])
    fy = np.reshape(fy, [B,1,1])
    x0 = np.reshape(x0, [B,1,1])
    y0 = np.reshape(y0, [B,1,1])

    # unproject
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)

    x = np.reshape(x, [B,-1])
    y = np.reshape(y, [B,-1])
    z = np.reshape(z, [B,-1])
    xyz = np.stack([x,y,z], axis=2)
    return xyz

def get_intrinsics_from_projection_matrix(proj_matrix, size):
    H, W = size
    vfov = 2.0 * math.atan(1.0/proj_matrix[1][1]) * 180.0/ np.pi
    vfov = vfov / 180.0 * np.pi
    tan_half_vfov = np.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov * H / float(H)
    fx = W / 2.0 / tan_half_hfov  # focal length in pixel space
    fy = H / 2.0 / tan_half_vfov

    pix_T_cam = np.array([[fx, 0, W / 2.0],
                           [0, fy, H / 2.0],
                                   [0, 0, 1]])
    return pix_T_cam



def get_depth_values(image: np.array, depth_pass: str = "_depth", width: int = 256, height: int = 256, near_plane: float = 0.1, far_plane: float = 100) -> np.array:
    """
    Get the depth values of each pixel in a _depth image pass.
    The far plane is hardcoded as 100. The near plane is hardcoded as 0.1.
    (This is due to how the depth shader is implemented.)
    :param image: The image pass as a numpy array.
    :param depth_pass: The type of depth pass. This determines how the values are decoded. Options: `"_depth"`, `"_depth_simple"`.
    :param width: The width of the screen in pixels. See output data `Images.get_width()`.
    :param height: The height of the screen in pixels. See output data `Images.get_height()`.
    :param near_plane: The near clipping plane. See command `set_camera_clipping_planes`. The default value in this function is the default value of the near clipping plane.
    :param far_plane: The far clipping plane. See command `set_camera_clipping_planes`. The default value in this function is the default value of the far clipping plane.
    :return An array of depth values.
    """
    image = np.flip(np.reshape(image, (height, width, 3)), 1)

    # Convert the image to a 2D image array.
    if depth_pass == "_depth":
        depth_values = np.array((image[:, :, 0] + image[:, :, 1] / 256.0 + image[:, :, 2] / (256.0 ** 2)))
    elif depth_pass == "_depth_simple":
        depth_values = image[:, :, 0] / 256.0
    else:
        raise Exception(f"Invalid depth pass: {depth_pass}")
    # Un-normalize the depth values.
    return (depth_values * ((far_plane - near_plane) / 256.0)).astype(np.float32)

def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)  # this is 1, 1, H, W
    y, x = utils_basic.meshgrid2D_py(H, W)
    y = np.repeat(y[np.newaxis, :, :], B, axis=0)
    x = np.repeat(x[np.newaxis, :, :], B, axis=0)
    z = np.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz


def mesh_to_particles(mesh_filename, spacing=0.2, visualization=False, remove_outofbbox_pts=True):
    """
    mesh_filename:"/home/htung/Documents/2021/example_meshes/Rubber_duck.obj"
    spacing: the size of the voxel grid in real-world scale # what is the distance between 2 particles
    """

    # the output path used by binvox
    output_binvox_filename = mesh_filename.replace(".obj", ".binvox")
    # check if output file exists
    assert not os.path.isfile(output_binvox_filename), f"binvox file {output_binvox_filename} exists, please delete it first"

    #load mesh
    mesh = as_mesh(trimesh.load_mesh(mesh_filename, process=True))
    # make the mesh transparent
    mesh_ori = copy.deepcopy(mesh) # for visualization
    mesh_ori.visual.face_colors[:,3] = 120



    edges = mesh.bounding_box.extents
    maxEdge = max(edges)
    meshLower0 = mesh.bounds[0,:]
    meshUpper0 = mesh.bounds[1,:]

    # shift the mesh to it is in some bounding box [0, +x], [0, +y], [0, +z]
    #mesh.vertices -= meshLower0


    edges = mesh.bounding_box.extents
    maxEdge = max(edges)
    meshLower = mesh.bounds[0,:]
    meshUpper = mesh.bounds[1,:]
    #  tweak spacing to avoid edge cases for particles laying on the boundary
    # just covers the case where an edge is a whole multiple of the spacing.
    spacingEps = spacing*(1.0 - 1e-4)
    spacingEps_p = 0 #(9e-4) if spacing >= 0.001 else 0


    # naming is confusing, dx denotes the number of voxels in each dimension
    # make sure to have at least one particle in each dimension
    dx = 1 if spacing > edges[0] else int(edges[0]/spacingEps)
    dy = 1 if spacing > edges[1] else int(edges[1]/spacingEps)
    dz = 1 if spacing > edges[2] else int(edges[2]/spacingEps)

    maxDim = max(max(dx, dy), dz);

    #expand border by two voxels to ensure adequate sampling at edges
    # extending by a small offset to avoid point sitting exactly on the boundary
    meshLower_spaced = meshLower - 2.0 * spacing - spacingEps_p
    meshUpper_spaced = meshUpper +  2.0 * spacing + spacingEps_p

    maxDim_spaced = maxDim + 4

    voxelsize_limit = 512
    if maxDim_spaced > voxelsize_limit :
        print(meshLower_spaced)
        print(meshUpper_spaced)
        print("=====")
        for dim in range(3):
            if edges[dim] < (voxelsize_limit - 4) * spacing:
                continue # short edge, no need to chunk
            else:
                amount_to_cut = edges[dim] - (voxelsize_limit - 4) * spacing
                meshLower_spaced[dim] += amount_to_cut * 0.5
                meshUpper_spaced[dim] -= amount_to_cut * 0.5
        maxDim_spaced = voxelsize_limit

    # we shift the voxelization bounds so that the voxel centers
    # lie symmetrically to the center of the object. this reduces the
    # chance of missing features, and also better aligns the particles
    # with the mesh
    # ex. |1|1|1|0.3| --> |0.15|1|1|0.15|
    meshOffset = np.zeros((3))
    meshOffset[0] = 0.5 * (spacing - (edges[0] - (dx-1)*spacing))
    meshOffset[1] = 0.5 * (spacing - (edges[1] - (dy-1)*spacing))
    meshOffset[2] = 0.5 * (spacing - (edges[2] - (dz-1)*spacing))
    meshLower_spaced -= meshOffset;

    # original space
    #meshLower_spaced += meshLower0
    meshUpper_spaced = meshLower_spaced + maxDim_spaced * spacing + 2 * spacingEps_p

    #print(meshLower_spaced, meshUpper_spaced)
    #print(f'binvox -aw -dc -d {maxDim_spaced} -pb -bb {meshLower_spaced[0]} {meshLower_spaced[1]} {meshLower_spaced[2]} {meshUpper_spaced[0]} {meshUpper_spaced[1]} {meshUpper_spaced[2]} -t binvox {mesh_filename}')

    # voxelsize_limit = 512
    # if maxDim_spaced > voxelsize_limit :
    #     import ipdb; ipdb.set_trace()
    #     cutting_space = spacing * (maxDim_spaced -voxelsize_limit ) * 0.5
    #     maxDim_spaced = voxelsize_limit
    #     meshLower_spaced += cutting_space
    #     meshUpper_spaced -= cutting_space

    #     import ipdb; ipdb.set_trace()


    os.system(f'binvox -aw -dc -d {maxDim_spaced} -pb -bb {meshLower_spaced[0]} {meshLower_spaced[1]} {meshLower_spaced[2]} {meshUpper_spaced[0]} {meshUpper_spaced[1]} {meshUpper_spaced[2]} -t binvox {mesh_filename}')
    #print(meshLower_spaced, meshUpper_spaced)

    # binvox -aw -dc -d 5 -pb -bb -0.9 -0.4 -0.9 0.9 1.4 0.9 -t binvox {mesh_filename}
    #os.system(f"binvox -aw -dc -d 5 -pb -bb -0.9 -0.4 -0.9 0.9 1.4 0.9 -t binvox {mesh_filename}")

    # convert voxel into points

    with open(output_binvox_filename, 'rb') as f:
         m1 = binvox_rw.read_as_3d_array(f)



    adjusted_spacing = (maxDim_spaced * spacing + 2 * spacingEps_p)/maxDim_spaced
    x, y, z = np.nonzero(m1.data)
    points = np.expand_dims(meshLower_spaced, 0) + np.stack([(x + 0.5)*adjusted_spacing, (y + 0.5)*adjusted_spacing, (z + 0.5)*adjusted_spacing], axis=1)
    os.remove(output_binvox_filename)


    if remove_outofbbox_pts:

        bbox = mesh_ori.bounds
        lower_bound = bbox[0, :]
        upper_bound = bbox[1, :]

        idx = (points[:, 0] - upper_bound[0] <= 0) * (points[:, 0] - lower_bound[0] >= 0)
        idy = (points[:, 1] - upper_bound[1] <= 0) * (points[:, 1] - lower_bound[1] >= 0)
        idz = (points[:, 2] - upper_bound[2] <= 0) * (points[:, 2] - lower_bound[2] >= 0)

        points = points[idx*idy*idz]

    if visualization:
        # for visualization
        axis = trimesh.creation.axis(axis_length=1)
        pcd = trimesh.PointCloud(points)
        (axis + mesh_ori).show()
        (trimesh.Scene(pcd) + axis).show()

    return points

