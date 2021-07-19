# Trainer for Total3D.
# author: ynie
# date: Feb, 2020
from models.training import BaseTrainer
import torch
import numpy as np
import os
from net_utils import visualization as vis

class Trainer(BaseTrainer):
    '''
    Trainer object for total3d.
    '''

    def eval_step(self, data):
        '''
        performs a step in evaluation
        :param data (dict): data dictionary
        :return:
        '''
        loss = self.compute_loss(data)
        loss['total'] = loss['total'].item()
        return loss

    def visualize_step(self, epoch, phase, iter, data):
        ''' Performs a visualization step.
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        with torch.no_grad():
            '''network forwarding'''
            est_data = self.net({**data, 'export_shape':True})
            voxels_out, proposal_to_gt_box_w_cls_list = est_data[2:4]

        if proposal_to_gt_box_w_cls_list is None:
            return

        sample_ids = np.random.choice(voxels_out.shape[0], 3, replace=False) if voxels_out.shape[0]>=3 else range(voxels_out.shape[0])
        n_shapes_per_batch = self.cfg.config['data']['completion_limit_in_train']
        for idx, i in enumerate(sample_ids):
            voxel_path = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s_%03d_pred.png' % (epoch, phase, iter, idx))
            vis.visualize_voxels(voxels_out[i].cpu().numpy(), voxel_path)

            batch_index = i // n_shapes_per_batch
            in_batch_id = i % n_shapes_per_batch
            box_id = proposal_to_gt_box_w_cls_list[batch_index][in_batch_id][1].item()
            cls_id = proposal_to_gt_box_w_cls_list[batch_index][in_batch_id][2].item()

            voxels_gt = data['object_voxels'][batch_index][box_id].cpu().numpy()
            voxel_path = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s_%03d_gt_cls%d.png' % (epoch, phase, iter, idx, cls_id))
            vis.visualize_voxels(voxels_gt, voxel_path)

    def to_device(self, data):
        device = self.device
        for key in data:
            if key not in ['object_voxels', 'shapenet_catids', 'shapenet_ids']:
                data[key] = data[key].to(device)
        return data

    def compute_loss(self, data):
        '''
        compute the overall loss.
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)
        #self.writer.add_iamge

        if self.visualization:
            import trimesh
            point_cloud = data["point_clouds"][0].cpu().numpy()
            axis = trimesh.creation.axis(axis_length=1)
            world_colors = np.repeat(np.array([[0,0,0,0.5]]), point_cloud.shape[0], axis=0)
            #for object_id in range(nobjects):
            #    world_colors[point_instance_labels == object_id] = self.color_base[object_id][np.newaxis, :]

            box_label_mask = data["box_label_mask"][0].cpu().numpy()
            center_label = data["center_label"][0].cpu().numpy()
            size_residual_label = data["size_residual_label"][0].cpu().numpy()

            meshes = []
            for obj_id in range(box_label_mask.shape[0]):
                if box_label_mask[obj_id] == 1:
                    size = size_residual_label[obj_id] + np.array([0.4, 0.4, 0.4])
                    obj_center = center_label[obj_id]
                    mat = np.eye(4)
                    mat[:3, 3] = obj_center
                    mesh = trimesh.creation.box(extents=size, transform=mat)
                    mesh.visual.face_colors = [255, 100, 200, 100]
                    meshes.append(mesh)

                    #center_mesh = trimesh.creation.box(extents=np.array([0.03, 0.03, 0.03]), transform=mat)
                    #center_mesh.visual.face_colors = [255, 0, 0, 255]
                    #meshes.append(center_mesh)

            #for object_id in range(nobjects):
            pcds = trimesh.PointCloud(point_cloud[:,:3], world_colors)
            merged_mesh = sum(meshes)
            (trimesh.Scene(pcds) + axis + merged_mesh).show()

        # box_label_mask = data["box_label_mask"][0].cpu().numpy()
        # point_cloud = data["point_clouds"][:1, :, :3].cpu().numpy()
        # center_label = data["center_label"][0].cpu().numpy()
        # size_residual_label = data["size_residual_label"][0].cpu().numpy()
        # grid = np.array([[-1, -1, -1],
        #                  [-1, -1,  1],
        #                  [-1,  1, -1],
        #                  [-1,  1,  1],
        #                  [ 1, -1, -1],
        #                  [ 1, -1,  1],
        #                  [ 1,  1, -1],
        #                  [ 1,  1,  1]])
        # face_ids = np.array([[1,  2,  3],
        #                   [2,  3,  7],
        #                   [1,  2,  5],
        #                   [2,  5,  6],
        #                   [3,  7,  4],
        #                   [7,  8,  4]])

        # obj_centers = []
        # colors = []
        # sizes = []
        # faces = []
        # start_id  = point_cloud.shape[1]
        # for obj_id in range(box_label_mask.shape[0]):
        #     if box_label_mask[obj_id] == 1:
        #         size = size_residual_label[obj_id] + np.array([0.4, 0.4, 0.4])
        #         obj_center = center_label[obj_id]
        #         corners = grid * 0.5 * size[np.newaxis, :] + obj_center[np.newaxis, :]

        #         faces.append(face_ids + start_id)
        #         start_id += 8


        #         obj_centers.append(corners[np.newaxis, :])
        #         colors.append(np.array([[[255, 0, 0]]]))

        # point_cloud_merged = np.concatenate([point_cloud]+obj_center, axis=1)
        # colors_merged = np.concatenate([np.zeros_like(point_cloud) ]+colors, axis=1)
        # faces_merged = np.concatenate(faces, axis=0)[np.newaxis, :, :]
        # self.writer.add_mesh("input point cloud", vertices=point_cloud_merged, colors=colors_merged, faces=faces)
        # self.writer.add_image("input rgb image", data["rgb_image"][0, :, :, :].permute(2, 0, 1))
        '''network forwarding'''
        est_data = self.net(data)


        #self.writer()

        import ipdb; ipdb.set_trace()

        '''computer losses'''
        loss = self.net.module.loss(est_data, data)
        return loss
