# Trainer for Total3D.
# author: ynie
# date: Feb, 2020
from models.training import BaseTrainer
import torch
import numpy as np
import os
from net_utils import visualization as vis
from net_utils.ap_helper import parse_predictions, parse_groundtruths

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
            end_points, _,voxels_out, proposal_to_gt_box_w_cls_list = est_data

        pts_world = data["point_clouds"][0, :, :3].cpu().numpy()
        box_label_mask = data["box_label_mask"][0].cpu().numpy()
        center_label = data["center_label"][0].cpu().numpy()
        size_residual_label = data["size_residual_label"][0].cpu().numpy()
        rgb_image = data["rgb_image"][0].cpu().numpy()

        mat = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0,-1, 0]])

        parsed_gts = parse_groundtruths(data, self.cfg.eval_config)
        bboxes_corner = parsed_gts["gt_corners_3d_upright_camera"][0]#world
        bboxes_corner = np.matmul(mat[np.newaxis, :, :], bboxes_corner.transpose(0, 2, 1)).transpose(0, 2, 1)

        bboxes_center = np.mean(bboxes_corner, axis=1)
        bboxes_extent = np.max(bboxes_corner, axis=1) - np.min(bboxes_corner, axis=1)
        bboxes_gt = np.concatenate([bboxes_center, bboxes_extent], axis=1)
        #bboxes_gt = np.concatenate([center_label, size_residual_label + 0.4], axis=1)
        vis.visualize_pointcloud_boxes(pts_world, bboxes_gt, box_label_mask=box_label_mask, out_file=os.path.join(self.cfg.config['log']['vis_path'], "%s_%s_%s_gt_detection.png" % (epoch, phase, iter)))


        eval_dict, parsed_predictions = parse_predictions(end_points, data, self.cfg.eval_config)
        bboxes_corner = parsed_predictions["pred_corners_3d_upright_camera"][0]
        bboxes_corner = np.matmul(mat[np.newaxis, :, :], bboxes_corner.transpose(0, 2, 1)).transpose(0, 2, 1)

        bboxes_center = np.mean(bboxes_corner, axis=1)
        bboxes_extent = np.max(bboxes_corner, axis=1) - np.min(bboxes_corner, axis=1)
        bboxes_pred = np.concatenate([bboxes_center, bboxes_extent], axis=1)

        vis.visualize_pointcloud_boxes(pts_world, bboxes_pred, box_label_mask=eval_dict["pred_mask"][0] * parsed_predictions["obj_prob"][0], out_file=os.path.join(self.cfg.config['log']['vis_path'], "%s_%s_%s_pred_detection.png" % (epoch, phase, iter)))

        parsed_gts = parse_groundtruths(data, self.cfg.eval_config)



        if proposal_to_gt_box_w_cls_list is None: # when doing object detection only
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
        self.visualization = False
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
            import imageio
            imageio.imwrite("rgb.png", data["rgb_image"][0].cpu().numpy())
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
                    mesh.visual.face_colors = [255, 100, 200, 180]
                    meshes.append(mesh)

                    #center_mesh = trimesh.creation.box(extents=np.array([0.03, 0.03, 0.03]), transform=mat)
                    #center_mesh.visual.face_colors = [255, 0, 0, 255]
                    #meshes.append(center_mesh)

            #for object_id in range(nobjects):
            pcds = trimesh.PointCloud(point_cloud[:,:3], world_colors)
            merged_mesh = sum(meshes)
            (trimesh.Scene(pcds) + axis + merged_mesh).show()

        '''network forwarding'''
        est_data = self.net(data)


        #self.writer()

        '''computer losses'''
        loss = self.net.module.loss(est_data, data)


        return loss
