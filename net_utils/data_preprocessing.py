import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

def correct_bad_chair(phases_dict):
    """
    bad chair b'648972_chair_poliform_harmony' is not completely removed in current data
    try to fix it here
    """
    if len(phases_dict["instance_idx"]) - 1 != phases_dict["n_objects"]:
        # remove the empty object
        obj_points = []
        n_empty_obj = 0
        opt_ids = []
        for opt_id, opts in enumerate(phases_dict["obj_points"]):
            if not opts.shape[0] == 0:
                obj_points.append(opts)
                opt_ids.append(opt_id)
            else:
                n_empty_obj += 1
        phases_dict["obj_points"] = obj_points
        phases_dict["before_fix_n_objects"] = phases_dict["n_objects"]
        phases_dict["n_objects"] = len(obj_points)
        phases_dict["bad_lamp"] = True
        phases_dict["ok_obj_id"] = opt_ids
        assert(len(phases_dict["instance_idx"]) - 1 == phases_dict["n_objects"])
        return True
    else:
        # there is empty mesh in drop

        if "drop" in phases_dict["trial_dir"] and "train/50" in phases_dict["trial_dir"]:

            n_empty_obj = 0
            opt_ids = []
            for opt_id, opts in enumerate(phases_dict["obj_points"]):
                if not opts.shape[0] == 0:
                    opt_ids.append(opt_id)
                else:
                    n_empty_obj += 1
            if n_empty_obj > 0:


                list_items = ["root_des_radius", "root_num", "clusters", "instance", "material", "obj_points"]
                for item in list_items:
                    phases_dict[item] = [phases_dict[item][a] for a in opt_ids]
                new_instance_idx = [0]
                for obj_pts in phases_dict["obj_points"]:
                    new_instance_idx.append(new_instance_idx[-1] + obj_pts.shape[0])

                phases_dict["instance_idx"] = new_instance_idx
                phases_dict["n_objects"] = len(phases_dict["obj_points"])
                phases_dict["ok_obj_id"] = opt_ids

                assert(phases_dict["n_particles"] == new_instance_idx[-1])
                assert(len(phases_dict["instance_idx"]) - 1 == phases_dict["n_objects"])
                assert(len(phases_dict["root_num"]) == phases_dict["n_objects"])
                return True
            else:
                return False


        return False



def load_data_dominoes(data_names, path, static_data_info, load_data_names=["obj_positions", "obj_rotations"]):
    """
    static_data_info: num_objects, object point cloud


    """
    if not isinstance(path, list):
        paths = [path]
        one_item = True
    else:
        paths = path
        one_item = False

    n_objects = static_data_info["n_objects"]
    obj_points = static_data_info["obj_points"]


    flex_engine = False
    if "clothSagging" in static_data_info["trial_dir"] or "clothsagging" in static_data_info["trial_dir"]:
        flex_engine = True
        load_data_names = ['particle_positions', "particle_velocities"]

    if "is_subsample" in static_data_info:
        is_subsample = static_data_info["is_subsample"]
        obj_subsample_idx = static_data_info["obj_subsample_idx"]
        instance_idx_old = static_data_info["instance_idx_before_subsample"]

    else:
        is_subsample = False
    #import ipdb; ipdb.set_trace()
    multiple_data = []
    for path in paths:
        hf = h5py.File(path, 'r')
        data_raw = dict()

        for i, data_name in enumerate(load_data_names):
            d = np.array(hf.get(data_name))
            data_raw[data_name] = d
        hf.close()
        data = []

        for data_name in data_names:
            if data_name == "positions":

                if flex_engine:

                    particle_positions = data_raw["particle_positions"]


                    particle_positions *= 0.05/0.035
                    if is_subsample:
                        n_particles = static_data_info["n_particles"]
                        positions = []

                        for obj_id in range(n_objects):
                            st, ed = instance_idx_old[obj_id], instance_idx_old[obj_id + 1]
                            pos = particle_positions[st:ed, :]
                            positions.append(pos[obj_subsample_idx[obj_id]])


                        particle_positions = np.concatenate(positions, axis=0)

                        # add scale correction
                        assert(n_particles == particle_positions.shape[0])

                    data.append(particle_positions)
                else:
                    transformed_obj_pts = []
                    # object point cloud and rotation/positions to compute particles
                    obj_rotations = data_raw["obj_rotations"]
                    obj_positions = data_raw["obj_positions"]

                    if "ok_obj_id" in static_data_info:

                        if "bad_lamp" in static_data_info and obj_positions.shape[0] != static_data_info["before_fix_n_objects"]:
                            print("good trial with bad lamp", static_data_info["trial_dir"])
                            import ipdb; ipdb.set_trace()

                        # print(obj_rotations.shape)
                        # print(obj_positions.shape)
                        # print(len(obj_points))
                        # print("ok_obj_id", static_data_info["ok_obj_id"])

                        #if (len(obj_points) < obj_rotations.shape[0]):
                        #    import ipdb; ipdb.set_trace()
                        for idx, obj_id in enumerate(static_data_info["ok_obj_id"]):
                            rot = R.from_quat(obj_rotations[obj_id]).as_matrix()
                            trans = obj_positions[obj_id]
                            transformed_pts = np.matmul(rot, obj_points[idx].T).T + np.expand_dims(trans, axis=0)
                            transformed_obj_pts.append(transformed_pts)

                    else:
                        #print("#######", n_objects, obj_rotations.shape, obj_positions.shape)
                        if not n_objects == obj_rotations.shape[0]:
                            import ipdb; ipdb.set_trace()
                        assert(n_objects == obj_rotations.shape[0])
                        for obj_id in range(n_objects):
                            rot = R.from_quat(obj_rotations[obj_id]).as_matrix()
                            trans = obj_positions[obj_id]
                            transformed_pts = np.matmul(rot, obj_points[obj_id].T).T + np.expand_dims(trans, axis=0)
                            transformed_obj_pts.append(transformed_pts)
                    positions = np.concatenate(transformed_obj_pts, axis=0)
                    # if is_subsample:
                    #     positions_tmp = []
                    #     for obj_id in range(n_objects):
                    #         st, ed = instance_idx_old[obj_id], instance_idx_old[obj_id + 1]

                    #         pos = positions[st:ed, :]

                    #         positions_tmp.append(pos[obj_subsample_idx[obj_id], :])
                    #     positions = np.concatenate(positions_tmp, axis=0)
                    #     print("2", positions.shape)
                    #import ipdb; ipdb.set_trace()
                    data.append(positions)

            elif data_name in ["obj_positions", "obj_rotations", "world_T_cam", "pix_T_cam", "image", "depth", "id_map"]:
                data.append(data_raw[data_name])
            elif data_name == "velocities":
                # should compute later on
                data.append(None)

            else:
                print(data_name)
                raise (ValueError, f"{data_name}" + " not supported")
        multiple_data.append(data)

    if one_item:
        return multiple_data[0]

    if mode == "avg":
        nitems = len(multiple_data[0])
        outputs = []
        for itemid in range(nitems):
           outputs.append(np.mean(np.stack([data[itemid] for data in multiple_data], 0), 0))
        return outputs
    else:
        raise ValueError


    return data