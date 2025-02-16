import math
from pathlib import Path
import json

import numpy as np
import torch

from diff_physics_engine.actuation.pid import PID


def process_real_data(data_path,
                      config_path,
                      raw_data_path,
                      output_path,
                      dt=0.01,
                      smoothing_type=None,
                      smooth_params=None):
    data_arrays = load_endcap_data(data_path)
    ext_mat = get_camera_ext_mat(config_path)
    # ext_mat = np.eye(4)
    raw_json_data = load_raw_json_data(raw_data_path)

    endcaps_data_world = cam_frame_to_world_frame(data_arrays,
                                                  ext_mat)
    endcaps_data_world = shift_end_caps(endcaps_data_world)
    # smoothed_data = endcaps_data_world

    # dt, t_0, t_end = get_data_sample_rate(raw_json_data)
    interp_data, d_times = linear_interp(endcaps_data_world,
                                         raw_json_data,
                                         dt)
    smoothed_data = interp_data
    # if smoothing_type == 'window':
    #     smoothed_data = rolling_avg_smoothing(interp_data,
    #                                           **smooth_params)
    # elif smoothing_type == 'gaussian':
    #     smoothed_data = gaussian_smoothing(interp_data,
    #                                        **smooth_params)
    # else:
    #     smoothed_data = interp_data

    poses = compute_poses(smoothed_data)
    # pose_json = [{"time": dt * i, "pos": poses[i].tolist()}
    #              for i in range(poses.shape[0])]
    # time_json = [{'header':{'secs': t}} for t in d_times]
    # pose_json, target_gaits = combine_endcaps_and_json_data(poses, smoothed_data, time_json)
    pose_json, target_gaits = combine_endcaps_and_json_data(poses, endcaps_data_world, raw_json_data)

    with Path(output_path).open("w") as fp:
        json.dump(pose_json, fp)

    # with Path(output_path.parent, "target_gaits.json").open("w") as fp:
    #     json.dump(target_gaits, fp)


def load_endcap_data(npy_dir):
    npy_dir = Path(npy_dir)

    data = []
    np_array_paths = sorted([p for p in npy_dir.iterdir() if p.name[0].isdigit()],
                            key=lambda p: int(p.name.split("_")[0]))
    for p in np_array_paths:
        endcap_data = np.load(p.as_posix())
        data.append(endcap_data)

    return data


def combine_endcaps_and_json_data(poses, end_caps_data, json_data):
    data_jsons = []
    target_gaits = []
    for i in range(poses.shape[0]):
        gaits = [
            json_data[i]['motors']["0"]['target'],
            json_data[i]['motors']["1"]['target'],
            json_data[i]['motors']["2"]['target'],
            json_data[i]['motors']["3"]['target'],
            json_data[i]['motors']["4"]['target'],
            json_data[i]['motors']["5"]['target']
        ]
        controls = [
            json_data[i]['motors']["0"]['speed'] / json_data[i]['info']['max_speed'],
            json_data[i]['motors']["1"]['speed'] / json_data[i]['info']['max_speed'],
            json_data[i]['motors']["2"]['speed'] / json_data[i]['info']['max_speed'],
            json_data[i]['motors']["3"]['speed'] / json_data[i]['info']['max_speed'],
            json_data[i]['motors']["4"]['speed'] / json_data[i]['info']['max_speed'],
            json_data[i]['motors']["5"]['speed'] / json_data[i]['info']['max_speed']
        ]
        data = {
            "time": round(json_data[i]['header']['secs'], 3),
            'pos': poses[i].tolist(),
            'rod_01_end_pt1': end_caps_data[0][i].tolist(),
            'rod_01_end_pt2': end_caps_data[1][i].tolist(),
            'rod_23_end_pt1': end_caps_data[2][i].tolist(),
            'rod_23_end_pt2': end_caps_data[3][i].tolist(),
            'rod_45_end_pt1': end_caps_data[4][i].tolist(),
            'rod_45_end_pt2': end_caps_data[5][i].tolist(),
            'target_gaits': gaits,
            'controls': controls
        }
        data_jsons.append(data)

        if len(target_gaits) == 0 or gaits != target_gaits[-1]['target_gait']:
            target_gaits.append({
                'idx': i,
                'target_gait': gaits
            })

    # for i in range(poses.shape[0]):
    #     data = {
    #         "time": round(json_data[i]['header']['secs'], 3),
    #         'pos': poses[i].tolist(),
    #         'rod_01_end_pt1': end_caps_data[i][0].tolist(),
    #         'rod_01_end_pt2': end_caps_data[i][1].tolist(),
    #         'rod_23_end_pt1': end_caps_data[i][2].tolist(),
    #         'rod_23_end_pt2': end_caps_data[i][3].tolist(),
    #         'rod_45_end_pt1': end_caps_data[i][4].tolist(),
    #         'rod_45_end_pt2': end_caps_data[i][5].tolist(),
    #     }
    #     data_jsons.append(data)

    data_jsons = sorted(data_jsons, key=lambda d: d['time'])

    return data_jsons, target_gaits


def shift_end_caps(end_caps_data, end_cap_radius=0.0175, scale=10):
    end_cap_radius = scale * end_cap_radius

    e0 = end_caps_data[0] * scale
    e1 = end_caps_data[1] * scale
    e2 = end_caps_data[2] * scale
    e3 = end_caps_data[3] * scale
    e4 = end_caps_data[4] * scale
    e5 = end_caps_data[5] * scale

    prin01 = (e1 - e0) / np.linalg.norm(e1 - e0, axis=1)[:, None]
    prin23 = (e3 - e2) / np.linalg.norm(e3 - e2, axis=1)[:, None]
    prin45 = (e5 - e4) / np.linalg.norm(e5 - e4, axis=1)[:, None]

    e0 += end_cap_radius * prin01
    e1 -= end_cap_radius * prin01
    e2 += end_cap_radius * prin23
    e3 -= end_cap_radius * prin23
    e4 += end_cap_radius * prin45
    e5 -= end_cap_radius * prin45

    com = (e0[0] + e1[0] + e2[0] + e3[0] + e4[0] + e5[0]) / 6.0
    min_z = np.min([e0[0, 2], e1[0, 2], e2[0, 2], e3[0, 2], e4[0, 2], e5[0, 2]])
    shift = np.array([com[0], com[1], -end_cap_radius + min_z])

    e0 -= shift
    e1 -= shift
    e2 -= shift
    e3 -= shift
    e4 -= shift
    e5 -= shift

    return [e0, e1, e2, e3, e4, e5]


def load_raw_json_data(raw_data_dir):
    raw_data_dir = Path(raw_data_dir)

    data = []
    data_paths = sorted(raw_data_dir.glob("*.json"),
                        key=lambda p: int(p.name.split(".")[0]))
    for p in data_paths:
        with p.open('r') as j:
            data_json = json.load(j)
        data.append(data_json)

    return data


def get_camera_ext_mat(config_path):
    with Path(config_path).open("r") as j:
        config = json.load(j)

    ext_mat = np.array(config['cam_extr'])

    return ext_mat


def cam_frame_to_world_frame(endcaps_data, ext_mat):
    ext_mat_inv = ext_mat
    # R = ext_mat[:3, :3]
    # t = ext_mat[:3, 3]
    # ext_mat_inv = np.linalg.inv(ext_mat)

    endcaps_data_world = []
    for endcap in endcaps_data:
        inhomo_endcap = np.hstack([endcap, np.ones((endcap.shape[0], 1))]).T
        endcap_world = np.matmul(ext_mat_inv, inhomo_endcap).T
        # print(endcap[0])
        # endcap_world = R @ endcap[0] + t
        # print(endcap_world)
        # print()
        endcaps_data_world.append(endcap_world[:, :3])

    return endcaps_data_world


def linear_interp_seq(endcaps_data, raw_json_data):
    # seq starts at 1
    raw_json_data_aug = [(i, data) for i, data in enumerate(raw_json_data)]
    raw_json_data_aug = sorted(raw_json_data_aug, key=lambda d: d[1]['header']['seq'])
    indices = [d[0] for d in raw_json_data_aug][:endcaps_data[0].shape[0] - 1]

    intrep_data = [[endcaps_data[i][0]] for i in range(len(endcaps_data))]

    for i in range(len(endcaps_data)):
        for j in indices:
            prev_seq = raw_json_data[j]['header']['seq']
            next_seq = raw_json_data[j + 1]['header']['seq']

            pt1, pt2 = endcaps_data[i][j:j + 2]
            pts = linear_two_pts(prev_seq, next_seq, pt1, pt2) if next_seq - prev_seq > 1 else [pt2]
            intrep_data[i].extend(pts)

    for i in range(len(intrep_data)):
        intrep_data[i] = np.vstack(intrep_data[i])

    return intrep_data


def linear_interp(endcaps_data, raw_json_data, dt):
    raw_json_data_aug = enumerate(raw_json_data)
    raw_json_data_aug = sorted(raw_json_data_aug, key=lambda d: d[1]['header']['secs'])

    start_time = raw_json_data_aug[0][1]['header']['secs']
    end_time = raw_json_data_aug[-1][1]['header']['secs']

    d_times = [dt * i for i in range(int((end_time - start_time) // dt) + 1)]
    interp_data = []
    curr = -1
    t1, t2 = 0, -1
    idx1, idx2 = 0, 0
    for t in d_times:
        while t > t2:
            curr += 1
            idx1 = raw_json_data_aug[curr][0]
            idx2 = raw_json_data_aug[curr + 1][0]
            t1 = raw_json_data_aug[curr][1]['header']['secs'] - start_time
            t2 = raw_json_data_aug[curr + 1][1]['header']['secs'] - start_time

        interp_endpts = [linear_two_pts(t1, t2, e[idx1], e[idx2], t) for e in endcaps_data]
        interp_data.append(interp_endpts)

    for i in range(len(interp_data)):
        interp_data[i] = np.vstack(interp_data[i])

    return interp_data, d_times


def linear_rounding(endcaps_data, raw_json_data, dt):
    raw_json_data_aug = sorted(raw_json_data, key=lambda d: d['header']['secs'])

    start_time = raw_json_data_aug[0][1]['header']['secs']

    rounded_data = [endcaps_data[0]]
    d_times = [0]
    for i in range(1, len(raw_json_data_aug)):
        t = raw_json_data_aug[i]['header']['secs'] - start_time
        t_1 = int(t // dt) * dt
        t_2 = t_1 + dt

        if abs(t - t_1) < abs(t - t_2):
            tt = raw_json_data_aug[i - 1]['header']['secs'] - start_time
            interp_endpts = [linear_two_pts(tt, t, e[i - 1], e[i], t_1) for e in endcaps_data]
            d_times.append(t_1)
        else:
            tt = raw_json_data_aug[i + 1]['header']['secs'] - start_time
            interp_endpts = [linear_two_pts(t, tt, e[i], e[i + 1], t_2) for e in endcaps_data]
            d_times.append(t_2)

        rounded_data.append(interp_endpts)

    for i in range(len(rounded_data)):
        rounded_data[i] = np.vstack(rounded_data[i])

    return rounded_data


def linear_two_pts_seq(t1, t2, pt1, pt2):
    delta_t = t2 - t1
    delta_pt = pt2 - pt1

    pts = [(delta_pt / delta_t) * t_i + pt2 - (delta_pt / delta_t) * t2
           for t_i in range(t1 + 1, t2)] + [pt2]

    return pts


def linear_two_pts(t1, t2, pt1, pt2, t):
    dt = t2 - t1
    w1 = np.abs(t2 - t) / dt
    w2 = 1 - w1

    pt = pt1 * w1 + pt2 * w2

    return pt


def get_data_sample_rate(raw_json_data):
    first_data_pt = min(raw_json_data, key=lambda d: d['header']['seq'])
    last_data_pt = max(raw_json_data, key=lambda d: d['header']['seq'])

    t_0 = first_data_pt['header']['secs']
    t_end = last_data_pt['header']['secs']

    num_samples = last_data_pt['header']['seq'] - first_data_pt['header']['seq']
    dt = (t_end - t_0) / num_samples

    return dt, t_0, t_end


def smoothing(data_arrays, filter):
    for i in range(len(data_arrays)):
        for j in range(data_arrays[0].shape[1]):
            data_arrays[i][:, j] = np.convolve(filter, data_arrays[i][:, j], 'same')

    return data_arrays


def rolling_avg_smoothing(data_arrays, window_size):
    window = np.full(window_size, 1.0 / window_size)
    data_arrays = smoothing(data_arrays, window)

    return data_arrays


def gaussian_smoothing(data_arrays, sigma, size):
    x = np.linspace(-int(size / 2), int(size / 2), size)
    filter = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-x ** 2 / sigma ** 2)

    data_arrays = smoothing(data_arrays, filter)

    return data_arrays


def compute_poses(data_arrays,
                  rod_length=0.36,
                  scaling_factor=55,
                  min_x=35.0,
                  min_y=0.0,
                  min_z=1.0):
    """
    Assume data arrays are arranged such that i (even) and i+1 (odd) is end pts of the same rod e.g (0, 1) or (2, 3)
    :param data_arrays:
    :param scaling_factor:
    :param min_z:
    :return:
    """
    poses = []

    # scaling_factor *= rod_length / np.linalg.norm(data_arrays[0][0] - data_arrays[1][0])
    #
    # x_offset = min([d[0][0] for d in data_arrays]) * scaling_factor - min_x
    # y_offset = min([d[0][1] for d in data_arrays]) * scaling_factor - min_y
    # z_offset = min([d[0][2] for d in data_arrays]) * scaling_factor - min_z
    #
    # offset = np.array([x_offset, y_offset, z_offset])
    # scaling_factor = 1.0
    # offset = np.array([0.0, 0.0, 0.0])
    #
    # for arr in data_arrays:
    #
    #     data1 = scaling_factor * arr[::2] - offset
    #     data2 = scaling_factor * arr[1::2] - offset
    #
    #     com = (data1 + data2) / 2.0
    #
    #     principal_axis = data2 - data1
    #     unit_prin_axis = principal_axis / np.linalg.norm(principal_axis, axis=1).reshape(-1, 1)
    #     quat = np.hstack([1 + unit_prin_axis[:, 2:3],
    #                       -unit_prin_axis[:, 1:2],
    #                       unit_prin_axis[:, 0:1],
    #                       np.zeros((unit_prin_axis.shape[0], 1))])
    #     quat /= np.linalg.norm(quat, axis=1).reshape(-1, 1)
    #
    #     pose = np.hstack([com, quat])
    #     poses.append(pose.reshape(1, -1))
    #
    # poses_array = np.vstack(poses)

    scaling_factor = 1.0
    offset = np.array([0.0, 0.0, 0.0])

    for i in range(0, len(data_arrays), 2):
        data1 = scaling_factor * data_arrays[i] - offset
        data2 = scaling_factor * data_arrays[i + 1] - offset

        print(i, data1[0])
        print(i + 1, data2[0])
        # print(np.linalg.norm(data1[0] - data2[0]) / scaling_factor)

        com = (data1 + data2) / 2.0

        principal_axis = data2 - data1
        unit_prin_axis = principal_axis / np.linalg.norm(principal_axis, axis=1).reshape(-1, 1)
        quat = np.hstack([1 + unit_prin_axis[:, 2:3],
                          -unit_prin_axis[:, 1:2],
                          unit_prin_axis[:, 0:1],
                          np.zeros((unit_prin_axis.shape[0], 1))])
        quat /= np.linalg.norm(quat, axis=1).reshape(-1, 1)

        pose = np.hstack([com, quat])
        poses.append(pose)

    poses_array = np.hstack(poses)

    return poses_array


def compute_controls(raw_data_path,
                     processed_json_path,
                     output_path,
                     num_actuators,
                     k_p=6.0,
                     k_d=0.5,
                     k_i=0.01,
                     min_length=100,
                     range_=100,
                     tol=0.15):
    pids = [PID(k_p, k_i, k_d, min_length, range_, tol) for _ in range(num_actuators)]

    with Path(processed_json_path).open("r") as fp:
        processed_json = json.load(fp)

    data = []
    for path in Path(raw_data_path).iterdir():
        if path.suffix == '.json':
            with path.open('r') as fp:
                data.append(json.load(fp))

    data = sorted(data, key=lambda d: d['header']['secs'])
    t0 = data[0]['header']['secs']

    # controls = []
    prev_target_gait = None
    for p, d in zip(processed_json, data):
        assert p['time'] == d['header']['secs']

        target_gait = [d['motors'][str(i)]['target'] for i in range(num_actuators)]
        if target_gait != prev_target_gait:
            print("\ngait change")
            prev_target_gait = target_gait
            for pid in pids:
                pid.reset()

        # ctrls = {
        #     f"spring_{i}": pids[i].compute_ctrl_target_gait(
        #         torch.tensor(d['motors'][str(i)]['position']),
        #         pids[i].min_length,
        #         pids[i].RANGE,
        #         torch.tensor(d['motors'][str(i)]['target'])
        #     ).item()
        #     for i in range(num_actuators)
        # }
        ctrls = {}
        for i in range(num_actuators):
            ctrl = pids[i].compute_ctrl_target_gait(
                torch.tensor(d['motors'][str(i)]['position']),
                pids[i].min_length,
                pids[i].RANGE,
                torch.tensor(d['motors'][str(i)]['target'])
            ).item()
            ctrls[f'spring_{i}'] = ctrl
        print(list(ctrls.values()))
        # controls.append(ctrls)
        p['controls'] = ctrls
        p['time'] -= t0

    with Path(output_path).open('w') as fp:
        json.dump(processed_json, fp)


def factor_graph_process(path, processed_data_path):
    def norm(lis1, lis2):
        return math.sqrt(sum([(l2-l1) ** 2 for l1, l2 in zip(lis1, lis2)]))

    processed_data_path = Path(processed_data_path)

    rod_suffixes = ["01_1", "01_2", "23_1", "23_2", "45_1", "45_2"]
    with Path(processed_data_path).open('r') as fp:
        processed_data = json.load(fp)

    end_pts = []
    for n in rod_suffixes:
        with Path(path, f'estimation_{n}.txt').open('r') as fp:
            lines = fp.readlines()
            lines = [l.split()[5:8] for l in lines]
        end_pts.append(lines)

    end_pts = list(zip(*end_pts))
    for i in range(1, len(processed_data)):
        rod_01_endpt1 = [float(x) for x in end_pts[i - 1][0]]
        rod_01_endpt2 = [float(x) for x in end_pts[i - 1][1]]
        rod_23_endpt1 = [float(x) for x in end_pts[i - 1][2]]
        rod_23_endpt2 = [float(x) for x in end_pts[i - 1][3]]
        rod_45_endpt1 = [float(x) for x in end_pts[i - 1][4]]
        rod_45_endpt2 = [float(x) for x in end_pts[i - 1][5]]

        pos_01 = [(rod_01_endpt2[j] + rod_01_endpt1[j]) / 2 for j in range(3)]
        pos_23 = [(rod_23_endpt2[j] + rod_23_endpt1[j]) / 2 for j in range(3)]
        pos_45 = [(rod_45_endpt2[j] + rod_45_endpt1[j]) / 2 for j in range(3)]

        prin_01 = [(rod_01_endpt2[j] - rod_01_endpt1[j]) / norm(rod_01_endpt1, rod_01_endpt2) for j in range(3)]
        prin_23 = [(rod_23_endpt2[j] - rod_23_endpt1[j]) / norm(rod_23_endpt1, rod_23_endpt2) for j in range(3)]
        prin_45 = [(rod_45_endpt2[j] - rod_45_endpt1[j]) / norm(rod_45_endpt1, rod_45_endpt2) for j in range(3)]

        half_len = 3.25 / 2

        new_rod_01_endpt1 = [pos_01[j] - half_len * prin_01[j] for j in range(3)]
        new_rod_01_endpt2 = [pos_01[j] + half_len * prin_01[j] for j in range(3)]
        new_rod_23_endpt1 = [pos_23[j] - half_len * prin_23[j] for j in range(3)]
        new_rod_23_endpt2 = [pos_23[j] + half_len * prin_23[j] for j in range(3)]
        new_rod_45_endpt1 = [pos_45[j] - half_len * prin_45[j] for j in range(3)]
        new_rod_45_endpt2 = [pos_45[j] - half_len * prin_45[j] for j in range(3)]

        processed_data[i]['pos'] = pos_01 + pos_23 + pos_45

        processed_data[i]['rod_01_end_pt1'] = new_rod_01_endpt1
        processed_data[i]['rod_01_end_pt2'] = new_rod_01_endpt2
        processed_data[i]['rod_23_end_pt1'] = new_rod_23_endpt1
        processed_data[i]['rod_23_end_pt2'] = new_rod_23_endpt2
        processed_data[i]['rod_45_end_pt1'] = new_rod_45_endpt1
        processed_data[i]['rod_45_end_pt2'] = new_rod_45_endpt2

    with Path(processed_data_path.parent, "smoothed_processed_data.json").open("w") as fp:
        json.dump(processed_data, fp)


if __name__ == '__main__':
    dataset_name = "R2S2Rrolling_1"
    base_path = Path("/home/nelsonchen/research/tensegrity/data_sets/"
                     "tensegrity_real_datasets/RSS_demo_old_platform/R2S2R_rolling/")
    data_path = Path(base_path, 'poses-proposed')
    config_path = Path(base_path, 'config.json')
    raw_data_path = Path(base_path, 'data')
    output_path = Path(base_path, "processed_data_0.01v2.json")

    # smoothing_type, smooth_params = 'gaussian', {'sigma': 5, 'size': 10}
    #
    process_real_data(data_path, config_path, raw_data_path, output_path, 0.01, None, None)
    # compute_controls(raw_data_path, data_path, output_path, 6)
    # for path in base_path.iterdir():
    #     if not path.name.startswith("R2S2R"):
    #         continue
    #     processed_data_path = Path(path, "processed_data.json")
    #     factor_graph_process(path, processed_data_path)
