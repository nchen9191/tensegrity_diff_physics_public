import json
from copy import deepcopy
from pathlib import Path

import torch
import tqdm

from diff_physics_engine.simulators.tensegrity_simulator import TensegrityRobotSimulator as Simulator
from diff_physics_engine.state_objects.rods import RodState
from mujoco_visualizer_utils.mujoco_visualizer import MuJoCoVisualizer


def run_by_control(simulator, ctrls, dt, gt_data, start_state=None):
    with torch.no_grad():
        time = 0.0
        frames = []

        curr_state = start_state if start_state is not None else simulator.get_curr_state()
        num_bodies = int(curr_state.flatten().shape[0] / 13)
        pos = torch.hstack([curr_state.flatten()[i * 13: i * 13 + 7] for i in range(num_bodies)]).detach().numpy()

        frames.append({"time": time, "pos": pos.tolist()})

        for j, ctrl in enumerate(tqdm.tqdm(ctrls)):
            curr_state = simulator.step_w_controls(
                curr_state,
                dt,
                controls=ctrl
            )

            pos = torch.hstack([curr_state.flatten()[i * 13: i * 13 + 7] for i in range(num_bodies)]).detach().numpy()
            frames.append({"time": round(time, 3), "pos": pos.tolist()})

    return frames


def run_by_gait(simulator, gaits, dt, start_state=None):
    with torch.no_grad():
        time = 0.0
        frames = []

        state = start_state if start_state is not None else simulator.get_curr_state()
        num_bodies = int(state.flatten().shape[0] / 13)
        pos = torch.hstack([state.flatten()[i * 13: i * 13 + 7] for i in range(num_bodies)]).detach().numpy()

        frames.append({"time": time, "pos": pos.tolist()})

        kf_pred_states = []

        gait_states = [state]
        for j, d in enumerate(tqdm.tqdm(gaits)):
            gait = {f"spring_{i}": v for i, v in enumerate(d['target_gait'])}

            gait_states, t = simulator.run_target_gait(dt, gait_states[-1], gait)
            kf_pred_states.append(gait_states[-1])

            for state in gait_states:
                time += dt.item()
                pos = torch.hstack([state.flatten()[i * 13: i * 13 + 7]
                                    for i in range(num_bodies)]).detach().numpy()
                frames.append({"time": round(time, 3), "pos": pos.tolist()})

    return frames, kf_pred_states


def get_endpts(gt_data):
    with torch.no_grad():
        dtype = torch.float64
        # dtype = self.simulator.sys_precision
        end_pts = [
            [
                torch.tensor(d['rod_01_end_pt1'], dtype=dtype).reshape(1, 3, 1),
                torch.tensor(d['rod_01_end_pt2'], dtype=dtype).reshape(1, 3, 1),
                torch.tensor(d['rod_23_end_pt1'], dtype=dtype).reshape(1, 3, 1),
                torch.tensor(d['rod_23_end_pt2'], dtype=dtype).reshape(1, 3, 1),
                torch.tensor(d['rod_45_end_pt1'], dtype=dtype).reshape(1, 3, 1),
                torch.tensor(d['rod_45_end_pt2'], dtype=dtype).reshape(1, 3, 1)
            ] for d in gt_data
        ]
    return end_pts


def kf_loss(sim, kf_pred_states, gt_data, target_gaits):
    gt_end_pts = get_endpts(gt_data)
    kf_gt_end_pts = torch.vstack(
        [torch.hstack(gt_end_pts[v['idx']])
         for v in target_gaits[1:]]
        + [torch.hstack(gt_end_pts[-1])]
    )
    kf_pred_states = torch.vstack(kf_pred_states)
    kf_pred_end_pts = batch_compute_end_pts(sim, kf_pred_states)

    loss = torch.mean((kf_pred_end_pts - kf_gt_end_pts) ** 2)
    return loss


def batch_compute_end_pts(sim, batch_state: torch.Tensor) -> torch.Tensor:
    """
    Compute end pts for entire batch

    :param batch_state: batch of states
    :return: batch of endpts
    """
    end_pts = []
    for i, rod in enumerate(sim.rigid_bodies.values()):
        state = batch_state[:, i * 13: i * 13 + 7]
        principal_axis = RodState.compute_principal_axis(state[:, 3:7])
        end_pts.extend(RodState.compute_end_pts_from_state(state, principal_axis, rod.length))

    return torch.concat(end_pts, dim=1)


def run_traj():
    # Set paths
    base_path = Path("../../tensegrity/data_sets/tensegrity_real_datasets/RSS_demo_old_platform/RSS_rolling/")
    config_file_path = "diff_physics_engine/simulators/configs/3_bar_tensegrity_upscaled_v1.json"

    # Get config
    with Path(config_file_path).open("r") as j:
        config = json.load(j)

    with Path(base_path, "processed_data.json").open('r') as fp:
        gt_data = json.load(fp)

    with Path(base_path, "target_gaits.json").open('r') as fp:
        gaits = json.load(fp)

    with Path(base_path, "processed_data_0.01v2.json").open('r') as fp:
        vis_gt_data = json.load(fp)

    # Set contact parameters

    # Will's sys id
    # contact_params = {
    #     "restitution": -0.16413898179172115,
    #     "baumgarte": 0.38783653279732216,
    #     "friction": 0.7593319176982094,
    #     "friction_damping": 0.8689685753695016
    # }

    # Patrick's sys id
    contact_params = {
        'restitution': -0.036485324706179156,
        'baumgarte': 0.03770246611256817,
        'friction': 0.4067791399229587,
        'friction_damping': 0.30583693613061996
    }

    # New open-source sys id
    # contact_params = {
    #     'restitution': -0.2395988899064052,
    #     'baumgarte': 0.3358065718363488,
    #     'friction': 1.1716651716431445,
    #     'friction_damping': 0.9846809241468255
    # }

    config['contact_params'] = contact_params

    # Instantiate simulator
    sim = Simulator.init_from_config_file(config)

    # Set pid params to get proper rolling
    if "ccw" in base_path.name:
        min_length = 1.0
        range_ = 1.0
        tol = 0.1
    elif "rolling" in base_path.name:
        min_length = 0.8
        range_ = 1.0
        tol = 0.1
    elif "cw" in base_path.name:
        min_length = 0.7
        range_ = 1.2
        tol = 0.1
    else:
        min_length = 1.0
        range_ = 1.0
        tol = 0.1

    # Initialize cable attach points by end pts
    end_pts = get_endpts(gt_data)
    start_endpts = [
        [end_pts[0][0], end_pts[0][1]],
        [end_pts[0][2], end_pts[0][3]],
        [end_pts[0][4], end_pts[0][5]],
    ]
    sim.init_by_endpts(start_endpts)

    # Set pid and cable params
    for pid in sim.pids.values():
        pid.min_length = torch.tensor([[[min_length]]], dtype=torch.float64)
        pid.RANGE = torch.tensor([[[range_]]], dtype=torch.float64)
        pid.tol = torch.tensor([[[tol]]], dtype=torch.float64)

    for i, cable in enumerate(sim.tensegrity_robot.actuated_cables.values()):
        act_length = torch.tensor(gt_data[0]['init_act_lens'][i], dtype=torch.float64)
        cable.actuation_length = act_length.reshape(1, 1, 1)
        cable.motor.speed = torch.tensor(0.8, dtype=torch.float64).reshape(1, 1, 1)
        cable.motor.motor_state.omega_t = torch.tensor(0., dtype=torch.float64).reshape(1, 1, 1)

    # Initialize starting state
    start_state = torch.tensor(gt_data[0]['stable_start'],
                               dtype=torch.float64).reshape(1, -1, 1)
    sim.update_state(start_state)

    dt = torch.tensor([[[0.01]]], dtype=torch.float64)
    frames, kf_pred_states = run_by_gait(sim, gaits[:], dt,  start_state,)

    # Visualize
    if len(vis_gt_data) >= len(frames):
        last_frame = deepcopy(frames[-1])
        for i, data in enumerate(vis_gt_data):
            t = data['time']
            # pos = data['pos']
            # quat = data['quat']
            # pose = [p for j in range(len(pos) // 3)
            #         for p in (pos[j * 3: (j + 1) * 3] + quat[j * 4: (j + 1) * 4])]
            pose = data['pos']

            if i < len(frames):
                frames[i]['pos'] += pose
            else:
                frames.append({
                    "time": t,
                    'pos': last_frame['pos'] + pose
                })
    else:
        last_frame = deepcopy(vis_gt_data[-1])
        for i in range(len(frames)):

            if i < len(vis_gt_data):
                # pos = vis_gt_data[i]['pos']
                # quat = vis_gt_data[i]['quat']
                # pose = [p for j in range(len(pos) // 3)
                #         for p in (pos[j * 3: (j + 1) * 3] + quat[j * 4: (j + 1) * 4])]
                pose = vis_gt_data[i]['pos']

                frames[i]['pos'] += pose
            else:
                # pos = last_frame['pos']
                # quat = last_frame['quat']
                # pose = [p for j in range(len(pos) // 3) for p in
                #         (pos[j * 3: (j + 1) * 3] + quat[j * 4: (j + 1) * 4])]
                pose = last_frame['pos']
                frames[i]['pos'] += pose

    xml_path = Path("mujoco_physics_engine/xml_models/3prism_real_upscaled_vis_w_gt.xml")
    visualizer = MuJoCoVisualizer()
    visualizer.set_xml_path(Path(xml_path))
    visualizer.data = frames
    visualizer.set_camera("camera")
    visualizer.visualize(Path(base_path, f"vid.mp4"), dt)