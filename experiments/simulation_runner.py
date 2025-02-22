import json
import time
from copy import deepcopy
from pathlib import Path
import multiprocessing

import torch
import tqdm

from diff_physics_engine.simulators.tensegrity_simulator import TensegrityRobotSimulator
from diff_physics_engine.state_objects.rods import RodState
from mujoco_visualizer_utils.mujoco_visualizer import MuJoCoVisualizer
from utilities import torch_quaternion


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


def detect_ground_endcaps(end_pts):
    aug_end_pts = [[(i, end_pts[i]), (i + 1, end_pts[i + 1])] for i in range(0, len(end_pts), 2)]
    aug_end_pts = [min(e, key=lambda x: x[1].flatten()[2].item()) for e in aug_end_pts]

    ground_encaps = tuple([a[0] for a in aug_end_pts])

    return ground_encaps


def init_sim(cfg, start_end_pts=None, start_state=None, rest_lengths=None, motor_speeds=None):
    assert start_end_pts is not None or start_state is not None

    sim = TensegrityRobotSimulator.init_from_config_file(cfg)

    if rest_lengths:
        for i, c in enumerate(sim.tensegrity_robot.actuated_cables.values()):
            act_len = c._rest_length - rest_lengths[i]
            c.actuation_length = act_len
            c.motor.speed = torch.tensor([[[0.99]]], dtype=torch.float64)

    if motor_speeds:
        for i, c in enumerate(sim.tensegrity_robot.actuated_cables.values()):
            c.motor.motor_state.omega_t = torch.tensor([[[motor_speeds[i]]]], dtype=torch.float64)

    if start_end_pts is None:
        sim.update_state(start_state)
        start_end_pts = [e for r in sim.tensegrity_robot.rods.values() for e in r.end_pts]
    elif start_state is None:
        end_pts = [e for r in start_end_pts for e in r]
        end_pts = torch.vstack(end_pts).reshape(-1, 6, 1)
        pos = (end_pts[:, 3:] + end_pts[:, :3]) / 2.

        prin = end_pts[:, 3:] - end_pts[:, :3]
        prin = prin / prin.norm(dim=1, keepdim=True)
        quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin)

        vel = torch.zeros_like(pos)
        start_state = torch.hstack([pos, quat, vel, vel]).reshape(1, -1, 1)
        sim.update_state(start_state)

    sim.init_by_endpts(start_end_pts)
    return sim, start_state


def run_primitive(sim,
                  curr_state,
                  init_rest_lengths,
                  init_motor_speeds,
                  dt,
                  prim_type,
                  left_range,
                  right_range,
                  max_steps=500):
    symmetry_mapping = {(0, 2, 5): [0, 1, 2, 3, 4, 5], (0, 3, 5): [0, 1, 2, 3, 4, 5],
                        (1, 2, 4): [1, 2, 0, 4, 5, 3], (1, 2, 5): [1, 2, 0, 4, 5, 3],
                        (0, 3, 4): [2, 0, 1, 5, 3, 4], (1, 3, 4): [2, 0, 1, 5, 3, 4]}

    if "ccw" == prim_type:
        min_length = 1.0
        range_ = 1.0
        tol = 0.1
    elif "roll" == prim_type:
        min_length = 0.8
        range_ = 1.0
        tol = 0.1
    elif "cw" == prim_type:
        min_length = 0.7
        range_ = 1.2
        tol = 0.1
    else:
        min_length = 1.0
        range_ = 1.0
        tol = 0.1

    prim_gaits = {
        'ccw': [[1, 1, 1, 0, 1, 1], [1, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        'cw': [[0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0.5, 0, 1, 1], [1, 1, 1, 1, 1, 1]],
        'roll': [[1, 1, 0.1, 1, 1, 0.1], [0, 1, 1, 0, 1, 0.1], [1, 1, 1, 1, 1, 1]]
    }

    for i, pid in enumerate(sim.pids.values()):
        pid.min_length = torch.tensor(min_length, dtype=torch.float64).reshape(1, 1, 1)

        if prim_type == 'roll':
            if i % 2 == 0:
                range_ = left_range
            else:
                range_ = right_range

        pid.RANGE = torch.tensor(range_, dtype=torch.float64).reshape(1, 1, 1)
        pid.tol = torch.tensor(tol, dtype=torch.float64).reshape(1, 1, 1)

    for i, c in enumerate(sim.tensegrity_robot.actuated_cables.values()):
        act_len = c._rest_length - init_rest_lengths[i]
        c.actuation_length = act_len
        c.motor.motor_state.omega_t = torch.tensor([[[init_motor_speeds[i]]]], dtype=torch.float64)

    sim.update_state(curr_state)
    end_pts = [e for r in sim.tensegrity_robot.rods.values() for e in r.end_pts]
    ground_endcaps = detect_ground_endcaps(end_pts)
    order = symmetry_mapping[ground_endcaps]

    all_controls = []
    all_states, all_rest_lens, all_motor_speeds = [], [], []
    for gait in tqdm.tqdm(prim_gaits[prim_type]):
        sim.reset_pids()
        sim.update_state(curr_state)
        target_gait = [gait[i] for i in order]
        target_gait_dict = {f"spring_{i}": v for i, v in enumerate(target_gait)}
        gait_states, gait_rest_lengths, gait_motor_speeds, gait_controls = (
            sim.run_target_gait(dt, curr_state, target_gait_dict))
        curr_state = gait_states[-1]

        all_controls.extend(gait_controls)
        all_states.extend(gait_states)
        all_rest_lens.extend(gait_rest_lengths)
        all_motor_speeds.extend(gait_motor_speeds)

    return all_states, all_rest_lens, all_motor_speeds, all_controls


# def multi_process_run_prim(sim, start_states, dt, prim_list):
#     num_threads = min(len(start_states), 16)
#     sim_copies = [deepcopy(sim) for
#     processes = [
#         multiprocessing.Process(
#             target=run_primitive(),
#             args=(start_state[i * nsamples: (i + 1) * nsamples],
#                   actions[i * nsamples: (i + 1) * nsamples],
#                   envs[i], i, queues[i])
#         ) for i in range(num_threads)
#     ]
#     ]


def visualize(all_states, dt, output_path):
    frames = [
        {
            'time': dt * i,
            'pos': all_states[i].reshape(-1, 13)[:, :7].flatten().numpy()
        } for i in range(len(all_states))
    ]
    xml_path = Path("mujoco_physics_engine/xml_models/3prism_real_upscaled_vis.xml")
    visualizer = MuJoCoVisualizer()
    visualizer.set_xml_path(Path(xml_path))
    visualizer.data = frames
    visualizer.set_camera("camera")
    visualizer.visualize(Path(output_path), dt)


def run_traj():
    # Set paths
    config_file_path = "diff_physics_engine/simulators/configs/3_bar_tensegrity_upscaled_v2.json"
    #
    # Get config
    with Path(config_file_path).open("r") as j:
        config = json.load(j)
    #
    # with Path(base_path, "processed_data.json").open('r') as fp:
    #     gt_data = json.load(fp)
    #
    # with Path(base_path, "target_gaits.json").open('r') as fp:
    #     gaits = json.load(fp)
    #
    # with Path(base_path, "processed_data_0.01v2.json").open('r') as fp:
    #     vis_gt_data = json.load(fp)
    #
    # # Set contact parameters
    #
    # # Will's sys id
    # # contact_params = {
    # #     "restitution": -0.16413898179172115,
    # #     "baumgarte": 0.38783653279732216,
    # #     "friction": 0.7593319176982094,
    # #     "friction_damping": 0.8689685753695016
    # # }
    #
    # # Patrick's sys id
    contact_params = {
        'restitution': -0.036485324706179156,
        'baumgarte': 0.03770246611256817,
        'friction': 0.4067791399229587,
        'friction_damping': 0.30583693613061996
    }
    #
    # # New open-source sys id
    # # contact_params = {
    # #     'restitution': -0.2395988899064052,
    # #     'baumgarte': 0.3358065718363488,
    # #     'friction': 1.1716651716431445,
    # #     'friction_damping': 0.9846809241468255
    # # }
    #
    config['contact_params'] = contact_params

    prims = [
        ("ccw", 1.0, 1.0),
        ("cw", 1.0, 1.0),
        ("roll", 1.0, 1.0),
    ]

    for prim in prims:
        start = time.time()
        start_end_pts = [
            [torch.tensor([-0.8607014597960383, -1.076264039468757, 0.2718694524182526],
                          dtype=torch.float64).reshape(1, 3, 1),
             torch.tensor([0.4619218124455742, 0.9927949124400106, 1.9079067849290041],
                          dtype=torch.float64).reshape(1, 3, 1)],
            [torch.tensor([0.8895405888183956, -1.0097047952850744, 0.18641452381580476],
                          dtype=torch.float64).reshape(1, 3, 1),
             torch.tensor([-1.1154402297545136, 1.0273033407592496, 0.9193600788678237],
                          dtype=torch.float64).reshape(1, 3, 1)],
            [torch.tensor([0.3035689922554887, -1.2530591438557919, 1.6217887425373758],
                          dtype=torch.float64).reshape(1, 3, 1),
             torch.tensor([0.3211102960310983, 1.3189297254103636, 0.17500000000000002],
                          dtype=torch.float64).reshape(1, 3, 1)]
        ]

        rest_lengths = [2.4 - 0.8349991846996137,
                        2.4 - 0.5910817854216738,
                        2.4 - 0.43105745581176963,
                        2.4 - 0.7215502303603479,
                        2.4 - 0.4086138651057926,
                        2.4 - 0.9001504827005256]
        motor_speeds = [0., 0., 0., 0., 0., 0.]

        sim, start_state = init_sim(config,
                                    start_end_pts,
                                    rest_lengths=rest_lengths,
                                    motor_speeds=motor_speeds)
        start_rest_lengths = [c.rest_length for c in sim.tensegrity_robot.actuated_cables.values()]
        start_motor_speed = [c.motor.motor_state.omega_t for c in sim.tensegrity_robot.actuated_cables.values()]
        all_states, _, _, all_controls = run_primitive(sim,
                                                       start_state,
                                                       start_rest_lengths,
                                                       start_motor_speed,
                                                       0.01,
                                                       prim[0],
                                                       prim[1],
                                                       prim[2])

        print(time.time() - start)
        visualize(all_states, 0.01, f"./vid_{prim[0]}.mp4")
