import json
import time
from pathlib import Path
from typing import Tuple, List

import torch
import tqdm

from mujoco_visualizer_utils.mujoco_visualizer import MuJoCoVisualizer
from simulators.tensegrity_simulator import TensegrityRobotSimulator
from state_objects.rods import RodState
from utilities import torch_quaternion


def run_by_control(simulator, ctrls, dt, curr_state, start_rest_lens, start_motor_speeds):
    with torch.no_grad():
        time = 0.0
        frames = []

        num_bodies = int(curr_state.flatten().shape[0] / 13)
        pos = torch.hstack([curr_state.flatten()[i * 13: i * 13 + 7] for i in range(num_bodies)]).detach().numpy()

        frames.append({"time": time, "pos": pos.tolist()})

        rest_lens, motor_speeds = start_rest_lens, start_motor_speeds
        for j, ctrl in enumerate(tqdm.tqdm(ctrls)):
            curr_state, rest_lens, motor_speeds = simulator.forward(
                curr_state,
                ctrl,
                dt,
                rest_lens,
                motor_speeds
            )

            pos = torch.hstack([curr_state.flatten()[i * 13: i * 13 + 7] for i in range(num_bodies)]).detach().numpy()
            frames.append({"time": round(time, 3), "pos": pos.tolist()})

    return frames


def batch_compute_end_pts(batch_state: torch.Tensor, rod_length=0.175) -> List[torch.Tensor]:
    """
    Compute end pts for entire batch

    :param batch_state: batch of states
    :return: batch of endpts
    """
    end_pts = []
    for i in range(3):
        state = batch_state[:, i * 13: i * 13 + 7]
        principal_axis = RodState.compute_principal_axis(state[:, 3:7])
        end_pts.extend(RodState.compute_end_pts_from_state(state, principal_axis, rod_length))

    return end_pts


def get_gait_time(gt_data, start_idx, end_idx):
    t0 = gt_data[start_idx]['time']
    t1 = gt_data[end_idx]['time']
    return t1 - t0


def get_endpts(gt_data):
    with torch.no_grad():
        dtype = torch.float64
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


def detect_ground_endcaps(end_pts) -> Tuple[int, int, int]:
    aug_end_pts = [[(i, end_pts[i]), (i + 1, end_pts[i + 1])] for i in range(0, len(end_pts), 2)]
    aug_end_pts = [min(e, key=lambda x: x[1].flatten()[2].item()) for e in aug_end_pts]

    ground_endcaps = tuple([a[0] for a in aug_end_pts])

    return ground_endcaps


def run_primitive(sim,
                  curr_state,
                  init_rest_lengths,
                  init_motor_speeds,
                  dt,
                  prim_type,
                  left_range,
                  right_range):
    symmetry_mapping = {
        (0, 2, 5): [0, 1, 2, 3, 4, 5], (0, 3, 5): [0, 1, 2, 3, 4, 5],
        (1, 2, 4): [1, 2, 0, 4, 5, 3], (1, 2, 5): [1, 2, 0, 4, 5, 3],
        (0, 3, 4): [2, 0, 1, 5, 3, 4], (1, 3, 4): [2, 0, 1, 5, 3, 4]
    }

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

    end_pts = batch_compute_end_pts(curr_state)
    ground_endcaps = detect_ground_endcaps(end_pts)
    order = symmetry_mapping[ground_endcaps]

    ranges = []
    for i in range(6):
        if prim_type == 'roll':
            range_ = left_range if i % 2 == 0 else right_range
        ranges.append(range_)

    ranges = torch.tensor(ranges, dtype=torch.float64).reshape(-1, 1, 1)
    min_lengths = torch.tensor(min_length, dtype=torch.float64).repeat(6).reshape(-1, 1, 1)
    tols = torch.tensor(tol, dtype=torch.float64).repeat(6).reshape(-1, 1, 1)

    rest_lens = init_rest_lengths
    motor_speeds = init_motor_speeds

    all_controls = []
    all_states, all_rest_lens, all_motor_speeds = [], [], []
    for gait in tqdm.tqdm(prim_gaits[prim_type]):
        target_gaits = torch.tensor([gait[i] for i in order],
                                    dtype=torch.float64
                                    ).reshape(-1, 1, 1)

        gait_states, gait_rest_lens, gait_motor_speeds, gait_ctrls = \
            run_target_gaits(sim,
                             curr_state,
                             dt,
                             rest_lens,
                             motor_speeds,
                             min_lengths, ranges, tols,
                             target_gaits)
        all_states.extend(gait_states)
        all_rest_lens.extend(gait_rest_lens)
        all_motor_speeds.extend(gait_motor_speeds)
        all_controls.extend(gait_ctrls)

        curr_state = gait_states[-1]
        rest_lens = gait_rest_lens[-1]
        motor_speeds = gait_motor_speeds[-1]

    return all_states, all_rest_lens, all_motor_speeds, all_controls


def run_target_gaits(sim,
                     curr_state,
                     dt,
                     in_rest_lengths,
                     in_motor_speeds,
                     min_length, range_, tol,
                     target_gaits):
    first_step = torch.tensor(0., dtype=torch.float64)

    last_error = torch.zeros_like(min_length)
    cum_error = torch.zeros_like(min_length)
    done_flag = torch.zeros_like(min_length, dtype=torch.bool)

    ctrls = torch.ones_like(min_length)

    states = [curr_state.clone()]
    rest_lengths = [in_rest_lengths.clone()]
    motor_speeds = [in_motor_speeds.clone()]
    controls = []

    rest_lens, omega_t = in_rest_lengths, in_motor_speeds

    while (ctrls != 0.0).any():
        # curr_state, ctrls, rest_lens, omega_t, last_error, cum_error, done_flag = \
        #     sim.forward(curr_state,
        #                 target_gaits,
        #                 dt,
        #                 rest_lens,
        #                 omega_t,
        #                 last_error, cum_error, done_flag,
        #                 min_length, range_, tol,
        #                 first_step)
        sim.forward(curr_state, ctrls, dt, rest_lens, omega_t, torch.zeros(18, dtype=torch.float64))

        states.append(curr_state.clone())
        rest_lengths.append(rest_lens.clone())
        motor_speeds.append(omega_t.clone())
        controls.append(ctrls.clone())

        first_step = torch.tensor(1.0, dtype=torch.float64)

    return states, rest_lengths, motor_speeds, controls


def init_sim(cfg, start_end_pts=None, start_state=None, rest_lengths=None, motor_speeds=None):
    assert start_end_pts is not None or start_state is not None

    sim = TensegrityRobotSimulator.init_from_config_file(cfg)

    if rest_lengths is not None:
        for i, c in enumerate(sim.tensegrity_robot.actuated_cables.values()):
            act_len = c._rest_length - rest_lengths[:, i: i + 1]
            c.actuation_length = act_len
            c.motor.speed = torch.tensor([[[0.99]]], dtype=torch.float64)

    if motor_speeds is not None:
        for i, c in enumerate(sim.tensegrity_robot.actuated_cables.values()):
            c.motor.motor_state.omega_t = motor_speeds[:, i: i + 1]

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
    config_file_path = "simulators/configs/3_bar_tensegrity_upscaled_patrick.json"
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
    dt = torch.tensor([[[0.01]]], dtype=torch.float64)

    prims = [
        ("ccw", 1.0, 1.0),
        ("cw", 1.0, 1.0),
        ("roll", 1.0, 1.0),
        ("ccw", 1.0, 1.0)
    ]

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

    rest_lengths = torch.tensor([
        2.4 - 0.8349991846996137,
        2.4 - 0.5910817854216738,
        2.4 - 0.43105745581176963,
        2.4 - 0.7215502303603479,
        2.4 - 0.4086138651057926,
        2.4 - 0.9001504827005256
    ], dtype=torch.float64).reshape(1, 6, 1)
    motor_speeds = torch.tensor([0., 0., 0., 0., 0., 0.],
                                dtype=torch.float64
                                ).reshape(1, 6, 1)

    script_sim = None
    for k, prim in enumerate(prims):
        sim, start_state = init_sim(config,
                                    start_end_pts,
                                    rest_lengths=rest_lengths,
                                    motor_speeds=motor_speeds)

        if k == 0:
            script_sim = torch.jit.trace(
                sim.forward,
                (
                    start_state,
                    torch.ones((6, 1, 1), dtype=torch.float64),
                    dt,
                    rest_lengths,
                    motor_speeds,
                    torch.zeros((6, 1, 1), dtype=torch.float64),
                    torch.zeros((6, 1, 1), dtype=torch.float64),
                    torch.zeros((6, 1, 1), dtype=torch.bool),
                    torch.zeros((6, 1, 1), dtype=torch.float64),
                    torch.zeros((6, 1, 1), dtype=torch.float64),
                    torch.zeros((6, 1, 1), dtype=torch.float64),
                    torch.tensor(True),
                )
            )

        start = time.time()
        p, l, r = prim
        all_states, _, _, all_controls = run_primitive(
            sim,
            start_state,
            rest_lengths,
            motor_speeds,
            dt,
            p,
            l,
            r
        )
        end0 = time.time() - start

        start = time.time()
        all_states, _, _, all_controls = run_primitive(
            script_sim,
            start_state,
            rest_lengths,
            motor_speeds,
            dt,
            p,
            l,
            r
        )
        end1 = time.time() - start

        print(f"Uncompiled time: {end0}, Compiled time: {end1}")

        visualize(all_states, 0.01, f"./vid_{p}.mp4")


def edgar():
    # Set paths
    config_file_path = "simulators/configs/3_bar_tensegrity_upscaled_patrick.json"
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
    dt = torch.tensor([[[0.01]]], dtype=torch.float64)

    prims = [
        ("ccw", 1.0, 1.0),
        ("cw", 1.0, 1.0),
        ("roll", 1.0, 1.0),
        ("ccw", 1.0, 1.0)
    ]

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

    rest_lengths = torch.tensor([
        2.4 - 0.8349991846996137,
        2.4 - 0.5910817854216738,
        2.4 - 0.43105745581176963,
        2.4 - 0.7215502303603479,
        2.4 - 0.4086138651057926,
        2.4 - 0.9001504827005256
    ], dtype=torch.float64).reshape(1, 6, 1)
    motor_speeds = torch.tensor([0., 0., 0., 0., 0., 0.],
                                dtype=torch.float64
                                ).reshape(1, 6, 1)

    sim, start_state = init_sim(config,
                                start_end_pts,
                                rest_lengths=rest_lengths,
                                motor_speeds=motor_speeds)

    script_sim = torch.jit.trace(
        sim.forward,
        (
            start_state,
            torch.ones((1, 6, 1), dtype=torch.float64),
            dt,
            rest_lengths,
            motor_speeds,
            torch.zeros(18, dtype=torch.float64),
        )
    )

    ctrls = torch.ones((1, 6, 1), dtype=torch.float64)
    script_sim.forward(start_state, ctrls, dt, rest_lengths, motor_speeds, torch.zeros(18, dtype=torch.float64))


if __name__ == '__main__':
    torch.set_anomaly_enabled(True)
    # with torch.no_grad():
    run_traj()
