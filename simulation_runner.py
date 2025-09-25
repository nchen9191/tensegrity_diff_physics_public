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


def batch_compute_end_pts(batch_state: torch.Tensor, rod_length=2.95) -> List[torch.Tensor]:
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
        curr_state, ctrls, rest_lens, omega_t, last_error, cum_error, done_flag = \
            sim.forward(curr_state,
                        target_gaits,
                        dt,
                        rest_lens,
                        omega_t,
                        last_error, cum_error, done_flag,
                        min_length, range_, tol,
                        first_step)
        # sim.forward(curr_state, ctrls, dt, rest_lens, omega_t, torch.zeros(18, dtype=torch.float64))

        states.append(curr_state.clone())
        rest_lengths.append(rest_lens.clone())
        motor_speeds.append(omega_t.clone())
        controls.append(ctrls.clone())

        first_step = torch.tensor(1.0, dtype=torch.float64)

    return states, rest_lengths, motor_speeds, controls


def end_pts_to_start_state(end_pts):
    pos = (end_pts[:, 3:] + end_pts[:, :3]) / 2.

    prin = end_pts[:, 3:] - end_pts[:, :3]
    prin = prin / prin.norm(dim=1, keepdim=True)
    quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin)

    vel = torch.zeros_like(pos)
    start_state = torch.hstack([pos, quat, vel, vel]).reshape(1, -1, 1)
    return start_state


def init_sim(cfg, start_end_pts=None, start_state=None, rest_lengths=None, motor_speeds=None):
    assert start_end_pts is not None or start_state is not None

    sim = TensegrityRobotSimulator.init_from_config_file(cfg)
    for i, c in enumerate(sim.tensegrity_robot.actuated_cables.values()):
        if rest_lengths is not None:
            act_len = c._rest_length - rest_lengths[:, i: i + 1]
            c.actuation_length = act_len
            c.motor.speed = torch.tensor([[[0.7]]], dtype=torch.float64)

    if motor_speeds is not None:
        for i, c in enumerate(sim.tensegrity_robot.actuated_cables.values()):
            c.motor.motor_state.omega_t = motor_speeds[:, i: i + 1]

    if start_end_pts is None:
        sim.update_state(start_state)
        start_end_pts = [e for r in sim.tensegrity_robot.rods.values() for e in r.end_pts]
    elif start_state is None:
        end_pts = start_end_pts.reshape(-1, 6, 1)
        start_state = end_pts_to_start_state(end_pts)
        sim.update_state(start_state)

    start_end_pts = [
        [start_end_pts[2 * k].unsqueeze(0), start_end_pts[2 * k + 1].unsqueeze(0)]
        for k in range(start_end_pts.shape[0])
    ]
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


def stabilize_robot(sim_fn,
                    dt,
                    init_end_pts,
                    rest_lengths,
                    motor_speeds,
                    max_time=10.,
                    vel_tol=1e-2):
    # Scale
    max_steps = round(max_time / dt.item())
    dummy_ctrls = torch.zeros(6)
    curr_state = end_pts_to_start_state(init_end_pts)

    for _ in tqdm.tqdm(range(max_steps)):
        output = sim_fn(
            curr_state,
            dummy_ctrls,
            dt,
            rest_lengths,
            motor_speeds,
        )
        rest_lengths, motor_speeds, curr_state = output[-3:]
        vels = curr_state.reshape(-1, 13, 1)[:, 7:].reshape(-1, 3)

        if vels.norm(dim=1).max() < vel_tol:
            break

    curr_state_ = curr_state.reshape(-1, 13, 1)
    zero_vels = torch.zeros_like(curr_state_[:, :6])

    stable_state = torch.hstack([curr_state_[:, :7], zero_vels]).flatten()

    return stable_state, rest_lengths, motor_speeds


def align_prin_axis_2d(curr_end_pts, new_end_pts):
    new_end_pts = new_end_pts.reshape(-1, 3, 1)
    new_com = new_end_pts.mean(dim=0, keepdim=True)
    new_robot_prin = new_end_pts[1::2].mean(dim=1, keepdim=True) - new_end_pts[::2].mean(dim=1, keepdim=True)
    new_robot_prin[:, 2] = 0.
    new_robot_prin = new_robot_prin / new_robot_prin.norm(dim=1, keepdim=True)

    curr_end_pts = curr_end_pts.reshape(-1, 3, 1)
    curr_com = curr_end_pts.mean(dim=0, keepdim=True)
    curr_robot_prin = curr_end_pts[1::2].mean(dim=1, keepdim=True) - curr_end_pts[::2].mean(dim=1, keepdim=True)
    curr_robot_prin[:, 2] = 0.
    curr_robot_prin = curr_robot_prin / curr_robot_prin.norm(dim=1, keepdim=True)

    prin = curr_end_pts[1::2] - curr_end_pts[::2]
    prin = prin / prin.norm(dim=1, keepdim=True)
    curr_quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin)
    curr_pos = (curr_end_pts[1::2] + curr_end_pts[::2]) / 2.

    rot_axis = torch.cross(curr_robot_prin, new_robot_prin, dim=1)
    rot_axis = rot_axis / rot_axis.norm(dim=1, keepdim=True)
    angle = torch.linalg.vecdot(curr_robot_prin, new_robot_prin, dim=1).unsqueeze(1)
    angle = torch.acos(torch.clamp(angle, -1, 1)) / 2.
    rot_quat = torch.hstack([torch.cos(angle), torch.sin(angle) * rot_axis])

    new_com[:, 2] = curr_com[:, 2]
    new_rod_pos = curr_pos[:, :3] - curr_com
    new_rod_pos = torch_quaternion.rotate_vec_quat(rot_quat, new_rod_pos) + new_com
    new_rod_quat = torch_quaternion.quat_prod(rot_quat, curr_quat)
    new_vels = torch.zeros_like(new_rod_pos).repeat(1, 2, 1)

    new_robot_state = torch.hstack([new_rod_pos, new_rod_quat, new_vels]).reshape(1, -1, 1)

    return new_robot_state


def run_robot_init_stabilization(
        sim_fn,
        dt,
        init_end_pts,
        init_rest_lengths,
        init_motor_speeds,
        max_time=10.,
        vel_tol=1e-2,
        rod_length=0.325,
        num_act_cables=6,
        pid_min_len=0.8,
        pid_range=1.0,
        pid_tol=0.15
):
    with torch.no_grad():
        # Scale
        init_end_pts = 10 * init_end_pts
        init_rest_lengths = 10 * init_rest_lengths
        rod_length = 10 * rod_length

        # Run light stabilization
        curr_state, rest_lengths, motor_speeds = stabilize_robot(
            sim_fn,
            init_end_pts,
            init_rest_lengths,
            init_motor_speeds,
            dt,
            max_time=3.0,
            vel_tol=vel_tol
        )

        # Go to rest target lengths
        pid_params = [pid_min_len, pid_range, pid_tol]
        rest_target_gaits = [[1.0] * num_act_cables]
        states, rest_lengths, motor_speeds, _ = run_target_gaits(
            sim_fn,
            curr_state,
            dt,
            rest_lengths,
            motor_speeds,
            *pid_params,
            target_gaits=rest_target_gaits,
        )
        curr_end_pts = torch.vstack(batch_compute_end_pts(states[-1], rod_length))

        # Run full stabilization
        curr_state, rest_lengths, motor_speeds = stabilize_robot(
            sim_fn,
            curr_end_pts,
            rest_lengths[-1],
            motor_speeds[-1],
            dt,
            max_time=max_time,
            vel_tol=vel_tol
        )
        curr_end_pts = torch.vstack(batch_compute_end_pts(curr_state, rod_length))

        # Realign robot
        start_state = align_prin_axis_2d(curr_end_pts, init_end_pts).reshape(-1, 13, 1)
        start_state[:, :3] /= 10
        start_state[:, 7:10] /= 10
        start_state = start_state.flatten()

    return start_state,


def run_traj():
    # Set paths
    config_file_path = "simulators/configs/3_bar_tensegrity_upscaled_patrick.json"

    # Get config
    with Path(config_file_path).open("r") as j:
        config = json.load(j)

    # Set contact parameters

    # Will's sys id
    contact_params = {
        "restitution": -0.16413898179172115,
        "baumgarte": 0.38783653279732216,
        "friction": 0.7593319176982094,
        "friction_damping": 0.8689685753695016
    }
    #
    # # Patrick's sys id
    # contact_params = {
    #     'restitution': -0.036485324706179156,
    #     'baumgarte': 0.03770246611256817,
    #     'friction': 0.4067791399229587,
    #     'friction_damping': 0.30583693613061996
    # }
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
    config_file_path = "simulators/configs/3_bar_tensegrity_upscaled.json"
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
    # Set contact parameters
    #
    # Will's sys id
    contact_params = {
        "restitution": -0.16413898179172115,
        "baumgarte": 0.38783653279732216,
        "friction": 0.7593319176982094,
        "friction_damping": 0.8689685753695016
    }
    #
    # # Patrick's sys id
    # contact_params = {
    #     'restitution': -0.036485324706179156,
    #     'baumgarte': 0.03770246611256817,
    #     'friction': 0.4067791399229587,
    #     'friction_damping': 0.30583693613061996
    # }
    #
    # # New open-source sys id
    # contact_params = {
    #     "restitution": -0.09205133585930521,
    #     "baumgarte": 0.05383130669413745,
    #     "friction": 0.4788313614955433,
    #     "friction_damping": 0.976783322659442,
    # }
    #
    config['contact_params'] = contact_params
    dt = torch.tensor(0.01, dtype=torch.float64)

    base_path = Path(
        "/Users/nelsonchen/research/tensegrity/data_sets/tensegrity_real_datasets/R2S2R/train/")
    j = -1
    for path in base_path.iterdir():
        if 'mjc' in path.name:
            continue
        if 'cw' not in path.name and 'roll' not in path.name:
            continue
        if not path.is_dir():
            continue
        print(path.name)
        j += 1

        gt_data = json.load((path / "6d_processed_data.json").open('r'))
        gt_extra_data = json.load((path / "5d_extra_state_data_0.01.json").open('r'))

        start_end_pts = torch.tensor([
            gt_data[0]['rod_01_end_pt1'],
            gt_data[0]['rod_01_end_pt2'],
            gt_data[0]['rod_23_end_pt1'],
            gt_data[0]['rod_23_end_pt2'],
            gt_data[0]['rod_45_end_pt1'],
            gt_data[0]['rod_45_end_pt2']
        ], dtype=torch.float64).reshape(1, 3, 1)

        rest_lengths = torch.tensor(gt_extra_data[0]['rest_lengths'],
                                    dtype=torch.float64).reshape(1, 6, 1)
        motor_speeds = torch.tensor([0., 0., 0., 0., 0., 0.],
                                    dtype=torch.float64).reshape(1, 6, 1)

        sim, start_state = init_sim(config,
                                    start_end_pts,
                                    rest_lengths,
                                    motor_speeds)

        if j == 0:
            start_state = start_state.flatten()
            ctrls = torch.ones(6, dtype=torch.float64)
            dt = dt.flatten()
            rest_lengths = rest_lengths.flatten()
            motor_speeds = motor_speeds.flatten()
            script_sim = torch.jit.trace(
                sim.forward,
                (
                    start_state,
                    ctrls,
                    dt,
                    rest_lengths,
                    motor_speeds,
                )
            )

            script_sim.save("/Users/nelsonchen/Desktop/old_platform_script_sim.pt")

        # ctrls = torch.ones((1, 6, 1), dtype=torch.float64)
        # script_sim.forward(start_state, ctrls, dt, rest_lengths, motor_speeds, torch.zeros(18, dtype=torch.float64))


if __name__ == '__main__':
    with torch.no_grad():
        edgar()
