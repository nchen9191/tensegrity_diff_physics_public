import json
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import tqdm

from mujoco_physics_engine.cable_motor import DCMotor
from mujoco_physics_engine.mujoco_simulation import AbstractMuJoCoSimulator
from mujoco_physics_engine.pid import PID

import mujoco

from utilities import quat_utils


class TensegrityMuJoCoSimulator(AbstractMuJoCoSimulator):

    def __init__(self,
                 xml_path: Path,
                 visualize: bool = True,
                 render_size: (int, int) = (720, 720),
                 render_fps: int = 100):
        super().__init__(xml_path, visualize, render_size, render_fps)
        self.min_spring_length = 1.0
        self.max_spring_length = 2.0
        self.actuator_tendon_ids = list(range(6))
        self.n_actuators = len(self.actuator_tendon_ids)
        self.curr_ctrl = [0.0 for _ in range(6)]
        self.pids = [PID() for _ in range(6)]
        self.pid_freq = 0.01
        self.rod_names = {
            0: "r01",
            1: "r23",
            2: "r45",
            # 3: "r67",
            # 4: "r89",
            # 5: "r1011"
        }
        self.n_rods = len(self.rod_names)
        # self.cable_sites = [
        #     ("s_3_5", "s_5_3"),
        #     ("s_1_3", "s_3_1"),
        #     ("s_1_5", "s_5_1"),
        #     ("s_0_2", "s_2_0"),
        #     ("s_0_4", "s_4_0"),
        #     ("s_2_4", "s_4_2"),
        #     ("s_2_5", "s_5_2"),
        #     ("s_0_3", "s_3_0"),
        #     ("s_1_4", "s_4_1")
        # ]
        # self.cable_sites = [
        #     ("s_3_b5", "s_b5_3"),
        #     ("s_1_b3", "s_b3_1"),
        #     ("s_5_b1", "s_b1_5"),
        #     ("s_0_b2", "s_b2_0"),
        #     ("s_4_b0", "s_b0_4"),
        #     ("s_2_b4", "s_b4_2"),
        #     ("s_3_5", "s_5_3"),
        #     ("s_1_3", "s_3_1"),
        #     ("s_1_5", "s_5_1"),
        #     ("s_0_2", "s_2_0"),
        #     ("s_0_4", "s_4_0"),
        #     ("s_2_4", "s_4_2"),
        #     ("s_2_5", "s_5_2"),
        #     ("s_0_3", "s_3_0"),
        #     ("s_1_4", "s_4_1")
        # ]
        # self.cable_map = {
        #     0: 6,
        #     1: 7,
        #     2: 8,
        #     3: 9,
        #     4: 10,
        #     5: 11
        # }
        self.cable_sites = [
            ("s3", "s5"),
            ("s1", "s3"),
            ("s5", "s1"),
            ("s0", "s2"),
            ("s4", "s0"),
            ("s2", "s4"),
        ]
        # self.cable_sites = [
        #     ("s0", "s10"),
        #     ("s1", "s4"),
        #     ("s2", "s6"),
        #     ("s1", "s3"),
        #     ("s4", "s8"),
        #     ("s2", "s5"),
        #     ("s5", "s6"),
        #     ("s7", "s11"),
        #     ("s7", "s8"),
        #     ("s0", "s9"),
        #     ("s9", "s10"),
        #     ("s3", "s11"),
        #     ("s0", "s8"),
        #     ("s0", "s4"),
        #     ("s1", "s10"),
        #     ("s3", "s10"),
        #     ("s9", "s11"),
        #     ("s7", "s9"),
        #     ("s2", "s4"),
        #     ("s1", "s2"),
        #     ("s3", "s6"),
        #     ("s6", "s11"),
        #     ("s5", "s7"),
        #     ("s5", "s8")
        # ]
        self.end_pts = [
            "s0", "s1", "s2", "s3", "s4", "s5",
            # "s6", "s7", "s8", "s9", "s10", "s11"
        ]
        self.stiffness = self.mjc_model.tendon_stiffness.copy()
        self.cable_motors = [DCMotor() for _ in range(6)]
        self.winch_r = 0.035

    def bring_to_grnd(self):
        self.forward()
        qpos = self.mjc_data.qpos.copy().reshape(-1, 7)
        end_pts = self.get_endpts().reshape(-1, 3)
        min_z = end_pts[:, 2].min()
        qpos[:, 2] -= min_z - 0.175
        self.mjc_data.qpos = qpos.reshape(1, -1)

    def reset(self):
        super().reset()
        self.bring_to_grnd()

        for motor in self.cable_motors:
            motor.reset_omega_t()

    def sim_step(self, controls=None):
        if controls is None:
            controls = self.curr_ctrl.copy()

        mujoco.mj_forward(self.mjc_model, self.mjc_data)
        for i, sites in enumerate(self.cable_sites):
            rest_length = self.mjc_model.tendon_lengthspring[i, 0]

            s0 = self.mjc_data.sensor(f"pos_{sites[0]}").data
            s1 = self.mjc_data.sensor(f"pos_{sites[1]}").data
            dist = np.linalg.norm(s1 - s0)

            self.mjc_model.tendon_stiffness[i] = 0 if dist < rest_length else self.stiffness[i]

            if controls is not None and i < 6:
                mod_ctrl = np.array(controls[i])

                dl = self.cable_motors[i].compute_cable_length_delta(mod_ctrl, self.winch_r, self.dt)
                rest_length = rest_length - dl
                self.mjc_model.tendon_lengthspring[i] = rest_length

        mujoco.mj_step(self.mjc_model, self.mjc_data)
        mujoco.mj_forward(self.mjc_model, self.mjc_data)

    def run_w_ctrls(self, ctrls):
        frames = [{'time': 0.0, 'pos': self.mjc_data.qpos.copy()}]

        for i, ctrl in enumerate(ctrls):
            self.sim_step(ctrl)
            frames.append({'time': i * self.dt, 'pos': self.mjc_data.qpos.copy()})

        return frames

    def get_endpts(self):
        end_pts = []
        for end_pt_site in self.end_pts:
            end_pt = self.mjc_data.sensor(f"pos_{end_pt_site}").data
            end_pts.append(end_pt)

        end_pts = np.vstack(end_pts)
        return end_pts

    def run(self,
            end_time: float = None,
            num_steps: int = None,
            save_path: Path = None,
            pos_sensor_names: Optional[List] = None,
            quat_sensor_names: Optional[List] = None,
            linvel_sensor_names: Optional[List] = None,
            angvel_sensor_names: Optional[List] = None):

        mujoco.mj_forward(self.mjc_model, self.mjc_data)

        end_pts = [self.get_endpts()]
        pos = [self.mjc_data.qpos.copy()]
        vel = [self.mjc_data.qvel.copy()]
        frames = [self.render_frame("front")]
        num_steps_per_frame = int(1 / self.render_fps / self.dt)
        for n in range(num_steps):
            if (n + 1) % 100 == 0:
                print((n + 1) * self.dt)

            self.sim_step()

            mujoco.mj_forward(self.mjc_model, self.mjc_data)
            end_pts.append(self.get_endpts())
            pos.append(self.mjc_data.qpos.copy())
            vel.append(self.mjc_data.qvel.copy())

            if self.visualize and ((n + 1) % num_steps_per_frame == 0 or n == num_steps - 1):
                frame = self.render_frame()
                frames.append(frame.copy())

        self.save_video(Path(save_path, "gt_vid.mp4"), frames)

        return end_pts, pos, vel

    def run_w_target_gaits(self, target_gaits, save_path=None):
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)

        max_steps = 120000

        mujoco.mj_forward(self.mjc_model, self.mjc_data)
        pose_idx = np.arange(7 * self.n_rods).reshape(-1, 7)
        vel_idx = np.arange(6 * self.n_rods).reshape(-1, 6)
        pos_idx = pose_idx[:, :3].flatten().tolist()
        quat_idx = pose_idx[:, 3:7].flatten().tolist()
        linvel_idx = vel_idx[:, :3].flatten().tolist()
        angvel_idx = vel_idx[:, 3:].flatten().tolist()

        data = [{
            "time": 0.0,
            "end_pts": [
                self.mjc_data.sensor(f"pos_{s}").data.tolist()
                for s in self.end_pts
            ],
            "sites": {
                s: self.mjc_data.sensor(f"pos_{s}").data.tolist()
                for c in self.cable_sites for s in c
            },
            "pos": self.mjc_data.qpos[pos_idx].tolist(),
            "quat": self.mjc_data.qpos[quat_idx].tolist(),
            "linvel": self.mjc_data.qvel[linvel_idx].tolist(),
            "angvel": self.mjc_data.qvel[angvel_idx].tolist(),
            # "init_rest_lengths": self.mjc_model.tendon_lengthspring[:6, 0].tolist(),
            "pid": {
                "min_length": self.pids[0].min_length,
                "RANGE": self.pids[0].RANGE,
                "tol": self.pids[0].tol,
                "motor_speed": self.cable_motors[0].speed.item()
            }
        }]
        extra_data = []
        target_gaits_dicts = []
        key_frame_ids = []

        frames = [self.render_frame()]

        num_steps_per_frame = int(1 / self.render_fps / self.dt)
        global_steps = 0
        for k, target_gait_dict in enumerate(tqdm.tqdm(target_gaits)):
            for pid in self.pids:
                pid.reset()
            target_gait = target_gait_dict['target_gait']
            # print(k)
            step = 0
            controls = [1, 0, 0, 0, 0, 0]

            target_gaits_dicts.append({'idx': global_steps, "target_gait": target_gait})

            while any([c != 0 for c in controls]) and step < max_steps:
                step += 1
                global_steps += 1
                mujoco.mj_forward(self.mjc_model, self.mjc_data)

                e_i = [p.cum_error.item() if p.cum_error is not None else 0.0 for p in self.pids]
                e_d = [p.last_error.item() if p.cum_error is not None else 0.0 for p in self.pids]

                if global_steps % (self.pid_freq // self.dt) == 0 or step == 1:
                    controls = []
                    for i in range(len(target_gait)):
                        pid = self.pids[i]
                        gait = target_gait[i]

                        rest_length = self.mjc_model.tendon_lengthspring[i, 0]
                        # curr_length = self.mjc_model.tendon_length0[i]
                        idx = self.cable_map[i] if hasattr(self, "cable_map") and self.cable_map else i
                        s0 = self.mjc_data.sensor(f"pos_{self.cable_sites[idx][0]}").data
                        s1 = self.mjc_data.sensor(f"pos_{self.cable_sites[idx][1]}").data
                        curr_length = np.linalg.norm(s1 - s0)

                        ctrl, _ = pid.update_control_by_target_gait(curr_length, gait, rest_length)
                        controls.append(ctrl)

                extra_data.append({
                    "time": self.dt * (global_steps - 1),
                    "dt": self.dt,
                    "rest_lengths": self.mjc_model.tendon_lengthspring[:6, 0].tolist(),
                    "motor_speeds": [c.motor_state.omega_t[0] for c in self.cable_motors],
                    "e_i": e_i,
                    "e_d": e_d,
                    "controls": [c[0] for c in controls]
                })

                self.sim_step(controls)
                mujoco.mj_forward(self.mjc_model, self.mjc_data)

                data.append({
                    "time": self.dt * global_steps,
                    "end_pts": [
                        self.mjc_data.sensor(f"pos_{s}").data.tolist()
                        for s in self.end_pts
                    ],
                    "sites": {
                        s: self.mjc_data.sensor(f"pos_{s}").data.tolist()
                        for c in self.cable_sites for s in c
                    },
                    "pos": self.mjc_data.qpos[pos_idx].tolist(),
                    "quat": self.mjc_data.qpos[quat_idx].tolist(),
                    "linvel": self.mjc_data.qvel[linvel_idx].tolist(),
                    "angvel": self.mjc_data.qvel[angvel_idx].tolist()
                })

                if self.visualize and (global_steps % num_steps_per_frame == 0):
                    frame = self.render_frame()
                    frames.append(frame.copy())

            key_frame_ids.append(global_steps)

        if save_path:
            self.save_video(Path(save_path, "gt_vid.mp4"), frames)

            with Path(save_path, "processed_data.json").open("w") as fp:
                json.dump(data, fp)

            with Path(save_path, "extra_state_data.json").open("w") as fp:
                json.dump(extra_data, fp)

            with Path(save_path, "target_gaits.json").open("w") as fp:
                json.dump(target_gaits_dicts, fp)

        return data, key_frame_ids


if __name__ == '__main__':
    xml1 = Path("xml_models/3prism_real_upscaled_all_cables.xml")
    xml2 = Path("xml_models/3prism_real_upscaled_vis.xml")

    basebase = Path("/Users/nelsonchen/Documents/Rutgers-CS-PhD/Research/tensegrity/data_sets/tensegrity_real_datasets/")
    output_path = Path(basebase, "synthetic/mjc_synthetic_6d_0.001//")
    output_path.mkdir(exist_ok=True)

    base_path = Path(basebase, "R2S2R/test/")
    for path in base_path.iterdir():
        if "R2S2R" not in path.name:
            continue

        print(path.name)

        final_output_path = output_path / path.name
        final_output_path.mkdir(exist_ok=True)

        with (path / "target_gaits.json").open('r') as fp:
            target_gaits = json.load(fp)

        sim = TensegrityMuJoCoSimulator(xml1)
        mujoco.mj_forward(sim.mjc_model, sim.mjc_data)

        sim.bring_to_grnd()
        frames = sim.run_w_target_gaits(target_gaits, final_output_path)

        del sim
