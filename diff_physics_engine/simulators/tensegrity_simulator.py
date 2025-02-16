from typing import Dict, Tuple, Union, Optional

import torch

from diff_physics_engine.actuation.pid import PID
from diff_physics_engine.contact.collision_detector import get_detector
from diff_physics_engine.contact.learnable_collision_response import \
    CollisionResponseGenerator as DiffCollisionResponseGenerator
from diff_physics_engine.robots.tensegrity import TensegrityRobot
from diff_physics_engine.simulators.abstract_simulator import AbstractSimulator
from diff_physics_engine.state_objects.rods import RodState
from diff_physics_engine.state_objects.springs import ActuatedCable
from utilities import misc_utils, torch_quaternion


class TensegrityRobotSimulator(AbstractSimulator):

    def __init__(self,
                 tensegrity_robot,
                 gravity,
                 contact_params,
                 sys_precision=torch.float64,
                 use_contact=True):
        super().__init__(sys_precision)
        self.tensegrity_robot = tensegrity_robot
        self.gravity = gravity
        self.use_contact = use_contact
        self.contact_params = contact_params

        if use_contact:
            self.collision_resp_gen = DiffCollisionResponseGenerator()
            self.collision_resp_gen.set_contact_params('default', contact_params)
        else:
            self.collision_resp_gen = None

        self.pids = {f"pid_spring_{i}": PID(min_length=100, tol=0.10, RANGE=100) for i in range(6)}
        # self.pids = {f"pid_spring_{i}": PID(min_length=70, tol=0.1, RANGE=100) for i in range(9, 15)}
        self.sys_params.update({"pids": self.pids})

    def forward(self, x, ctrls, dt):
        return self.step(x, dt, control_signals=ctrls)

    def move_tensors(self, device):
        self.tensegrity_robot = self.tensegrity_robot.move_tensors(device)
        self.gravity = self.gravity.to(device)

        if self.collision_resp_gen:
            self.collision_resp_gen.move_tensors(device)

        for k, pid in self.pids.items():
            self.pids[k] = pid.move_tensors(device)

        return self

    # def detach(self):
    #     self.tensegrity_robot.detach()
    #     if self.collision_resp_gen is not None:
    #         self.collision_resp_gen.detach()

    @classmethod
    def init_from_config_file(cls, config_json: Dict):
        """
        Instantiate SpringRodSimulator object with config json file
        :param config_json: Dict
        :return: SpringRodSimulator object
        """
        sys_precision = misc_utils.get_num_precision(config_json['sys_precision'])
        contact_params = {k1: torch.tensor(v1, dtype=sys_precision)
                          for k1, v1 in config_json['contact_params'].items()}

        gravity = config_json['gravity'] \
            if 'gravity' in config_json else [0, 0, -9.81]
        gravity = torch.tensor(gravity, dtype=sys_precision).reshape(1, 3, 1)

        robot = TensegrityRobot(config_json['tensegrity_robot_config'],
                                sys_precision)

        instance = cls(robot, gravity, contact_params, sys_precision)

        return instance

    @property
    def rigid_bodies(self):
        return self.tensegrity_robot.rods

    def get_body_vecs(self, curr_state, acting_pts):
        num_bodies = len(self.tensegrity_robot.rods)
        pos = torch.hstack([curr_state[:, i * 13: i * 13 + 3]
                            for i in range(num_bodies)])
        body_vecs = acting_pts - pos

        return body_vecs

    def compute_forces(self,
                       external_forces: torch.Tensor,
                       external_pts: torch.Tensor
                       ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        net_spring_forces, spring_forces, acting_pts \
            = self.tensegrity_robot.compute_spring_forces()
        gravity_forces = torch.hstack([rod.mass * self.gravity
                                       for rod in self.tensegrity_robot.rods.values()])
        net_forces = net_spring_forces + gravity_forces
        # net_forces = net_spring_forces + gravity_forces + external_forces

        return net_forces, spring_forces, acting_pts

    def compute_torques(self,
                        forces: torch.Tensor,
                        body_vecs: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = forces.shape
        net_torques = torch.cross(
            body_vecs.view(-1, 3, shape[2]),
            forces.view(-1, 3, shape[2]),
            dim=1
        ).view(shape).sum(dim=2, keepdim=True)

        return net_torques, None

    def compute_accelerations(self,
                              net_force: torch.Tensor,
                              net_torque: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        lin_accs, ang_accs = [], []
        for i, rod in enumerate(self.tensegrity_robot.rods.values()):
            start, end = i * 3, (i + 1) * 3

            lin_acc = net_force[:, start: end] / rod.mass
            ang_acc = torch.matmul(rod.I_world_inv,
                                   net_torque[:, start: end])

            if rod.fixed:
                lin_acc[:] = 0
                ang_acc[:] = 0

            lin_accs.append(lin_acc)
            ang_accs.append(ang_acc)

        lin_acc = torch.hstack(lin_accs)
        ang_acc = torch.hstack(ang_accs)

        return lin_acc, ang_acc

    def time_integration(self,
                         lin_acc: torch.Tensor,
                         ang_acc: torch.Tensor,
                         dt: float) -> torch.Tensor:
        n = len(self.tensegrity_robot.rods)

        ang_vel = (self.tensegrity_robot.ang_vel + dt * ang_acc).reshape(-1, 3, 1)
        lin_vel = (self.tensegrity_robot.linear_vel + dt * lin_acc).reshape(-1, 3, 1)
        pos = self.tensegrity_robot.pos.reshape(-1, 3, 1) + dt * lin_vel

        # q = self.tensegrity_robot.quat.view(-1, n, 4).transpose(1, 2)
        # ang_vel_ = ang_vel.view(-1, n, 3).transpose(1, 2)
        # quat = (torch_quaternion.update_quat(q, ang_vel_, dt)
        #         .transpose(1, 2)
        #         .reshape(-1, n * 4, 1))
        q = self.tensegrity_robot.quat.reshape(-1, 4, 1)
        quat = torch_quaternion.update_quat(q, ang_vel, dt)

        next_state = torch.hstack([pos, quat, lin_vel, ang_vel]).reshape(-1, 13 * n, 1)

        # next_state = torch.hstack([
        #     pos.view(-1, n, 3).transpose(1, 2),
        #     quat.view(-1, n, 4).transpose(1, 2),
        #     lin_vel.view(-1, n, 3).transpose(1, 2),
        #     ang_vel.view(-1, n, 3).transpose(1, 2)
        # ]).transpose(1, 2).reshape(-1, n * 13, 1)
        # print(next_state.detach().flatten())

        return next_state

    def get_curr_state(self) -> torch.Tensor:
        return torch.hstack([
            torch.hstack([
                rod.pos,
                rod.quat,
                rod.linear_vel,
                rod.ang_vel,
            ])
            for rod in self.tensegrity_robot.rods.values()
        ])

    def update_state(self, next_state: torch.Tensor) -> None:
        n = len(self.tensegrity_robot.rods)
        self.tensegrity_robot.update_state(
            torch.hstack([next_state[:, i * 13: i * 13 + 3] for i in range(n)]),
            torch.hstack([next_state[:, i * 13 + 7: i * 13 + 10] for i in range(n)]),
            torch.hstack([next_state[:, i * 13 + 3: i * 13 + 7] for i in range(n)]),
            torch.hstack([next_state[:, i * 13 + 10: i * 13 + 13] for i in range(n)]),
        )

        self.update_system_topology()

    def compute_aux_states(self, curr_state: torch.Tensor) -> None:
        # stored_state = self.get_curr_state()
        # if ((curr_state.shape != stored_state.shape)
        #         or (curr_state != stored_state).any()):
        self.update_state(curr_state)

    def update_system_topology(self) -> None:
        self.tensegrity_robot.update_system_topology()

    def compute_contact_deltas(self,
                               pre_next_state: torch.Tensor,
                               dt: Union[torch.Tensor, float]
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # n = len(self.tensegrity_robot.rods)
        # dummy_rod = list(self.tensegrity_robot.rods.values())[0]
        # p = torch.vstack([rod.pos for rod in self.tensegrity_robot.rods.values()])
        # q = torch.vstack([rod.quat for rod in self.tensegrity_robot.rods.values()])
        # lv = torch.vstack([rod.linear_vel for rod in self.tensegrity_robot.rods.values()])
        # av = torch.vstack([rod.ang_vel for rod in self.tensegrity_robot.rods.values()])
        # dummy_rod.update_state(p, lv, q, av)
        # pre_next_state_ = torch.vstack([pre_next_state[:, i * 13: (i + 1) * 13] for i in range(n)])
        #
        # detector = get_detector(dummy_rod, self.collision_resp_gen.ground)
        # _, _, delta_v, delta_w, toi = self.collision_resp_gen.resolve_contact_ground(dummy_rod,
        #                                                                              pre_next_state_,
        #                                                                              dt,
        #                                                                              detector)
        delta_v, delta_w, toi = [], [], []
        for i, rod in enumerate(self.tensegrity_robot.rods.values()):
            rod_pre_next_state = pre_next_state[:, i * 13: (i + 1) * 13]
            detector = get_detector(rod, self.collision_resp_gen.ground)
            _, _, dv, dw, dtoi = self.collision_resp_gen.resolve_contact_ground(rod,
                                                                                rod_pre_next_state,
                                                                                dt,
                                                                                detector)
            delta_v.append(dv)
            delta_w.append(dw)
            toi.append(dtoi)

        delta_v = torch.vstack(delta_v)
        delta_w = torch.vstack(delta_w)
        toi = torch.vstack(toi)

        return delta_v, delta_w, toi

    def resolve_contacts(self,
                         pre_next_state: torch.Tensor,
                         dt: Union[torch.Tensor, float],
                         delta_v,
                         delta_w,
                         toi) -> torch.Tensor:
        # print(pre_next_state.detach().flatten())
        rods = self.tensegrity_robot.rods.values()
        n = len(rods)
        pre_next_state_ = torch.vstack([pre_next_state[:, i * 13: (i + 1) * 13] for i in range(n)])
        size = pre_next_state.shape[0]
        num = int(delta_v.shape[0] // size)

        lin_vel = pre_next_state_[:, 7:10, ...] + delta_v
        ang_vel = pre_next_state_[:, 10:, ...] + delta_w
        pos = torch.vstack([r.pos for r in rods]) + dt * lin_vel - delta_v * toi
        # pos2 = pre_next_state_[:, :3] + delta_v * (dt - toi)

        # curr_quat = torch.vstack([r.quat for r in rods])
        # prin_axis = RodState.compute_principal_axis(curr_quat)
        # prin_ang_vel = torch.linalg.vecdot(ang_vel, prin_axis, dim=1).unsqueeze(1) * prin_axis
        # ang_vel = ang_vel - 0.9 * prin_ang_vel

        # quat = torch_quaternion.update_quat(curr_quat, ang_vel, dt)
        # quat = torch_quaternion.update_quat(quat, -delta_w, toi)
        quat = torch_quaternion.update_quat(pre_next_state_[:, 3:7], delta_w, dt - toi)

        next_state_ = torch.hstack([pos, quat, lin_vel, ang_vel])
        next_state = torch.hstack([next_state_[i * size: (i + 1) * size] for i in range(num)])

        return next_state

    def _compute_cable_length(self, cable):
        end_pt0 = self.tensegrity_robot.system_topology.sites_dict[cable.end_pts[0]]
        end_pt1 = self.tensegrity_robot.system_topology.sites_dict[cable.end_pts[1]]

        x_dir = end_pt1 - end_pt0
        length = x_dir.norm(dim=1, keepdim=True)
        x_dir = x_dir / length

        return length, x_dir

    def step_w_target_lengths(self,
                              curr_state: torch.Tensor,
                              dt: Union[torch.Tensor, float],
                              external_forces: Dict = None,
                              external_pts: Dict = None,
                              target_lengths: Dict = None) -> torch.Tensor:

        for k, v in target_lengths.items():
            if not isinstance(v, torch.Tensor):
                target_lengths[k] = torch.tensor(
                    v,
                    dtype=curr_state.dtype,
                    device=curr_state.device
                ).reshape(-1, 1, 1)

        self.compute_aux_states(curr_state)
        name_order = [n for n, s in self.tensegrity_robot.springs.items()
                      if isinstance(s, ActuatedCable)]

        curr_lengths_tensor, _ = torch.hstack([
            self._compute_cable_length(self.tensegrity_robot.springs[n])
            for n in name_order
        ])
        target_lengths_tensor = torch.hstack([
            (
                target_lengths[n]
                if n in target_lengths
                else curr_lengths_tensor[:, i: i + 1]
            )
            for i, n in enumerate(name_order)
        ])

        controls = self.pid.update_control_target_length(curr_lengths_tensor,
                                                         target_lengths_tensor)
        for i, name in enumerate(name_order):
            cable = self.tensegrity_robot.actuated_cables[name]
            cable.update_rest_length(controls[:, i],
                                     curr_lengths_tensor[:, i],
                                     dt)

        print(curr_lengths_tensor[0, 0, 0])
        self.tensegrity_robot.springs.update(self.tensegrity_robot.actuated_cables)
        next_state = super().step(curr_state,
                                  dt,
                                  external_forces,
                                  external_pts)

        return next_state

    def step_w_controls(self,
                        curr_state: torch.Tensor,
                        dt: Union[torch.Tensor, float],
                        external_forces: Dict = None,
                        external_pts: Dict = None,
                        controls_dict: Dict = None) -> torch.Tensor:
        if isinstance(controls_dict, torch.Tensor):
            controls_dict = {f'spring_{i}': controls_dict[:, i: i + 1, None]
                             for i in range(controls_dict.shape[1])}
        elif isinstance(controls_dict, list):
            controls_dict = {
                f'spring_{i}':
                    ctrl.reshape(-1, 1, 1)
                    if isinstance(ctrl, torch.Tensor)
                    else torch.tensor(ctrl, dtype=self.sys_precision).reshape(-1, 1, 1)
                for i, ctrl in enumerate(controls_dict)
            }

        self.compute_aux_states(curr_state)

        for name, control in controls_dict.items():
            if not isinstance(control, torch.Tensor):
                control = torch.tensor(control,
                                       dtype=self.sys_precision,
                                       device=curr_state.device).reshape(-1, 1)

            measure_name = self.tensegrity_robot.cable_map[name]
            measure_cable = self.tensegrity_robot.springs[measure_name]
            cable = self.tensegrity_robot.springs[name]

            curr_length, _ = self.tensegrity_robot.compute_cable_length(measure_cable)
            cable.update_rest_length(control, curr_length, dt)

        self.tensegrity_robot.springs.update(self.tensegrity_robot.actuated_cables)
        next_state = super().step(curr_state,
                                  dt,
                                  external_forces,
                                  external_pts)

        return next_state

    def step_with_target_gait(self,
                              curr_state: torch.Tensor,
                              dt: Union[torch.Tensor, float],
                              external_forces: Dict = None,
                              external_pts: Dict = None,
                              target_gait_dict: Dict = None):
        if external_forces is None or external_pts is None:
            ext_f_dim = len(self.tensegrity_robot.rods) * 3
            size = (curr_state.shape[0], ext_f_dim, 1)
            external_forces = torch.zeros(size, dtype=self.sys_precision, device=curr_state.device)
            external_pts = torch.zeros(size, dtype=self.sys_precision, device=curr_state.device)

        controls = []
        self.compute_aux_states(curr_state)

        for name, target in target_gait_dict.items():
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(
                    target,
                    dtype=self.sys_precision,
                    device=curr_state.device
                ).reshape(-1, 1).detach()

            measure_name = self.tensegrity_robot.cable_map[name]
            measure_cable = self.tensegrity_robot.springs[measure_name]
            cable = self.tensegrity_robot.actuated_cables[name]

            curr_length, _ = self.tensegrity_robot.compute_cable_length(measure_cable)

            control, pos = self.pids[f"pid_{name}"].update_control_by_target_gait(
                curr_length,
                target,
                cable.rest_length
            )

            cable.update_rest_length(control.detach(), curr_length, dt)

            controls.append(control)

            # if name == 'spring_0':
            #     print(curr_length.squeeze())
            # del target
        self.tensegrity_robot.springs.update(self.tensegrity_robot.actuated_cables)
        next_state = super().step(curr_state,
                                  dt,
                                  external_forces,
                                  external_pts)

        return next_state, controls

    def step(self,
             curr_state: torch.Tensor,
             dt: Union[torch.Tensor, float],
             external_forces: Dict = None,
             external_pts: Dict = None,
             control_signals: Dict = None,
             target_lengths_dict: Dict = None,
             target_gait_dict: Dict = None) -> torch.Tensor:

        assert (control_signals is None
                or target_lengths_dict is None
                or target_gait_dict is None)

        if external_forces is None or external_pts is None:
            ext_f_dim = len(self.tensegrity_robot.rods) * 3
            size = (curr_state.shape[0], ext_f_dim, 1)
            external_forces = torch.zeros(size, dtype=self.sys_precision, device=curr_state.device)
            external_pts = torch.zeros(size, dtype=self.sys_precision, device=curr_state.device)

        if control_signals is not None:
            next_state = self.step_w_controls(curr_state,
                                              dt,
                                              external_forces,
                                              external_pts,
                                              controls_dict=control_signals)
        elif target_lengths_dict is not None:
            next_state, _ = self.step_w_target_lengths(curr_state,
                                                       dt,
                                                       external_forces,
                                                       external_pts,
                                                       target_lengths=target_lengths_dict)
        elif target_gait_dict is not None:
            next_state, _ = self.step_with_target_gait(curr_state,
                                                       dt,
                                                       external_forces,
                                                       external_pts,
                                                       target_gait_dict=target_gait_dict)
        else:
            next_state = super().step(curr_state,
                                      dt,
                                      external_forces,
                                      external_pts)

        return next_state

    def run_until_stable(self, dt, tol=1e-4, max_time=3):
        with torch.no_grad():
            time = 0.0
            curr_state = self.get_curr_state()
            num_bodies = len(self.tensegrity_robot.rods)
            vels = torch.ones(num_bodies * 3, dtype=self.sys_precision)
            states = [curr_state.clone()]

            while (torch.abs(vels) > tol).any():
                if time > max_time:
                    break
                #     raise Exception('Stability could not be reached within 5 seconds')
                self.update_state(curr_state)
                curr_state = self.step(curr_state, dt)
                states.append(curr_state.clone())

                time += dt
                vels = torch.hstack([
                    curr_state[:, i * 13 + 7: i * 13 + 10]
                    for i in range(num_bodies)
                ])

                print(time, torch.abs(vels).max())

            self.update_state(curr_state)

        print("Stabilization complete")

        return curr_state, states

    def reset_pids(self):
        for k, pid in self.pids.items():
            pid.reset()

    def reset_actuation(self):
        for k, cable in self.tensegrity_robot.actuated_cables.items():
            cable.reset_cable()

    def run_target_gait(self, dt, curr_state, target_gait_dict, time=0.0, max_steps=1e8):
        self.reset_pids()
        controls = torch.ones(len(target_gait_dict),
                              dtype=self.sys_precision)
        states = [curr_state.clone()]

        while (controls != 0.0).any() and len(states) < max_steps:
            # print(round(time, 3))
            time += dt
            curr_state, controls = self.step_with_target_gait(
                states[-1],
                dt,
                target_gait_dict=target_gait_dict
            )

            lengths = [self.tensegrity_robot.compute_cable_length(c)[0].item() for c in
                       self.tensegrity_robot.actuated_cables.values()]
            combo = [(round(c.item(), 3), round(l, 3)) for c, l in zip(controls, lengths)]
            # print(combo)
            # print([c.actuation_length.item() for c in self.tensegrity_robot.actuated_cables.values()])
            # r = [c.rest_length.item() for c in self.tensegrity_robot.actuated_cables.values()]
            states.append(curr_state.clone())
            controls = torch.hstack(controls)

        return states, time

    def init_by_endpts(self, end_pts):
        self.tensegrity_robot.init_by_endpts(end_pts)

    def print_state(self, state: torch.Tensor, time: float) -> Dict:
        with torch.no_grad():
            def single_rod_end_pt(state, length):
                principal_axis = torch_quaternion.quat_as_rot_mat(state[..., 3:7, :])[..., :, 2:]
                end_pt1, end_pt2 = RodState.compute_end_pts_from_state(state[..., :7, :], principal_axis, length)
                #
                return end_pt1, end_pt2

            num_bodies = len(self.tensegrity_robot.rods)
            poses = [state[..., i * 13: i * 13 + 7, :] for i in range(num_bodies)]
            lin_vels = [state[..., i * 13 + 7: i * 13 + 10, :] for i in range(num_bodies)]
            ang_vels = [state[..., i * 13 + 10: i * 13 + 13, :] for i in range(num_bodies)]

            state_dict = {"time": round(time, 5)}
            state_dict['pos'] = torch.concat(poses, dim=1).squeeze().numpy().tolist()
            state_dict['lin_vel'] = torch.concat(lin_vels, dim=1).squeeze().numpy().tolist()
            state_dict['ang_vel'] = torch.concat(ang_vels, dim=1).squeeze().numpy().tolist()

            for i, rod in enumerate(self.tensegrity_robot.rods.values()):
                end_pt1, end_pt2 = single_rod_end_pt(poses[i], rod.length)
                state_dict[rod.name + "_end_pt1"] = end_pt1.squeeze().numpy().tolist()
                state_dict[rod.name + "_end_pt2"] = end_pt2.squeeze().numpy().tolist()

        return state_dict

    def grad_descent_rest_lengths(self):
        with torch.enable_grad():
            springs = [s for s in self.tensegrity_robot.springs.values()]
            params = []
            for s in springs[6:]:
                length, _ = self._compute_cable_length(s)
                params.append(s._rest_length - length)
            # for s in springs[6:7]:
            #     # length, _ = self._compute_cable_length(s)
            #     params.append(s.rest_length)
            params = torch.nn.ParameterList(params)
            # + [torch.rand(1) * 4000 for _ in range(1)])
            # print([r.detach().item() for r in params])
            # params = torch.nn.ParameterList([s._rest_length for s in springs[:]])
            #                                 + [springs[0]._rest_length, springs[6]._rest_length]
            #                                 + [springs[0].stiffness, springs[6].stiffness])

            # params = torch.nn.ParameterList([springs[0]._rest_length, springs[0].stiffness,
            #                                  springs[6]._rest_length, springs[6].stiffness])
            max_lengths = []
            # for i, s in enumerate(springs[:6]):
            #     s.actuation_length = params[i]
            for i, s in enumerate(springs[6:]):
                s._rest_length = params[i]
            # length, _ = self._compute_cable_length(s)
            # max_lengths.append(length.detach().item())

            # for s in springs[:6]:
            #     s._rest_length = params[-4]
            #     s.stiffness = params[-2]

            # for s in springs[6:9]:
            #     s._rest_length = params[-3]
            #     s.stiffness = params[-1]

            # short_passive_max_length = min([self._compute_cable_length(s) for s in springs[:6]],
            #                                key=lambda x: x[0])[0].item()
            #
            # long_passive_max_length = min([self._compute_cable_length(s) for s in springs[6:9]],
            #                               key=lambda x: x[0])[0].item()

            optimizer = torch.optim.Adam(params, lr=0.01)
            start_state = self.get_curr_state()
            best_max_vel = 100.0

            # print(start_state.flatten())

            for i in range(5000):
                # if i == 500:
                #     optimizer = torch.optim.Adam(params, lr=0.001)
                if i == 2000:
                    optimizer = torch.optim.Adam(params, lr=0.001)
                curr_state = start_state.clone()
                vels = []
                for _ in range(1):
                    curr_state = self.step(curr_state, 0.001)
                    curr_vel = torch.vstack([curr_state[0, 7:13], curr_state[0, 20:26], curr_state[0, 33:39]])
                    vels.append(curr_vel)
                # start_vel = torch.vstack([start_state[0, 7:10], start_state[0, 20:23], start_state[0, 33:36]])
                vels = torch.vstack(vels)

                loss = (vels ** 2).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                for j, p in enumerate(params[:6]):
                    p.data.clamp_(0.0, 3.0)
                # params[6].data.clamp_(0.0, max_length1.item())
                # params[7].data.clamp_(0.0, max_length2.item())

                # params[-4].data.clamp_(0.8, short_passive_max_length)
                # params[-3].data.clamp_(0.8, long_passive_max_length)
                # params[-2].data.clamp_(1000, 15000)
                # params[-1].data.clamp_(1000, 15000)

                if torch.abs(curr_vel).max().detach().item() < best_max_vel:
                    best_max_vel = torch.abs(curr_vel).max().item()
                    final_params = [r.detach().item() for r in params]
                print(i, torch.abs(curr_vel).sum().item(), torch.abs(curr_vel).max().item())

            # for i, s in enumerate(springs[:6]):
            #     delattr(s, "actuation_length")
            #     s.actuation_length = torch.tensor([[[final_params[i]]]], dtype=curr_state.dtype)
            # for s in springs[6:]:
            #     delattr(s, "_rest_length")
            #     s._rest_length = torch.tensor([[[final_params[-1]]]], dtype=curr_state.dtype)
            print(best_max_vel, final_params)