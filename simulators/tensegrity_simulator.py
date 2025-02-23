from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

import torch
import xitorch.optimize
from torch.autograd import Variable

from actuation.pid import PID
from contact.collision_detector import get_detector
from contact.collision_response import CollisionResponseGenerator
from simulators.abstract_simulator import AbstractSimulator, rod_initializer
from state_objects.rods import RodState
from state_objects.springs import ActuatedCable, get_spring, SpringState
from state_objects.system_topology import SystemTopology
from robots.tensegrity import TensegrityRobot
from utilities import misc_utils, torch_quaternion
from utilities.tensor_utils import zeros


class TensegrityRobotSimulator(AbstractSimulator):
    tensegrity_robot: TensegrityRobot
    use_contact: bool
    contact_params: Dict
    collision_resp_gen: CollisionResponseGenerator

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

        self.collision_resp_gen = CollisionResponseGenerator()
        self.collision_resp_gen.set_contact_params('default', contact_params)

        self.pid = PID()

    def forward2(self, x, ctrls, dt, rest_lens, motor_speeds, gt_acc):
        params = torch.nn.ParameterList([x, ctrls])
        x, ctrls = params

        self.update_state(x)

        for i, c in enumerate(self.tensegrity_robot.actuated_cables.values()):
            c.actuation_length = c._rest_length - rest_lens[:, i: i + 1]
            c.motor.motor_state.omega_t = motor_speeds[:, i: i + 1]

        next_x = self.step_w_controls(x, dt, controls=ctrls)
        next_rest_lens = torch.hstack([c.rest_length for c in self.tensegrity_robot.actuated_cables.values()])
        next_motor_speeds = torch.hstack([c.motor.motor_state.omega_t
                                          for c in self.tensegrity_robot.actuated_cables.values()])

        old_vel = x.reshape(-1, 13, 1)[:, 7:].flatten()
        new_vel = next_x.reshape(-1, 13, 1)[:, 7:].flatten()

        acc = (new_vel - old_vel) / dt.flatten()
        loss = ((gt_acc - acc) ** 2).mean()
        loss.backward()

        x_grad = x.grad
        ctrls_grad = ctrls.grad

        return next_x, next_rest_lens, next_motor_speeds, x_grad, ctrls_grad

    def forward(self,
                curr_state,
                target_gaits,
                dt,
                rest_lens,
                motor_speeds,
                last_error, cum_error, done_flag,
                min_length, range_, tol,
                first_step):
        for i, c in enumerate(self.tensegrity_robot.actuated_cables.values()):
            c.actuation_length = c._rest_length - rest_lens[:, i: i + 1]
            c.motor.motor_state.omega_t = motor_speeds[:, i: i + 1]

        self.pid.min_length = min_length
        self.pid.RANGE = range_
        self.pid.tol = tol

        next_state, controls, rest_lengths, motor_speeds, last_error, cum_error, done_flag = (
            self.step_with_target_gait(
                curr_state,
                dt,
                target_gaits,
                last_error, cum_error, done_flag,
                first_step
            ))

        return next_state, controls, rest_lengths, motor_speeds, last_error, cum_error, done_flag

    def run_target_gaits(self,
                         curr_state,
                         dt,
                         in_rest_lengths,
                         in_motor_speeds,
                         pid_params,
                         target_gaits):
        first_step = torch.tensor(1., dtype=torch.float64)
        for i, c in enumerate(self.tensegrity_robot.actuated_cables.values()):
            c.actuation_length = c._rest_length - in_rest_lengths[:, i: i + 1]
            c.motor.motor_state.omega_t = in_motor_speeds[:, i: i + 1]

        min_length, range_, tol = pid_params
        self.pid.min_length = min_length
        self.pid.RANGE = range_
        self.pid.tol = tol

        last_error = torch.zeros_like(min_length)
        cum_error = torch.zeros_like(min_length)
        done_flag = torch.zeros_like(min_length, dtype=torch.bool)

        ctrls = torch.ones_like(min_length)

        states = [curr_state.clone()]
        rest_lengths = [in_rest_lengths.clone()]
        motor_speeds = [in_motor_speeds.clone()]
        controls = []

        while (ctrls != 0.0).any():
            curr_state, ctrls, rest_lens, omega_t, last_error, cum_error, done_flag = \
                self.step_with_target_gait(curr_state, dt, target_gaits, last_error, cum_error, done_flag, first_step)

            states.append(curr_state.clone())
            rest_lengths.append(rest_lens.clone())
            motor_speeds.append(omega_t.clone())
            controls.append(ctrls.clone())

            first_step = torch.tensor(0., dtype=torch.float64)

        return states, rest_lengths, motor_speeds, controls

    def step_with_target_gait(self,
                              curr_state: torch.Tensor,
                              dt: Union[torch.Tensor, float],
                              target_gaits: torch.Tensor,
                              last_error, cum_error, done_flag,
                              first_step):
        self.update_state(curr_state)

        curr_length, rest_lengths = [], []
        for i in range(target_gaits.shape[0]):
            # measure_name = self.tensegrity_robot.cable_map[name]
            # measure_cable = self.tensegrity_robot.springs[measure_name]
            cable = self.tensegrity_robot.actuated_cables[f"spring_{i}"]
            l, _ = self.tensegrity_robot.compute_cable_length(cable)
            rest_lengths.append(cable.rest_length)
            curr_length.append(l)

        curr_length = torch.vstack(curr_length)
        rest_lengths = torch.vstack(rest_lengths)

        controls, last_error, cum_error, done_flag, = self.pid.update_control_by_target_gait_fn(
            curr_length,
            target_gaits,
            rest_lengths,
            last_error, cum_error, done_flag,
            first_step
        )
        controls = controls.reshape(1, -1, 1)

        for i, c in enumerate(self.tensegrity_robot.actuated_cables.values()):
            c.update_rest_length(controls[:, i: i + 1], curr_length[i: i + 1], dt)

        next_state = super().step(curr_state, dt)

        rest_lengths = torch.hstack([c.rest_length
                                     for c in self.tensegrity_robot.actuated_cables.values()])
        motor_speeds = torch.hstack([c.motor.motor_state.omega_t
                                     for c in self.tensegrity_robot.actuated_cables.values()])

        return next_state, controls, rest_lengths, motor_speeds, last_error, cum_error, done_flag

    def move_tensors(self, device):
        self.tensegrity_robot = self.tensegrity_robot.move_tensors(device)
        self.gravity = self.gravity.to(device)

        self.collision_resp_gen.move_tensors(device)

        return self

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

    def compute_forces(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        net_spring_forces, spring_forces, acting_pts \
            = self.tensegrity_robot.compute_spring_forces()
        gravity_forces = torch.hstack([rod.mass * self.gravity
                                       for rod in self.tensegrity_robot.rods])
        net_forces = net_spring_forces + gravity_forces

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
        for i, rod in enumerate(self.tensegrity_robot.rods):
            start, end = i * 3, (i + 1) * 3

            lin_acc = net_force[:, start: end] / rod.mass
            ang_acc = torch.matmul(rod.I_world_inv,
                                   net_torque[:, start: end])

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

        q = self.tensegrity_robot.quat.reshape(-1, 4, 1)
        quat = torch_quaternion.update_quat(q, ang_vel, dt)

        next_state = torch.hstack([pos, quat, lin_vel, ang_vel]).reshape(-1, 13 * n, 1)

        return next_state

    def get_curr_state(self) -> torch.Tensor:
        return torch.hstack([
            torch.hstack([
                rod.pos,
                rod.quat,
                rod.linear_vel,
                rod.ang_vel,
            ])
            for rod in self.tensegrity_robot.rods
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
        self.update_state(curr_state)

    def update_system_topology(self) -> None:
        self.tensegrity_robot.update_system_topology()

    def compute_contact_deltas(self,
                               pre_next_state: torch.Tensor,
                               dt: Union[torch.Tensor, float]
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        delta_v, delta_w, toi = [], [], []
        for i, rod in enumerate(self.tensegrity_robot.rods):
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
                         delta_v: torch.Tensor,
                         delta_w: torch.Tensor,
                         toi) -> torch.Tensor:
        rods = self.tensegrity_robot.rods
        pre_next_state_ = pre_next_state.reshape(-1, 13, 1)

        lin_vel = pre_next_state_[:, 7:10, ...] + delta_v
        ang_vel = pre_next_state_[:, 10:, ...] + delta_w
        pos = torch.vstack([r.pos for r in rods]) + dt * lin_vel - delta_v * toi
        quat = torch_quaternion.update_quat(pre_next_state_[:, 3:7], delta_w, dt - toi)

        next_state_ = torch.hstack([pos, quat, lin_vel, ang_vel])
        next_state = next_state_.reshape(-1, len(rods) * 13, 1)

        return next_state

    def _compute_cable_length(self, cable: SpringState):
        end_pt0 = self.tensegrity_robot.system_topology.sites_dict[cable.end_pts[0]]
        end_pt1 = self.tensegrity_robot.system_topology.sites_dict[cable.end_pts[1]]

        x_dir = end_pt1 - end_pt0
        length = x_dir.norm(dim=1, keepdim=True)
        x_dir = x_dir / length

        return length, x_dir

    def step_w_controls(self,
                        curr_state: torch.Tensor,
                        dt: Union[torch.Tensor, float],
                        controls: torch.Tensor = None) -> torch.Tensor:
        for i in range(controls.shape[1]):
            name = f"spring_{i}"
            control = controls[:, i: i + 1]

            measure_name = self.tensegrity_robot.cable_map[name]
            measure_cable = self.tensegrity_robot.springs[measure_name]
            cable = self.tensegrity_robot.springs[name]

            curr_length, _ = self.tensegrity_robot.compute_cable_length(measure_cable)
            cable.update_rest_length(control, curr_length, dt)

        self.tensegrity_robot.springs.update(self.tensegrity_robot.actuated_cables)
        next_state = super().step(curr_state,
                                  dt)

        return next_state

    def step(self,
             curr_state: torch.Tensor,
             dt: Union[torch.Tensor, float],
             controls: torch.Tensor = None) -> torch.Tensor:

        next_state = self.step_w_controls(curr_state,
                                          dt,
                                          controls=controls)
        return next_state

    def reset_actuation(self):
        for k, cable in self.tensegrity_robot.actuated_cables.items():
            cable.reset_cable()

    def init_by_endpts(self, end_pts):
        self.tensegrity_robot.init_by_endpts(end_pts)
