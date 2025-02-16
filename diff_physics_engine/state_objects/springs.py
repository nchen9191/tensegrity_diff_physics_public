from typing import Union, List, Tuple

import torch

from diff_physics_engine.actuation.dc_motor import DCMotor
from diff_physics_engine.state_objects.base_state_object import BaseStateObject
from utilities.tensor_utils import zeros


class SpringState(BaseStateObject):

    def __init__(self,
                 stiffness: torch.Tensor,
                 damping: torch.Tensor,
                 rest_length: torch.Tensor,
                 end_pts: Union[List, Tuple],
                 name: str):
        """
        :param stiffness: spring stiffness
        :param damping: spring damping coefficient
        :param rest_length: spring rest length
        :param end_pts: (end_pt1 site_name, end_pt2 site_name), site names should match whats in system topology
        :param name: unique name
        """
        super().__init__(name)

        self.stiffness = stiffness
        self.damping = damping
        self._rest_length = rest_length
        self.end_pts = end_pts

    def move_tensors(self, device):
        self.stiffness = self.stiffness.to(device)
        self.damping = self.damping.to(device)
        self._rest_length = self._rest_length.to(device)

    @classmethod
    def init_to_torch_tensors(cls,
                              stiffness: torch.Tensor,
                              damping: torch.Tensor,
                              rest_length: torch.Tensor,
                              end_pts: List,
                              name: str,
                              sys_precision: torch.dtype = torch.float64):
        """
        Method to instantiate spring to tensors with input that are not torch tensors
        """
        stiffness = torch.tensor(stiffness, dtype=sys_precision).reshape(1, 1, 1)
        damping = torch.tensor(damping, dtype=sys_precision).reshape(1, 1, 1)
        rest_length = torch.tensor(rest_length, dtype=sys_precision).reshape(1, 1, 1)

        return cls(stiffness, damping, rest_length, end_pts, name)

    @property
    def rest_length(self):
        return self._rest_length

    def compute_potential_energy(self, end_pt1, end_pt2, rest_length=None):
        if rest_length is None:
            rest_length = self.rest_length

        curr_len = self.compute_curr_length(end_pt1, end_pt2)
        dx2 = (rest_length - curr_len) ** 2
        energy = 0.5 * self.stiffness * dx2

        return energy

    def compute_curr_length(self, end_pt1, end_pt2):
        spring_pos_vec = end_pt2 - end_pt1
        spring_pos_len = spring_pos_vec.norm(dim=1, keepdim=True)

        return spring_pos_len

    def compute_force(self,
                      end_pt1: torch.Tensor,
                      end_pt2: torch.Tensor,
                      vel_1: torch.Tensor,
                      vel_2: torch.Tensor) -> torch.Tensor:
        """
        Computes a spring force with the equation F = stiffness * (curr len - rest len) - damping * relative velocity
        Force direction relative to (endpt2 - endpt1) vector

        :param end_pt1: One end point
        :param end_pt2: Other end point
        :param vel_1: Velocity of end_pt1
        :param vel_2: Velocity of end_pt2
        :return: Spring's force
        """
        # Compute spring direction
        spring_pos_vec = end_pt2 - end_pt1
        spring_pos_len = spring_pos_vec.norm(dim=1, keepdim=True)
        spring_pos_vec_unit = spring_pos_vec / spring_pos_len

        # Compute spring velocity
        rel_vel_1 = torch.linalg.vecdot(vel_1, spring_pos_vec_unit, dim=1).unsqueeze(2)
        rel_vel_2 = torch.linalg.vecdot(vel_2, spring_pos_vec_unit, dim=1).unsqueeze(2)

        # Compute spring force based on hooke's law and damping
        stiffness_mag = self.stiffness * (spring_pos_len - self.rest_length)

        damping_mag = self.damping * (rel_vel_1 - rel_vel_2)
        spring_force_mag = stiffness_mag - damping_mag

        spring_force = spring_force_mag * spring_pos_vec_unit

        return spring_force


class OneEndFixedSpringState(SpringState):
    def __init__(self,
                 stiffness: torch.Tensor,
                 damping: torch.Tensor,
                 rest_length: torch.Tensor,
                 fixed_end_pt_name: str,
                 free_end_pt_name: str,
                 name: str):
        self.fixed_end = fixed_end_pt_name
        self.free_end = free_end_pt_name
        end_pts = [fixed_end_pt_name, free_end_pt_name]

        super().__init__(stiffness, damping, rest_length, end_pts, name)

    def compute_force(self,
                      end_pt1: torch.Tensor,
                      end_pt2: torch.Tensor,
                      vel_1: torch.Tensor,
                      vel_2: torch.Tensor = None) -> torch.Tensor:
        vel_2 = torch.zeros(vel_1.shape, dtype=vel_1.dtype)
        spring_force = super().compute_force(end_pt1, end_pt2, vel_1, vel_2)

        return spring_force

    @classmethod
    def init_to_torch_tensors(cls,
                              stiffness: torch.Tensor,
                              damping: torch.Tensor,
                              rest_length: torch.Tensor,
                              end_pts: List,
                              name: str,
                              sys_precision: torch.dtype = torch.float64):
        """
        See parent documentation for other params.
        :param end_pts: List of end pt site names. Assumes first is fixed and second is free
        """

        stiffness = torch.tensor(stiffness, dtype=sys_precision).reshape(1, 1, 1)
        damping = torch.tensor(damping, dtype=sys_precision).reshape(1, 1, 1)
        rest_length = torch.tensor(rest_length, dtype=sys_precision).reshape(1, 1, 1)

        return cls(stiffness, damping, rest_length, end_pts[0], end_pts[1], name)


class Cable(SpringState):

    def compute_potential_energy(self, end_pt1, end_pt2, rest_length=None):
        if rest_length is None:
            rest_length = self.rest_length

        curr_len = self.compute_curr_length(end_pt1, end_pt2)
        dx2 = torch.clamp_min(curr_len - rest_length, 0.0) ** 2
        energy = 0.5 * self.stiffness * dx2

        return energy

    def compute_force(self,
                      end_pt1: torch.Tensor,
                      end_pt2: torch.Tensor,
                      vel_1: torch.Tensor,
                      vel_2: torch.Tensor,
                      pull_only=True) -> torch.Tensor:
        """
        Computes a spring force with the equation F = stiffness * (curr len - rest len) - damping * relative velocity
        Force direction relative to (endpt2 - endpt1) vector

        :param end_pt1: One end point
        :param end_pt2: Other end point
        :param vel_1: Velocity of end_pt1
        :param vel_2: Velocity of end_pt2
        :return: Spring's force
        """
        # Compute spring direction
        spring_pos_vec = end_pt2 - end_pt1
        spring_pos_len = spring_pos_vec.norm(dim=1, keepdim=True)
        spring_pos_vec_unit = spring_pos_vec / spring_pos_len

        # Compute spring velocity
        rel_vel_1 = torch.linalg.vecdot(vel_1, spring_pos_vec_unit, dim=1).unsqueeze(2)
        rel_vel_2 = torch.linalg.vecdot(vel_2, spring_pos_vec_unit, dim=1).unsqueeze(2)

        # Compute spring force based on hooke's law and damping
        stiffness_mag = self.stiffness * (spring_pos_len - self.rest_length)
        damping_mag = self.damping * (rel_vel_1 - rel_vel_2)

        if pull_only:
            stiffness_mag = torch.clamp_min(stiffness_mag, 0.0)
            # damping_mag = torch.clamp_max(damping_mag, 0.0)

        spring_force_mag = stiffness_mag - damping_mag

        spring_force = spring_force_mag * spring_pos_vec_unit

        return spring_force


class ActuatedCable(Cable):

    def __init__(self,
                 stiffness,
                 damping,
                 rest_length,
                 end_pts,
                 name,
                 winch_r,
                 min_winch_r=0.01,
                 max_winch_r=0.07,
                 sys_precision=torch.float64,
                 motor=None,
                 motor_speed=0.6,
                 init_act_length=0.0):
        super().__init__(stiffness,
                         damping,
                         rest_length,
                         end_pts,
                         name)
        motor_speed = torch.tensor(motor_speed, dtype=sys_precision)
        self.motor = DCMotor(motor_speed) if motor is None else motor
        self.init_act_length = torch.tensor(init_act_length, dtype=sys_precision)
        self.actuation_length = self.init_act_length.clone().reshape(1, 1, 1)
        self.dl = 0
        self.min_winch_r = torch.tensor(min_winch_r, dtype=sys_precision)
        self.max_winch_r = torch.tensor(max_winch_r, dtype=sys_precision)
        self._winch_r = self._set_winch_r(winch_r)

    def _set_winch_r(self, winch_r):
        assert self.min_winch_r <= winch_r <= self.max_winch_r

        if not isinstance(winch_r, torch.Tensor):
            winch_r = torch.tensor(winch_r, dtype=self.sys_precision)

        delta = self.max_winch_r - self.min_winch_r
        winch_r = torch.logit((winch_r - self.min_winch_r) / delta)

        return winch_r

    @classmethod
    def init_to_torch_tensors(cls,
                              stiffness: torch.Tensor,
                              damping: torch.Tensor,
                              rest_length: torch.Tensor,
                              end_pts: List,
                              name: str,
                              winch_r,
                              motor_speed,
                              sys_precision: torch.dtype = torch.float64,
                              init_act_length=0.0):
        """
        Method to instantiate spring to tensors with input that are not torch tensors
        """
        stiffness = torch.tensor(stiffness, dtype=sys_precision).reshape(1, 1, 1)
        damping = torch.tensor(damping, dtype=sys_precision).reshape(1, 1, 1)
        rest_length = torch.tensor(rest_length, dtype=sys_precision).reshape(1, 1, 1)
        winch_r = torch.tensor(winch_r, dtype=sys_precision).reshape(1, 1, 1)

        return cls(stiffness,
                   damping,
                   rest_length,
                   end_pts,
                   name,
                   winch_r,
                   sys_precision=sys_precision,
                   motor_speed=motor_speed,
                   init_act_length=init_act_length)

    def move_tensors(self, device):
        super().move_tensors(device)
        self.motor = self.motor.move_tensors(device)
        self.actuation_length = self.actuation_length.to(device)
        self.init_act_length = self.init_act_length.to(device)
        self._winch_r = self._winch_r.to(device)
        self.min_winch_r = self.min_winch_r.to(device)
        self.max_winch_r = self.max_winch_r.to(device)

    @property
    def winch_r(self):
        winch_r_range = self.max_winch_r - self.min_winch_r
        dwinch_r = torch.sigmoid(self._winch_r) * winch_r_range

        winch_r = dwinch_r + self.min_winch_r
        return winch_r

    @property
    def rest_length(self):
        if self.actuation_length is None:
            return self._rest_length

        rest_length = self._rest_length - self.actuation_length
        return rest_length

    def update_rest_length(self,
                           control,
                           cable_length,
                           dt):
        if self.actuation_length is None:
            self.actuation_length = zeros(cable_length.shape,
                                          ref_tensor=cable_length)
        self.dl = self.motor.compute_cable_length_delta(control,
                                                        self.winch_r,
                                                        dt)
        actuation_length = (self.actuation_length
                            + self.dl * self.rest_length / cable_length)
        # actuation_length = self.actuation_length + self.dl
        self.actuation_length = torch.clamp_max(actuation_length,
                                                self._rest_length)
        self.dl = self.dl - (actuation_length - self.actuation_length)

    def reset_cable(self):
        self.actuation_length = self.init_act_length.clone()
        self.motor.reset_omega_t()

    def compute_force(self,
                      end_pt1: torch.Tensor,
                      end_pt2: torch.Tensor,
                      vel_1: torch.Tensor,
                      vel_2: torch.Tensor,
                      pull_only=True) -> torch.Tensor:
        spring_force = super().compute_force(end_pt1,
                                             end_pt2,
                                             vel_1,
                                             vel_2,
                                             True)
        return spring_force


def get_spring(spring_type):
    if spring_type.lower() == 'cable':
        return Cable
    elif spring_type.lower() == 'actuated_cable':
        return ActuatedCable
    elif spring_type.lower() == 'fixed_spring':
        return OneEndFixedSpringState
    else:
        return SpringState
