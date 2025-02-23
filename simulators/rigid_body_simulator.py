from typing import Tuple, Union, Dict

import torch

from contact.collision_detector import get_detector
from contact.collision_response import CollisionResponseGenerator
from simulators.abstract_simulator import AbstractSimulator
from state_objects.rigid_object import RigidBody
from utilities import torch_quaternion


class RigidBodySimulator(AbstractSimulator):
    """
    Class for rod simulations
    """

    def __init__(self,
                 rigid_body: RigidBody,
                 gravity: torch.Tensor,
                 sys_precision: torch.dtype = torch.float64,
                 contact_params: Dict[str, Dict[str, torch.Tensor]] = dict(),
                 use_contact: bool = True):
        """
        :param rigid_body: RodState objects
        :param gravity: Gravity constant
        """
        super().__init__(sys_precision)

        self.rigid_body = rigid_body
        self.gravity = gravity

        self.collision_resp_gen = CollisionResponseGenerator()
        self.collision_resp_gen.set_contact_params('default', contact_params)

        self.sys_params[rigid_body.name] = self.rigid_body
        self.sys_params.update(**contact_params)

    def move_tensors(self, device: str):
        self.rigid_body.move_tensors(device)
        self.collision_resp_gen.move_tensors(device)
        self.gravity = self.gravity.to(device)

    def compute_forces(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        See parent documentation.
        """
        gravity_force = self.rigid_body.compute_gravity_force(self.gravity, self.rigid_body.pos.shape[0])

        forces = torch.concat([gravity_force], dim=2)
        acting_pts = torch.concat([self.rigid_body.pos], dim=-1)

        net_force = forces.sum(dim=-1).unsqueeze(-1)

        return net_force, forces, acting_pts

    def compute_accelerations(self, net_force, net_torque):
        """
        Compute rigid-body accelerations
        See parent documentation
        """

        # Linear acceleration of rigid body = sum(Forces) / mass
        lin_acc = net_force / self.rigid_body.mass

        # Angular acceleration of rigid body = (I^-1) * sum(Torques)
        ang_acc = torch.matmul(self.rigid_body.I_world_inv, net_torque)

        return lin_acc, ang_acc

    def time_integration(self, lin_acc, ang_acc, dt):
        """
        Semi-implicit euler

        See parent documentation.
        """
        # Semi-implicit euler step for velocity and position
        linear_vel = self.rigid_body.linear_vel + lin_acc * dt
        pos = self.rigid_body.pos + linear_vel * dt

        # Semi-implicit euler step for angular velocity
        ang_vel = self.rigid_body.ang_vel + ang_acc * dt

        # Angular velocity in quaternion format, semi-implicit euler on quaternion
        quat = self.update_quat(self.rigid_body.quat, ang_vel, dt)

        # Concat position, quaternion, linear velocity, and angular velocity to form next state
        next_state = torch.concat([pos, quat, linear_vel, ang_vel], dim=1)

        return next_state

    def compute_contact_deltas(self,
                               pre_next_state: torch.Tensor,
                               dt: Union[torch.Tensor, float]
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detector = get_detector(self.rigid_body, self.collision_resp_gen.ground)
        _, _, delta_v, delta_w, toi = self.collision_resp_gen.resolve_contact_ground(self.rigid_body,
                                                                                     pre_next_state,
                                                                                     dt,
                                                                                     detector)

        return delta_v, delta_w, toi

    def resolve_contacts(self,
                         pre_next_state: torch.Tensor,
                         dt: Union[torch.Tensor, float],
                         delta_v: torch.Tensor,
                         delta_w: torch.Tensor,
                         toi) -> torch.Tensor:
        lin_vel = pre_next_state[:, 7:10, ...] + delta_v
        ang_vel = pre_next_state[:, 10:, ...] + delta_w
        pos = self.rigid_body.pos + dt * lin_vel - delta_v * toi
        quat = self.update_quat(self.update_quat(self.rigid_body.quat, ang_vel, dt), -delta_w, toi)

        next_state = torch.hstack([pos, quat, lin_vel, ang_vel])

        return next_state

    @staticmethod
    def update_quat(quat, ang_vel, dt):
        zero = torch.zeros((ang_vel.shape[0], 1, 1), dtype=ang_vel.dtype, device=ang_vel.device)
        ang_vel_quat = torch.concat([zero, ang_vel], dim=1)
        ang_vel_quat *= 0.5 * dt
        quat = torch_quaternion.quat_prod(torch_quaternion.quat_exp(ang_vel_quat), quat)
        return quat

    def update_state(self, next_state: torch.Tensor) -> None:
        """
        Update state and other internal attribute from state
        """
        next_state = next_state.reshape(-1, next_state.shape[1], 1)
        pos = next_state[..., :3, :]
        quat = next_state[..., 3:7, :]
        linear_vel = next_state[..., 7:10, :]
        ang_vel = next_state[..., 10:, :]

        self.rigid_body.update_state(pos, linear_vel, quat, ang_vel)

    def get_xyz_pos(self, curr_state: torch.Tensor) -> torch.Tensor:
        """
        See parent documentation
        """
        return self.rigid_body.pos

    def get_curr_state(self) -> torch.Tensor:
        """
        See parent documentation.
        """
        return self.rigid_body.state

    def compute_aux_states(self, curr_state: torch.Tensor) -> None:
        """
        See parent documentation
        """
        self.update_state(curr_state)
