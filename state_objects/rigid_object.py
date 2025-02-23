from typing import Union, List

import torch

from state_objects.base_state_object import BaseStateObject
from utilities import torch_quaternion
from utilities.inertia_tensors import body_to_world_torch


class RigidBody(BaseStateObject):

    def __init__(self,
                 name: str,
                 mass: Union[float, int, torch.Tensor],
                 I_body: torch.Tensor,
                 pos: torch.Tensor,
                 rot_val: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64):
        super().__init__(name)

        self.mass = mass
        self.I_body = I_body
        self.I_body_inv = torch.linalg.inv(I_body)

        self.pos = pos

        self.linear_vel = linear_vel
        self.ang_vel = ang_vel

        # Fixed constraint
        self.fixed = False

        # Compute an initial quaternion and rotation matrix
        self.update_rot(rot_val)

        self.I_world_inv = body_to_world_torch(self.rot_mat, self.I_body_inv).reshape(-1, 3, 3)

        self.sites = {s: None for s in sites}
        self.sys_precision = sys_precision

    def move_tensors(self, device: str):
        self.mass = self.mass.to(device)
        self.I_body = self.I_body.to(device)
        self.I_body_inv = self.I_body_inv.to(device)

        self.pos = self.pos.to(device)
        self.quat = self.quat.to(device)
        self.linear_vel = self.linear_vel.to(device)
        self.ang_vel = self.ang_vel.to(device)

        self.rot_mat = self.rot_mat.to(device)

        for k, v in self.sites.items():
            self.sites[k] = v.to(device)

        return self

    def world_to_body_coords(self, world_coords):
        assert len(world_coords.shape) == len(self.pos.shape)

        rot_mat_inv = self.rot_mat.transpose(-1, -2)
        return torch.matmul(rot_mat_inv, world_coords - self.pos)

    def body_to_world_coords(self, body_coords):
        assert len(body_coords.shape) == len(self.pos.shape)

        return torch.matmul(self.rot_mat, body_coords) + self.pos

    def update_state(self, pos, linear_vel, rot_val, ang_vel):
        self.pos = pos
        self.linear_vel = linear_vel
        self.ang_vel = ang_vel

        self.update_rot(rot_val)
        self.I_world_inv = body_to_world_torch(self.rot_mat, self.I_body_inv).reshape(-1, 3, 3)

    def update_rot(self, rot_val):
        self.quat = rot_val
        self.rot_mat = torch_quaternion.quat_as_rot_mat(self.quat).reshape(-1, 3, 3)

    def compute_gravity_force(self, gravity, num_batch: int):
        grav = self.mass * gravity.repeat(num_batch, 1, 1)
        grav = grav.reshape(num_batch, 3, -1)

        return grav

    def update_sites(self, site_name, relative_pos):
        self.sites[site_name] = relative_pos

    @property
    def state(self):
        state = torch.hstack([self.pos,
                              self.quat,
                              self.linear_vel,
                              self.ang_vel])
        return state

    @staticmethod
    def compute_init_quat_principal(principal_axis):
        principal_axis = principal_axis / principal_axis.norm(dim=1, keepdim=True)
        zeros = torch.zeros(principal_axis[..., 0, :].shape,
                            dtype=principal_axis.dtype,
                            device=principal_axis.device)

        # quaternion formula for rotation between z-axis and principal axis
        q = torch.concat([1 + principal_axis[..., 2, :],
                          -principal_axis[..., 1, :],
                          principal_axis[..., 0, :],
                          zeros],
                         dim=-1).reshape(-1, 4, 1)
        q = q / q.norm(dim=1, keepdim=True)

        return q.reshape(-1, 4, 1)
