from collections import OrderedDict

import torch

from diff_physics_engine.state_objects.rigid_object import RigidBody
from utilities import tensor_utils
from utilities.inertia_tensors import parallel_axis_offset


class CompositeBody(RigidBody):

    def __init__(self,
                 name,
                 linear_vel,
                 ang_vel,
                 rot_val,
                 rigid_bodies,
                 sites,
                 contact_checks=None,
                 sys_precision=torch.float64):
        self.rigid_bodies = self.init_rigid_bodies_dict(rigid_bodies)

        self.contact_checks = contact_checks if contact_checks \
            else [body for body in self.rigid_bodies.values()]

        mass = sum([
            body.mass
            for body in self.rigid_bodies.values()
        ])

        com = torch.stack(
            [body.pos * body.mass
             for body in self.rigid_bodies.values()],
            dim=-1
        ).sum(dim=-1) / mass

        self.update_rot(rot_val)

        I_body = self._compute_inertia_tensor(com)

        self.rigid_bodies_body_vecs = self._compute_body_vecs(com)
        self.body_vecs_tensor = torch.vstack(list(self.rigid_bodies_body_vecs.values()))

        super().__init__(name,
                         mass,
                         I_body,
                         com,
                         rot_val,
                         linear_vel,
                         ang_vel,
                         sites,
                         sys_precision)

    def init_rigid_bodies_dict(self, rigid_bodies):
        rigid_body_dict = OrderedDict()
        for body in rigid_bodies:
            rigid_body_dict[body.name] = body

        return rigid_body_dict

    def _compute_inertia_tensor(self, com):
        rot_mat_inv = self.rot_mat.transpose(1, 2)
        I_body_total = torch.zeros((3, 3), dtype=com.dtype)

        for body in self.rigid_bodies.values():
            offset_world = body.pos - com
            offset_body = torch.matmul(rot_mat_inv, offset_world)
            I_body = parallel_axis_offset(body.I_body, body.mass, offset_body)
            I_body_total += I_body

        I_body_total = torch.diag(torch.diag(I_body_total, 0))
        return I_body_total

    def compute_body_offset_inertia(self, body_name):
        body = self.rigid_bodies[body_name]

        offset_body = self.rigid_bodies_body_vecs[body_name]
        I_body = parallel_axis_offset(body.I_body, body.mass, offset_body)

        return I_body

    def _compute_body_vecs(self, com):
        body_vecs = {}
        rot_mat_inv = self.rot_mat.transpose(1, 2)

        for body in self.rigid_bodies.values():
            offset_world = body.pos - com
            offset_body = torch.matmul(rot_mat_inv, offset_world)
            body_vecs[body.name] = offset_body

        return body_vecs

    def move_tensors(self, device):
        super().move_tensors(device)
        self.body_vecs_tensor = self.body_vecs_tensor.to(device)
        for body in self.rigid_bodies.values():
            body.move_tensors(device)
            body_vec = self.rigid_bodies_body_vecs[body.name]
            self.rigid_bodies_body_vecs[body.name] = body_vec.to(device)

        return self

    def update_state(self, pos, linear_vel, rot_val, ang_vel):
        super().update_state(pos, linear_vel, rot_val, ang_vel)
        for name, body in self.rigid_bodies.items():
            body_vec = self.rigid_bodies_body_vecs[name]
            world_vec = torch.matmul(self.rot_mat, body_vec)
            world_vec += pos
            lin_vel = linear_vel + torch.cross(ang_vel, body_vec, dim=1)
            body.update_state(world_vec, lin_vel, rot_val, ang_vel)

    def get_inner_poses(self):
        pos = tensor_utils.interleave_tensors(
            *[body.pos for body in self.rigid_bodies.values()]
        )
        quat = tensor_utils.interleave_tensors(
            *[body.quat for body in self.rigid_bodies.values()]
        )
        poses = torch.hstack([pos, quat])

        return poses

    def get_inner_vels(self):
        lin_vel = tensor_utils.interleave_tensors(
            *[body.linear_vel for body in self.rigid_bodies.values()]
        )
        ang_vel = tensor_utils.interleave_tensors(
            *[body.ang_vel for body in self.rigid_bodies.values()]
        )
        vels = torch.hstack([lin_vel, ang_vel])

        return vels