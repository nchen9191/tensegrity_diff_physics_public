from typing import List, Optional

import torch

from diff_physics_engine.state_objects.rigid_object import RigidBody
from utilities.inertia_tensors import solid_sphere_body


class SphereState(RigidBody):

    def __init__(self,
                 name: str,
                 center: torch.Tensor,
                 linear_vel: Optional[torch.Tensor],
                 ang_vel: Optional[torch.Tensor],
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 principal_axis: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64):
        self.shape = "sphere"
        self.radius = radius

        linear_vel = linear_vel.reshape(-1, 3, 1) if linear_vel is not None else None
        ang_vel = ang_vel.reshape(-1, 3, 1) if ang_vel is not None else None

        quat = RigidBody.compute_init_quat_principal(principal_axis)

        super().__init__(name,
                         mass,
                         solid_sphere_body(mass, self.radius, sys_precision),
                         center,
                         quat,
                         linear_vel,
                         ang_vel,
                         sites,
                         sys_precision)

    def move_tensors(self, device):
        super(SphereState, self).move_tensors(device)
        self.radius = self.radius.to(device)

