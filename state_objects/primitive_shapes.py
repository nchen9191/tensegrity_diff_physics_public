from typing import List, Optional

from state_objects.rigid_object import RigidBody
from utilities import torch_quaternion
from utilities.inertia_tensors import *


class Cylinder(RigidBody):

    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64,
                 rot_val: Optional[torch.Tensor] = None):
        self.shape = "cylinder"
        self._body_verts = None
        self._faces = None

        self.end_pts = end_pts
        if isinstance(end_pts, torch.Tensor):
            self.end_pts = end_pts.reshape(-1, 3, 2)
            self.end_pts = [self.end_pts[:, :, :1], self.end_pts[:, :, 1:2]]

        self.radius = radius
        self.length = (self.end_pts[0] - self.end_pts[1]).norm(dim=1).squeeze()  # compute length

        linear_vel = linear_vel.reshape(-1, 3, 1)
        pos = (self.end_pts[0] + self.end_pts[1]) / 2.0  # compute pos from end points
        ang_vel = ang_vel.reshape(-1, 3, 1)

        # Compute an initial quaternion and rotation matrix
        if rot_val is None:
            rot_val = self._compute_init_quat()

        I_body = cylinder_body(mass, self.length, self.radius, sys_precision)

        super().__init__(name,
                         mass,
                         I_body,
                         pos,
                         rot_val,
                         linear_vel,
                         ang_vel,
                         sites,
                         sys_precision)

    @classmethod
    def init_min(cls, mass: float, radius: float, end_pts, dtype: torch.dtype = torch.float32):
        return cls(
            "cylinder",
            end_pts,
            torch.zeros((1, 3, 1), dtype=dtype),
            torch.zeros((1, 3, 1), dtype=dtype),
            torch.tensor(radius, dtype=dtype),
            torch.tensor(mass, dtype=dtype),
            [],
            dtype
        )

    def move_tensors(self, device: str):
        super(Cylinder, self).move_tensors(device)
        self.radius = self.radius.to(device)
        self.length = self.length.to(device)

        self.end_pts[0] = self.end_pts[0].to(device)
        self.end_pts[1] = self.end_pts[1].to(device)

    def get_center(self):
        return self.pos

    def get_principal_axis(self):
        """
        Method to get principal axis
        :return:
        """
        # z-axis aligned with rod cylinder axis along length
        return self.rot_mat[..., :, 2:]

    def _compute_init_quat(self) -> torch.Tensor:
        """
        Computes initial quaternion
        :return: quaternion tensor
        """
        # Initial quat orientation is the shortest arc rotation between world frame z-axis and initial principal axis
        principal_axis = self.end_pts[1] - self.end_pts[0]
        q = RigidBody.compute_init_quat_principal(principal_axis)

        return q

    def _compute_end_pts(self) -> List[torch.Tensor]:
        """
        Internal method to compute end points
        :return: End point tensors
        """
        principle_axis_vec = self.get_principal_axis()
        end_pts = self.compute_end_pts_from_state(self.pos[..., :3, :],
                                                  principle_axis_vec,
                                                  self.length)
        # end_pts = torch.concat(end_pts, dim=-1)

        return end_pts

    @staticmethod
    def compute_principal_axis(quat):
        """
        Computes principal axis from state input

        :param state: State (pos1, pos2, pos3, q1, q2, q3, q4, lin_v1, lin_v2, lin_v3, ang_v1, ang_v2, ang_v3)
        :return: principal axises
        """
        return torch_quaternion.quat_as_rot_mat(quat)[..., :, 2:]

    @staticmethod
    def compute_end_pts_from_state(rod_pos_state, principal_axis, rod_length):
        """
        :param rod_pos_state: (x, y, z, quat.w, quat.x, quat.y, quat.z)
        :param principal_axis: tensor of vector(s)
        :param rod_length: length of rod
        :return: ((x1, y1, z1), (x2, y2, z2))
        """
        # Get position
        pos = rod_pos_state[:, :3, ...]

        # Compute half-length vector from principal axis
        half_length_vec = rod_length * principal_axis / 2

        # End points are +/- of half-length vector from COM
        end_pt1 = pos - half_length_vec
        end_pt2 = pos + half_length_vec

        return [end_pt1, end_pt2]

    def update_state(self, pos, linear_vel, rot_val, ang_vel):
        super().update_state(pos, linear_vel, rot_val, ang_vel)
        self.end_pts = self._compute_end_pts()


class HollowCylinder(Cylinder):
    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 outer_radius: torch.Tensor,
                 inner_radius: torch.Tensor,
                 mass: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64):
        super().__init__(name,
                         end_pts,
                         linear_vel,
                         ang_vel,
                         outer_radius,
                         mass,
                         sites,
                         sys_precision)

        self.inner_radius = inner_radius

        self.I_body = hollow_cylinder_body(mass,
                                           self.length,
                                           outer_radius,
                                           inner_radius,
                                           sys_precision)

        self.I_body_inv = torch.linalg.inv(self.I_body)


class SphereState(RigidBody):

    def __init__(self,
                 name: str,
                 center: torch.Tensor,
                 linear_vel: Optional[torch.Tensor],
                 ang_vel: Optional[torch.Tensor],
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 principal_axis: Optional[torch.Tensor],
                 sites: List,
                 sys_precision: torch.dtype = torch.float64,
                 rot_val: Optional[torch.Tensor] = None):
        self.shape = "sphere"

        self.radius = radius

        linear_vel = linear_vel.reshape(-1, 3, 1)
        ang_vel = ang_vel.reshape(-1, 3, 1)

        if rot_val is None:
            rot_val = RigidBody.compute_init_quat_principal(principal_axis)

        self._body_verts = None
        self._faces = None

        super().__init__(name,
                         mass,
                         solid_sphere_body(mass, self.radius, sys_precision),
                         center,
                         rot_val,
                         linear_vel,
                         ang_vel,
                         sites,
                         sys_precision)

    def move_tensors(self, device: str):
        super(SphereState, self).move_tensors(device)
        self.radius = self.radius.to(device)


class Ground(RigidBody):

    def __init__(self, ground_z=0.0, sys_precision=torch.float64):
        self.shape = "ground"

        mass = torch.tensor(torch.inf, dtype=sys_precision)
        I_body = torch.diag(torch.tensor([torch.inf, torch.inf, torch.inf], dtype=sys_precision))
        pos = torch.tensor([0, 0, ground_z], dtype=sys_precision).reshape(1, -1, 1)
        quat = torch.tensor([1, 0, 0, 0], dtype=sys_precision).reshape(1, -1, 1)
        linear_vel = torch.tensor([0, 0, 0], dtype=sys_precision).reshape(1, -1, 1)
        ang_vel = torch.tensor([0, 0, 0], dtype=sys_precision).reshape(1, -1, 1)

        # self.ground_net_force = torch.tensor([0, 0, 0], dtype=sys_precision).reshape(1, -1, 1)

        super().__init__("ground", mass, I_body, pos, quat, linear_vel, ang_vel, [], sys_precision)

    def repeat_state(self, batch_size: int):
        self.reset_batch_size()

        self.pos = self.pos.repeat(batch_size, 1, 1)
        self.quat = self.quat.repeat(batch_size, 1, 1)
        self.rot_mat = self.rot_mat.repeat(batch_size, 1, 1)
        self.linear_vel = self.linear_vel.repeat(batch_size, 1, 1)
        self.ang_vel = self.ang_vel.repeat(batch_size, 1, 1)

        # self.ground_net_force = self.ground_net_force.repeat(batch_size, 1, 1)

    def reset_batch_size(self):
        self.pos = self.pos[0:1]
        self.quat = self.quat[0:1]
        self.rot_mat = self.rot_mat[0:1]
        self.linear_vel = self.linear_vel[0:1]
        self.ang_vel = self.ang_vel[0:1]
