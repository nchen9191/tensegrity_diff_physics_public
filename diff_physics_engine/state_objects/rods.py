from typing import List, Optional

import torch

from diff_physics_engine.state_objects.composite_body import CompositeBody
from diff_physics_engine.state_objects.primitive_shapes import Cylinder, SphereState, HollowCylinder
from diff_physics_engine.state_objects.rigid_object import RigidBody
from utilities import torch_quaternion
from utilities.inertia_tensors import (solid_sphere_body, parallel_axis_offset,
                                       hollow_cylinder_body)


class RodState(Cylinder):
    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64,
                 rot_val=None):
        """
        :param name: unique name
        :param end_pts: initial end points
        :param linear_vel: initial linear velocity
        :param ang_vel: initial angular velocity
        :param radius: radius
        :param mass: mass
        :param sys_precision: System precision
        """
        super().__init__(name,
                         end_pts,
                         linear_vel,
                         ang_vel,
                         radius,
                         mass,
                         sites,
                         sys_precision,
                         rot_val)

    def compute_angle_site_matching(self, curr_sites, new_sites, new_pos, new_prin):
        inv_quat_fn = lambda q, v: torch_quaternion.rotate_vec_quat(
            torch_quaternion.inverse_unit_quat(q), v)
        ref_curr_sites = {k: inv_quat_fn(self.quat, v - self.pos)
                          for k, v in curr_sites.items()}

        q1 = self.compute_init_quat_principal(new_prin)
        ref_new_sites = {k: inv_quat_fn(q1, v - new_pos)
                         for k, v in new_sites.items()}

        angles = []
        for k in ref_curr_sites.keys():
            ref_site = ref_curr_sites[k]
            new_site = ref_new_sites[k]

            angle = torch.linalg.vecdot(ref_site[:, :2], new_site[:, :2], dim=1)
            angle = torch.clamp(angle, -1, 1)
            angle = torch.arccos(angle)
            angles.append(angle)

        avg_angle = sum(angles) / len(angles)
        return avg_angle

    def compute_quat_site_matching(self, curr_sites, new_sites, new_pos, new_prin):
        inv_quat_fn = lambda q, v: torch_quaternion.rotate_vec_quat(
            torch_quaternion.inverse_unit_quat(q), v)
        ref_curr_sites = {k: inv_quat_fn(self.quat, v - self.pos)
                          for k, v in curr_sites.items()}

        q1 = self.compute_init_quat_principal(new_prin)
        ref_new_sites = {k: inv_quat_fn(q1, v - new_pos)
                         for k, v in new_sites.items()}

        angles = []
        for k in ref_curr_sites.keys():
            ref_site = ref_curr_sites[k]
            new_site = ref_new_sites[k]

            angle = torch.linalg.vecdot(ref_site[:, :2], new_site[:, :2], dim=1)
            angle = torch.clamp(angle, -1, 1)
            angle = torch.arccos(angle)
            angles.append(angle)

        avg_angle = self.compute_angle_site_matching(curr_sites, new_sites, new_pos, new_prin)
        q2 = torch.tensor([torch.cos(avg_angle / 2), 0, 0, torch.sin(avg_angle / 2)],
                          dtype=new_pos.dtype, device=new_pos.device)

        q = torch_quaternion.quat_prod(q1, q2)

        return q

    @classmethod
    def init_to_torch_tensors(cls,
                              config: dict,
                              sys_precision: torch.dtype = torch.float64):
        """
        See __init__()
        """
        end_pts = torch.concat([torch.tensor(end_pt, dtype=sys_precision).reshape(-1, 3, 1)
                                for end_pt in config['end_pts']], dim=2)
        linear_vel = torch.tensor(config['linear_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        ang_vel = torch.tensor(config['ang_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        radius = torch.tensor(config['radius'], dtype=sys_precision)
        mass = torch.tensor(config['mass'], dtype=sys_precision)

        instance = cls(config['name'],
                       end_pts,
                       linear_vel,
                       ang_vel,
                       radius,
                       mass,
                       config['sites'],
                       sys_precision=sys_precision)

        instance.fixed = config['fixed'] if "fixed" in config else False

        return instance

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

    @staticmethod
    def estimate_pos_from_endpts(end_pts, dtype=torch.float64):
        """

        :param end_pts:
        :param dtype:
        :return:
        """
        if not isinstance(end_pts, torch.Tensor):
            end_pts = torch.tensor(end_pts, dtype=dtype)

        com = (end_pts[..., 1, :] + end_pts[..., 0, :]).reshape(-1, 3, 1) / 2.

        principal_axis = (end_pts[..., 1, :] - end_pts[..., 0, :]).reshape(-1, 3, 1)
        unit_prin_axis = principal_axis / principal_axis.norm(dim=1)

        quat = torch.hstack([1 + unit_prin_axis[2],
                             -unit_prin_axis[1],
                             unit_prin_axis[0],
                             torch.zeros((principal_axis.shape[0], 1),
                                         dtype=dtype)]
                            ).reshape(-1, 4, 1)
        quat /= quat.norm(dim=1)
        pose = torch.hstack([com, quat])

        return pose

    def update_state(self, pos, linear_vel, rot_val, ang_vel):
        super().update_state(pos, linear_vel, rot_val, ang_vel)
        self.end_pts = self._compute_end_pts()


class RodSphericalEndCaps3(RodState):
    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sphere_radius: torch.Tensor,
                 sphere_mass: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64):
        super().__init__(name,
                         end_pts,
                         linear_vel,
                         ang_vel,
                         radius,
                         mass,
                         sites,
                         sys_precision=sys_precision)
        self.mass += 2 * sphere_mass

        self.sphere_radius = sphere_radius
        self.sphere_mass = sphere_mass
        self.sphere_offset = torch.tensor([0, 0, self.length / 2.0], dtype=sys_precision)

        sphere_inertia = solid_sphere_body(sphere_mass, sphere_radius, sys_precision)
        sphere_inertia = parallel_axis_offset(sphere_inertia, sphere_mass, self.sphere_offset)
        self.I_body += 2 * sphere_inertia
        self.I_body_inv = torch.linalg.inv(self.I_body)

    @classmethod
    def init_to_torch_tensors(cls,
                              config: dict,
                              sys_precision: torch.dtype = torch.float64):
        end_pts = torch.concat([torch.tensor(end_pt, dtype=sys_precision).reshape(-1, 3, 1)
                                for end_pt in config['end_pts']], dim=2)
        linear_vel = torch.tensor(config['linear_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        ang_vel = torch.tensor(config['ang_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        radius = torch.tensor(config['radius'], dtype=sys_precision)
        mass = torch.tensor(config['mass'], dtype=sys_precision)
        sphere_radius = torch.tensor(config['sphere_radius'], dtype=sys_precision)
        sphere_mass = torch.tensor(config['sphere_mass'], dtype=sys_precision)

        return cls(config['name'],
                   end_pts,
                   radius,
                   mass,
                   sphere_radius,
                   sphere_mass,
                   linear_vel,
                   ang_vel,
                   config['sites'],
                   sys_precision=sys_precision)

    def move_tensors(self, device):
        super().move_tensors(device)
        self.sphere_radius.to(device)
        self.sphere_mass.to(device)


class RodSphericalEndCaps(CompositeBody):
    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sphere_radius: torch.Tensor,
                 sphere_mass: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64,
                 rot_val: Optional[torch.Tensor] = None):
        if rot_val is None:
            prin_axis = end_pts[:, :, 1:2] - end_pts[:, :, 0:1]
            prin_axis /= prin_axis.norm(dim=1, keepdim=True)
            rot_val = RigidBody.compute_init_quat_principal(prin_axis)

        rod = RodState(name + "_rod",
                       end_pts,
                       linear_vel,
                       ang_vel,
                       radius,
                       mass,
                       [],
                       sys_precision,
                       rot_val)
        sphere1 = SphereState(name + "_sphere1",
                              end_pts[:, :, 0:1],
                              linear_vel.clone(),
                              ang_vel.clone(),
                              sphere_radius,
                              sphere_mass,
                              rod.get_principal_axis(),
                              [],
                              sys_precision,
                              rot_val)
        sphere2 = SphereState(name + "_sphere2",
                              end_pts[:, :, 1:],
                              linear_vel.clone(),
                              ang_vel.clone(),
                              sphere_radius,
                              sphere_mass,
                              rod.get_principal_axis(),
                              [],
                              sys_precision,
                              rot_val)

        rigid_bodies = [rod, sphere1, sphere2]
        self.length = rod.length
        self.sphere_radius = sphere_radius
        self.end_pts = rod.end_pts

        super().__init__(name,
                         linear_vel,
                         ang_vel,
                         rot_val,
                         rigid_bodies,
                         sites,
                         rigid_bodies,
                         sys_precision)

    @classmethod
    def init_min_rod(cls,
                     end_pts,
                     rod_radius,
                     rod_mass,
                     sphere_mass,
                     sphere_radius,
                     dtype=torch.float32,
                     quat=None):
        return cls(
            "comp_rod",
            torch.tensor(end_pts, dtype=dtype),
            torch.tensor(rod_radius, dtype=dtype),
            torch.tensor(rod_mass, dtype=dtype),
            torch.tensor(sphere_radius, dtype=dtype),
            torch.tensor(sphere_mass, dtype=dtype),
            torch.zeros((1, 3, 1), dtype=dtype),
            torch.zeros((1, 3, 1), dtype=dtype),
            [],
            dtype,
            quat
        )

    @classmethod
    def init_to_torch_tensors(cls,
                              config: dict,
                              sys_precision: torch.dtype = torch.float64):
        end_pts = torch.concat([torch.tensor(end_pt, dtype=sys_precision).reshape(-1, 3, 1)
                                for end_pt in config['end_pts']], dim=2)
        linear_vel = torch.tensor(config['linear_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        ang_vel = torch.tensor(config['ang_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        radius = torch.tensor(config['radius'], dtype=sys_precision)
        mass = torch.tensor(config['mass'], dtype=sys_precision)
        sphere_radius = torch.tensor(config['sphere_radius'], dtype=sys_precision)
        sphere_mass = torch.tensor(config['sphere_mass'], dtype=sys_precision)

        quat = torch.tensor(config['quat'], dtype=sys_precision).reshape(-1, 4, 1) \
            if 'quat' in config else None

        instance = cls(config['name'],
                       end_pts,
                       radius,
                       mass,
                       sphere_radius,
                       sphere_mass,
                       linear_vel,
                       ang_vel,
                       config['sites'],
                       sys_precision=sys_precision,
                       rot_val=quat)
        instance.fixed = config['fixed'] if "fixed" in config else False
        return instance

    def update_state(self, pos, linear_vel, rot_val, ang_vel):
        super().update_state(pos, linear_vel, rot_val, ang_vel)
        rod = self.rigid_bodies[self.name + "_rod"]
        # sphere1 = self.rigid_bodies[self.name + "_sphere1"]
        # sphere2 = self.rigid_bodies[self.name + "_sphere2"]
        #
        # rod.update_state(pos, linear_vel, rot_val, ang_vel)
        #
        # body_vec = rod.end_pts[0] - pos
        # lin_vel1 = linear_vel + torch.cross(ang_vel, body_vec, dim=1)
        #
        # body_vec = rod.end_pts[1] - pos
        # lin_vel2 = linear_vel + torch.cross(ang_vel, body_vec, dim=1)
        #
        # sphere1.update_state(rod.end_pts[0], lin_vel1, rot_val, ang_vel)
        # sphere2.update_state(rod.end_pts[1], lin_vel2, rot_val, ang_vel)

        self.end_pts = rod.end_pts

    def update_state_by_endpts(self, end_pts, lin_vel, ang_vel):
        rod = self.rigid_bodies[f'{self.name}_rod']
        # prev_prin = rod.get_principal_axis()
        curr_prin = end_pts[1] - end_pts[0]
        curr_prin = curr_prin / curr_prin.norm(dim=1, keepdim=True)
        # q_diff = torch_quaternion.compute_q_btwn_vecs(prev_prin, curr_prin)

        pos = (end_pts[0] + end_pts[1]) / 2.0
        # quat = torch_quaternion.quat_prod(q_diff, self.quat)
        quat = RodState.compute_init_quat_principal(curr_prin)

        self.update_state(pos, lin_vel, quat, ang_vel)


class RodCylinderMotorsSphericalEndCaps(RodSphericalEndCaps):

    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sphere_radius: torch.Tensor,
                 sphere_mass: torch.Tensor,
                 motor_radius: torch.Tensor,
                 motor_mass: torch.Tensor,
                 motor_length: torch.Tensor,
                 motor_offset: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64,
                 rot_val: Optional[torch.Tensor] = None):
        super().__init__(name,
                         end_pts,
                         radius,
                         mass,
                         sphere_radius,
                         sphere_mass,
                         linear_vel,
                         ang_vel,
                         sites,
                         sys_precision=sys_precision,
                         rot_val=rot_val)

        self.mass += 2 * motor_mass
        self.motor_radius = motor_radius

        prin_axis = end_pts[:, :, 1:2] - end_pts[:, :, 0:1]
        prin_axis /= prin_axis.norm(dim=1, keepdim=True)

        motor_e1_dist = (motor_length / 2 + motor_offset) * prin_axis
        motor_e2_dist = (-motor_length / 2 + motor_offset) * prin_axis

        ang_vel_comp = torch.cross(ang_vel, motor_offset * prin_axis)

        motor1 = HollowCylinder(f'{self.name}_motor1',
                                torch.concat([self.pos - motor_e1_dist,
                                              self.pos - motor_e2_dist],
                                             dim=2),
                                linear_vel - ang_vel_comp,
                                ang_vel.clone(),
                                motor_radius,
                                radius,
                                motor_mass,
                                [],
                                sys_precision)

        motor2 = HollowCylinder(f'{self.name}_motor2',
                                torch.concat([self.pos + motor_e2_dist,
                                              self.pos + motor_e1_dist],
                                             dim=2),
                                linear_vel + ang_vel_comp,
                                ang_vel.clone(),
                                motor_radius,
                                radius,
                                motor_mass,
                                [],
                                sys_precision)

        self.rigid_bodies[motor1.name] = motor1
        self.rigid_bodies[motor2.name] = motor2

        motor_inertia = hollow_cylinder_body(motor_mass,
                                             motor_length,
                                             motor_radius,
                                             radius,
                                             sys_precision)
        # motor_inertia = cylinder_body(motor_mass,
        #                               motor_length,
        #                               motor_radius,
        #                               sys_precision)
        offset_body = torch.matmul(self.rot_mat.transpose(1, 2),
                                   motor_offset * prin_axis)
        motor_inertia = parallel_axis_offset(motor_inertia,
                                             motor_mass,
                                             offset_body)
        self.I_body += 2 * motor_inertia
        self.I_body_inv = torch.linalg.inv(self.I_body)

        self.rigid_bodies_body_vecs = self._compute_body_vecs(self.pos)
        self.body_vecs_tensor = torch.vstack(list(self.rigid_bodies_body_vecs.values()))

    @classmethod
    def init_to_torch_tensors(cls,
                              config: dict,
                              sys_precision: torch.dtype = torch.float64):
        end_pts = torch.concat([torch.tensor(end_pt, dtype=sys_precision).reshape(-1, 3, 1)
                                for end_pt in config['end_pts']], dim=2)
        linear_vel = torch.tensor(config['linear_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        ang_vel = torch.tensor(config['ang_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        radius = torch.tensor(config['radius'], dtype=sys_precision)
        mass = torch.tensor(config['mass'], dtype=sys_precision)
        sphere_radius = torch.tensor(config['sphere_radius'], dtype=sys_precision)
        sphere_mass = torch.tensor(config['sphere_mass'], dtype=sys_precision)

        motor_mass = torch.tensor(config['motor_mass'], dtype=sys_precision)
        motor_radius = torch.tensor(config['motor_radius'], dtype=sys_precision)
        motor_offset = torch.tensor(config['motor_offset'], dtype=sys_precision)
        motor_length = torch.tensor(config['motor_length'], dtype=sys_precision)

        quat = torch.tensor(config['quat'], dtype=sys_precision).reshape(-1, 4, 1) \
            if 'quat' in config else None

        instance = cls(config['name'],
                       end_pts,
                       radius,
                       mass,
                       sphere_radius,
                       sphere_mass,
                       motor_radius,
                       motor_mass,
                       motor_length,
                       motor_offset,
                       linear_vel,
                       ang_vel,
                       config['sites'],
                       sys_precision=sys_precision,
                       rot_val=quat)
        instance.fixed = config['fixed'] if "fixed" in config else False
        return instance


class RodHousingMotorsSphericalEndCaps(RodCylinderMotorsSphericalEndCaps):

    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sphere_radius: torch.Tensor,
                 sphere_mass: torch.Tensor,
                 motor_radius: torch.Tensor,
                 motor_mass: torch.Tensor,
                 motor_length: torch.Tensor,
                 motor_offset: torch.Tensor,
                 housing_mass: torch.Tensor,
                 housing_length: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List,
                 sys_precision: torch.dtype = torch.float64,
                 rot_val: Optional[torch.Tensor] = None):
        super().__init__(name,
                         end_pts,
                         radius,
                         mass,
                         sphere_radius,
                         sphere_mass,
                         motor_radius,
                         motor_mass,
                         motor_length,
                         motor_offset,
                         linear_vel,
                         ang_vel,
                         sites,
                         sys_precision,
                         rot_val)

        self.mass += housing_mass

        prin_axis = end_pts[:, :, 1:2] - end_pts[:, :, 0:1]
        prin_axis /= prin_axis.norm(dim=1, keepdim=True)

        housing_e1 = self.pos - 0.5 * housing_length * prin_axis
        housing_e2 = self.pos + 0.5 * housing_length * prin_axis

        ang_vel_comp = torch.cross(ang_vel, motor_offset * prin_axis)

        housing = HollowCylinder(f'{self.name}_housing',
                                 torch.concat([housing_e1, housing_e2], dim=2),
                                 linear_vel - ang_vel_comp,
                                 ang_vel.clone(),
                                 motor_radius,
                                 radius,
                                 housing_mass,
                                 [],
                                 sys_precision)

        self.rigid_bodies[housing.name] = housing

        housing_inertia = hollow_cylinder_body(housing_mass,
                                               housing_length,
                                               motor_radius,
                                               radius,
                                               sys_precision)

        self.I_body += housing_inertia
        self.I_body_inv = torch.linalg.inv(self.I_body)

        self.rigid_bodies_body_vecs = self._compute_body_vecs(self.pos)
        self.body_vecs_tensor = torch.vstack(list(self.rigid_bodies_body_vecs.values()))

    @classmethod
    def init_to_torch_tensors(cls,
                              config: dict,
                              sys_precision: torch.dtype = torch.float64):
        end_pts = torch.concat([torch.tensor(end_pt, dtype=sys_precision).reshape(-1, 3, 1)
                                for end_pt in config['end_pts']], dim=2)
        linear_vel = torch.tensor(config['linear_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        ang_vel = torch.tensor(config['ang_vel'], dtype=sys_precision).reshape(-1, 3, 1)
        radius = torch.tensor(config['radius'], dtype=sys_precision)
        mass = torch.tensor(config['mass'], dtype=sys_precision)
        sphere_radius = torch.tensor(config['sphere_radius'], dtype=sys_precision)
        sphere_mass = torch.tensor(config['sphere_mass'], dtype=sys_precision)

        motor_mass = torch.tensor(config['motor_mass'], dtype=sys_precision)
        motor_radius = torch.tensor(config['motor_radius'], dtype=sys_precision)
        motor_offset = torch.tensor(config['motor_offset'], dtype=sys_precision)
        motor_length = torch.tensor(config['motor_length'], dtype=sys_precision)

        housing_mass = torch.tensor(config['housing_mass'], dtype=sys_precision)
        housing_length = torch.tensor(config['housing_length'], dtype=sys_precision)

        quat = torch.tensor(config['quat'], dtype=sys_precision).reshape(-1, 4, 1) \
            if 'quat' in config else None

        instance = cls(config['name'],
                       end_pts,
                       radius,
                       mass,
                       sphere_radius,
                       sphere_mass,
                       motor_radius,
                       motor_mass,
                       motor_length,
                       motor_offset,
                       housing_mass,
                       housing_length,
                       linear_vel,
                       ang_vel,
                       config['sites'],
                       sys_precision=sys_precision,
                       rot_val=quat)
        instance.fixed = config['fixed'] if "fixed" in config else False
        return instance
