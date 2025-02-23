from typing import Dict, Union

import torch
from torch.linalg import vecdot, matmul, cross, norm

from contact.collision_detector import *
from state_objects.base_state_object import BaseStateObject
from state_objects.primitive_shapes import Ground
from utilities import torch_quaternion
from utilities.inertia_tensors import body_to_world_torch
from utilities.tensor_utils import zeros


class ContactParameters(BaseStateObject):

    def __init__(self,
                 restitution: torch.Tensor,
                 baumgarte: torch.Tensor,
                 friction: torch.Tensor,
                 friction_damping: torch.Tensor,
                 rolling_friction: torch.Tensor = 0.0):
        super().__init__("contact_params")
        self.restitution = restitution
        self.baumgarte = baumgarte
        self.friction = friction
        self.friction_damping = friction_damping
        # self.rolling_friction = rolling_friction

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def move_tensors(self, device: str):
        self.restitution = self.restitution.to(device)
        self.baumgarte = self.baumgarte.to(device)
        self.friction = self.friction.to(device)
        self.friction_damping = self.friction_damping.to(device)

        return self


class CollisionResponseGenerator(BaseStateObject):

    def __init__(self, ground_z: float = 0, sys_precision: torch.dtype = torch.float64):
        super().__init__("collision_response_generator")
        self.ground = Ground(ground_z)
        self.contact_params = ContactParameters(**{
            "restitution": torch.tensor(0.7, dtype=sys_precision),
            "baumgarte": torch.tensor(0.1, dtype=sys_precision),
            "friction": torch.tensor(0.5, dtype=sys_precision),
            "friction_damping": torch.tensor(0.5, dtype=sys_precision)
        })

    def set_contact_params(self, key: str, contact_params: Dict):
        self.contact_params.update(**contact_params)

    def move_tensors(self, device: str):
        self.ground.move_tensors(device)
        self.contact_params.move_tensors(device)

    def resolve_contact(self,
                        rigid_body1: RigidBody,
                        rigid_body2: RigidBody,
                        next_state1: torch.Tensor,
                        next_state2: torch.Tensor,
                        delta_t: Union[float, torch.Tensor],
                        collision_detector: CollisionDetector):
        return self.compute_delta_vel_contact(rigid_body1,
                                              rigid_body2,
                                              next_state1,
                                              next_state2,
                                              delta_t,
                                              collision_detector)

    def resolve_contact_ground(self,
                               rigid_body1: RigidBody,
                               next_state1: torch.Tensor,
                               delta_t: Union[float, torch.Tensor],
                               collision_detector: CollisionDetector):
        return self.compute_delta_vel_contact(self.ground,
                                              rigid_body1,
                                              self.ground.state,
                                              next_state1,
                                              delta_t,
                                              collision_detector)

    def compute_delta_vel_contact(self,
                                  rigid_body1: RigidBody,
                                  rigid_body2: RigidBody,
                                  next_state1: torch.Tensor,
                                  next_state2: torch.Tensor,
                                  delta_t: Union[float, torch.Tensor],
                                  collision_detector: CollisionDetector):
        baumgarte, friction_mu, restit_coeff, friction_damping = self.get_contact_params(rigid_body1, rigid_body2)

        toi = zeros((next_state1.shape[0], 1, 1), ref_tensor=next_state1)

        dummy_tensor = zeros((next_state1.shape[0], 3, 1), ref_tensor=next_state1)
        delta_v1, delta_w1 = dummy_tensor.clone(), dummy_tensor.clone()
        delta_v2, delta_w2 = dummy_tensor.clone(), dummy_tensor.clone()
        impulse_pos, impulse_vel, impulse_friction = dummy_tensor.clone(), dummy_tensor.clone(), dummy_tensor.clone()

        # Check current state collision
        curr_state1, curr_state2 = rigid_body1.state, rigid_body2.state
        detection_params_t = collision_detector.detect(curr_state1,
                                                       curr_state2,
                                                       rigid_body1,
                                                       rigid_body2)

        for has_collision_t, contact1_t, contact2_t, normal in detection_params_t:
            curr_state1 = torch.concat([curr_state1[:, :7], next_state1[:, 7:]], dim=1)
            curr_state2 = torch.concat([curr_state2[:, :7], next_state2[:, 7:]], dim=1)
            params = self.compute_contact_params(curr_state1,
                                                 curr_state2,
                                                 rigid_body1,
                                                 rigid_body2,
                                                 contact1_t,
                                                 contact2_t,
                                                 normal)

            mass_norm, mass_tan, rel_vel_norm, rel_vel_tan, tangent, r1, r2, inv_inertia1, inv_inertia2 = params

            # Apply spring-mass
            baum = self.baumgarte_contact_impulse(mass_norm,
                                                  contact1_t,
                                                  contact2_t,
                                                  baumgarte,
                                                  delta_t,
                                                  normal)
            impulse_pos = impulse_pos + baum * has_collision_t

            # Apply impulse
            reaction_imp = self.reaction_impulse(mass_norm,
                                                 restit_coeff,
                                                 rel_vel_norm,
                                                 normal)
            impulse_vel = impulse_vel + reaction_imp * has_collision_t

            # Friction for current time step
            impulse_normal = impulse_vel + impulse_pos
            fric = self.friction_impulse(rel_vel_tan,
                                         tangent,
                                         impulse_normal,
                                         friction_mu,
                                         friction_damping,
                                         mass_tan)
            impulse_friction = impulse_friction + fric * has_collision_t

            # impulse_total = impulse_normal + impulse_friction
            dv1, dv2, dw1, dw2 = self.compute_delta_vels(impulse_normal,
                                                         impulse_friction,
                                                         r1,
                                                         r2,
                                                         rigid_body1.mass,
                                                         rigid_body2.mass,
                                                         inv_inertia1,
                                                         inv_inertia2)
            delta_v1 = delta_v1 + dv1 * has_collision_t
            delta_v2 = delta_v2 + dv2 * has_collision_t
            delta_w1 = delta_w1 + dw1 * has_collision_t
            delta_w2 = delta_w2 + dw2 * has_collision_t

        has_collision_t = torch.stack([d[0] for d in detection_params_t], dim=-1).max(dim=-1).values

        # Else, Check next state collision
        detection_params_tp = collision_detector.detect(next_state1,
                                                        next_state2,
                                                        rigid_body1,
                                                        rigid_body2)

        for has_collision, contact1_tp, contact2_tp, normal in detection_params_tp:
            has_collision_tp = torch.logical_and(has_collision, ~has_collision_t)

            params = self.compute_contact_params(next_state1,
                                                 next_state2,
                                                 rigid_body1,
                                                 rigid_body2,
                                                 contact1_tp,
                                                 contact2_tp,
                                                 normal)
            mass_norm, mass_tan, rel_vel_norm, rel_vel_tan, tangent, r1, r2, inv_inertia1, inv_inertia2 = params

            reaction_imp = self.reaction_impulse(mass_norm,
                                                 restit_coeff,
                                                 rel_vel_norm,
                                                 normal)
            # print(has_collision_tp, impulse_vel[has_collision_tp], reaction_imp[has_collision_tp])
            impulse_vel = impulse_vel + reaction_imp * has_collision_tp

            # toi
            pen_depth = norm(contact2_tp - contact1_tp, dim=1).unsqueeze(2)
            zero = zeros(delta_t.shape, ref_tensor=delta_t)
            toi += torch.clamp(delta_t * has_collision_tp + pen_depth * has_collision_tp
                                                / (rel_vel_norm * has_collision_tp + 1e-12),
                                                zero * has_collision_tp,
                                                delta_t * has_collision_tp)

            # Friction
            impulse_normal = impulse_vel
            fric = self.friction_impulse(rel_vel_tan,
                                         tangent,
                                         impulse_normal,
                                         friction_mu,
                                         friction_damping,
                                         mass_tan)
            impulse_friction = impulse_friction + fric * has_collision_tp

            dv1, dv2, dw1, dw2 = self.compute_delta_vels(impulse_normal,
                                                         impulse_friction,
                                                         r1,
                                                         r2,
                                                         rigid_body1.mass,
                                                         rigid_body2.mass,
                                                         inv_inertia1,
                                                         inv_inertia2)
            delta_v1 = delta_v1 + dv1 * has_collision_tp
            delta_v2 = delta_v2 + dv2 * has_collision_tp
            delta_w1 = delta_w1 + dw1 * has_collision_tp
            delta_w2 = delta_w2 + dw2 * has_collision_tp

        return delta_v1, delta_w1, delta_v2, delta_w2, toi

    def get_contact_params(self,
                           rigid_body1: RigidBody,
                           rigid_body2: RigidBody):
        restit_coeff = self.contact_params.restitution
        baumgarte = self.contact_params.baumgarte
        friction_mu = self.contact_params.friction
        friction_damping = self.contact_params.friction_damping

        return baumgarte, friction_mu, restit_coeff, friction_damping

    def compute_delta_vels(self,
                           impulse_normal: torch.Tensor,
                           impulse_tangent: torch.Tensor,
                           r1: torch.Tensor,
                           r2: torch.Tensor,
                           mass1: torch.Tensor,
                           mass2: torch.Tensor,
                           inv_inertia1: torch.Tensor,
                           inv_inertia2: torch.Tensor):
        impulse_total = impulse_normal + impulse_tangent

        delta_v1 = -impulse_total / mass1
        delta_v2 = impulse_total / mass2

        delta_w1 = matmul(inv_inertia1, cross(r1, -impulse_total, dim=1))
        delta_w2 = matmul(inv_inertia2, cross(r2, impulse_total, dim=1))

        return delta_v1, delta_v2, delta_w1, delta_w2

    def compute_contact_params(self,
                               state1: torch.Tensor,
                               state2: torch.Tensor,
                               rigid_body1: RigidBody,
                               rigid_body2: RigidBody,
                               contact1: torch.Tensor,
                               contact2: torch.Tensor,
                               normal: torch.Tensor):
        mass1, mass2 = rigid_body1.mass, rigid_body2.mass

        pos1, pos2 = state1[:, :3], state2[:, :3]
        quat1, quat2 = state1[:, 3:7], state2[:, 3:7]
        vel1, vel2 = state1[:, 7:10], state2[:, 7:10]
        ang_vel1, ang_vel2 = state1[:, 10:], state2[:, 10:]

        rot_mat1 = torch_quaternion.quat_as_rot_mat(quat1)
        rot_mat2 = torch_quaternion.quat_as_rot_mat(quat2)

        r1, r2 = contact1 - pos1, contact2 - pos2
        inertia_inv1 = body_to_world_torch(rot_mat1, rigid_body1.I_body_inv).reshape(-1, 3, 3)
        inertia_inv2 = body_to_world_torch(rot_mat2, rigid_body2.I_body_inv).reshape(-1, 3, 3)
        mass_norm = self.compute_contact_mass(mass1, mass2, inertia_inv1, inertia_inv2, r1, r2, normal)

        rel_vel_c = self.compute_rel_vel(vel1, vel2, ang_vel1, ang_vel2, r1, r2)
        tangent = rel_vel_c - vecdot(rel_vel_c, normal, dim=1).unsqueeze(2) * normal
        tangent /= norm(tangent + 1e-6, dim=1).unsqueeze(2)

        rel_vel_c_norm = self.compute_rel_vel_normal_comp(rel_vel_c, normal)
        rel_vel_c_tan = vecdot(rel_vel_c, tangent, dim=1).unsqueeze(2)

        mass_tan = self.compute_contact_mass(mass1, mass2, inertia_inv1, inertia_inv2, r1, r2, tangent)

        return mass_norm, mass_tan, rel_vel_c_norm, rel_vel_c_tan, tangent, r1, r2, inertia_inv1, inertia_inv2

    @staticmethod
    def reaction_impulse(mass_norm: torch.Tensor, restit_coeff: torch.Tensor,
                         rel_vel_c_normal: torch.Tensor, normal: torch.Tensor):
        impulse_vel = (-(1 + restit_coeff)
                       * rel_vel_c_normal
                       * mass_norm
                       * normal)

        return impulse_vel

    @staticmethod
    def baumgarte_contact_impulse(mass_norm: torch.Tensor, contact1: torch.Tensor, contact2: torch.Tensor,
                                  baumgarte: torch.Tensor, delta_t: torch.Tensor, normal: torch.Tensor):
        pen_depth = norm(contact2 - contact1, dim=1).unsqueeze(2)
        impulse_pos = (baumgarte
                       * pen_depth
                       * mass_norm
                       * normal
                       / delta_t)

        return impulse_pos

    @staticmethod
    def compute_rel_vel(vel1: torch.Tensor, vel2: torch.Tensor, ang_vel1: torch.Tensor, ang_vel2: torch.Tensor,
                        r1: torch.Tensor, r2: torch.Tensor):
        vel_c1 = vel1 + cross(ang_vel1, r1, dim=1)
        vel_c2 = vel2 + cross(ang_vel2, r2, dim=1)
        rel_vel_c = vel_c2 - vel_c1

        return rel_vel_c

    @staticmethod
    def compute_rel_vel_normal_comp(rel_vel: torch.Tensor, normal: torch.Tensor):
        v_c_norm = vecdot(rel_vel, normal, dim=1)
        v_c_norm = torch.clamp_max(v_c_norm, 0.)

        return v_c_norm.unsqueeze(2)

    @staticmethod
    def compute_contact_mass(mass1: torch.Tensor, mass2: torch.Tensor,
                             inv_inertia1: torch.Tensor, inv_inertia2: torch.Tensor,
                             r1: torch.Tensor, r2: torch.Tensor, dir_vec: torch.Tensor):
        mass_inv1 = vecdot(cross(matmul(inv_inertia1, cross(r1, dir_vec, dim=1)), r1, dim=1), dir_vec, dim=1)
        mass_inv2 = vecdot(cross(matmul(inv_inertia2, cross(r2, dir_vec, dim=1)), r2, dim=1), dir_vec, dim=1)
        mass_contact = torch.clamp_min((1 / mass1 + 1 / mass2 + mass_inv1 + mass_inv2) ** -1, 0.)

        return mass_contact.unsqueeze(2)

    @staticmethod
    def friction_impulse(rel_vel_tan: torch.Tensor, tangent: torch.Tensor,
                         impulse_normal: torch.Tensor, friction_mu: torch.Tensor,
                         friction_damping: torch.Tensor, mass_tangent: torch.Tensor):
        static_friction = mass_tangent * rel_vel_tan
        static_friction = static_friction * friction_damping
        max_friction = friction_mu * impulse_normal.norm(dim=1, keepdim=True)

        friction = -torch.minimum(static_friction, max_friction)

        impulse_tangent = tangent * friction

        return impulse_tangent
