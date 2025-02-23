import torch

from state_objects.composite_body import CompositeBody

from state_objects.rigid_object import RigidBody
from utilities import torch_quaternion


class CollisionDetector:

    @staticmethod
    def detect(state1, state2, rigid_body1, rigid_body2):
        """
        Normal standardized to always pointing from rigid_body1 to rigid_body2
        :param state1:
        :param state2:
        :param rigid_body1:
        :param rigid_body2:
        :return:
        """
        raise NotImplementedError


class GroundSphereDetector(CollisionDetector):

    @staticmethod
    def detect(ground_state, sphere_state, ground, sphere):
        pos = sphere_state[:, :3, ...]
        ground_z = ground_state[:, 2, ...]

        normal = torch.zeros_like(pos)
        normal[:, 2] = 1.
        radius = normal * sphere.radius

        min_pts = pos - radius
        ground_contacts = min_pts.clone()
        ground_contacts[:, 2, :] = ground_z

        has_collision = min_pts[:, 2] <= ground_z

        return [(has_collision.flatten(), ground_contacts, min_pts, normal)]


class CylinderGroundDetector(CollisionDetector):

    @staticmethod
    def detect(ground_state, state, ground, cylinder):
        ground_z = ground_state[:, 2, ...]

        rot_mat1 = torch_quaternion.quat_as_rot_mat(state[:, 3:7, ...])
        principal_axis = rot_mat1[:, :, 2:]
        end_pts = cylinder.compute_end_pts_from_state(state, principal_axis, cylinder.length)
        end_pts = torch.concat(end_pts, dim=2)

        min_indices = torch.argmin(end_pts[:, 2:, :], dim=2).flatten()
        min_endpt = end_pts[torch.arange(0, end_pts.shape[0]), :, min_indices].unsqueeze(2)

        normal = torch.zeros_like(state[:, :3])
        normal[:, 2] = 1.

        out_vec = torch.linalg.cross(normal, principal_axis, dim=1)
        r = torch.linalg.cross(principal_axis, out_vec, dim=1)
        r /= torch.linalg.norm(r + 1e-6, dim=1).unsqueeze(2) / cylinder.radius
        min_pts = min_endpt - r

        ground_contacts = min_pts.clone()
        ground_contacts[:, 2, :] = ground_z

        has_collision = min_pts[:, 2, ...] <= ground_z

        return [(has_collision.flatten(), ground_contacts, min_pts, normal)]


class CompositeBodyGroundDetector(CollisionDetector):

    @staticmethod
    def detect(ground_state, composite_state2, ground: RigidBody, composite_body2: RigidBody):
        rot_mat2 = torch_quaternion.quat_as_rot_mat(composite_state2[:, 3:7, ...])

        body_collisions = []

        for rigid_body2 in composite_body2.rigid_bodies.values():
            body_offset2 = composite_body2.rigid_bodies_body_vecs[rigid_body2.name]
            world_offset2 = torch.matmul(rot_mat2, body_offset2)
            body_state2 = composite_state2.clone()
            body_state2[:, :3, ...] += world_offset2

            detector = get_detector(rigid_body2, ground)

            state1, state2 = (ground_state, body_state2)
            body1, body2 = (ground, rigid_body2)

            detect_params = detector.detect(state1, state2, body1, body2)
            body_collisions.extend(detect_params)

        return body_collisions


def get_detector(body1: RigidBody, body2: RigidBody):
    detector_dict = {
        "composite_ground": CompositeBodyGroundDetector,
        "cylinder_ground": CylinderGroundDetector,
        "sphere_ground": GroundSphereDetector,
    }

    shapes = [body1.shape.lower(), body2.shape.lower()]
    return detector_dict["_".join(shapes)]
