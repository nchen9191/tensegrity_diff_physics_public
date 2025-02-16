from typing import Tuple, Union, Dict

import torch

import diff_physics_engine.state_objects.rods as rods
from diff_physics_engine.state_objects.base_state_object import BaseStateObject


class AbstractSimulator(BaseStateObject):
    """
    Abstract class for any simulator to extend
    """

    def __init__(self, sys_precision: torch.dtype = torch.float64):
        super().__init__("simulator")
        self.sys_precision = sys_precision
        self.sys_params = {}

    def set_sys_precision(self, precision: torch.dtype) -> None:
        """
        Set system precision

        :param precision: data type for precision
        """
        self.sys_precision = precision

    def compute_forces(self, external_forces: torch.Tensor, external_pts: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to compute rigid body forces

        :param external_forces: Applied external/active forces
        :param external_pts: Points where external_forces are applied. Need to align with external_forces.
        :return: Net force, list of forces, and acting points of forces
        """
        pass

    def compute_torques(self, forces: torch.Tensor, body_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute torques from forces and body vector
        torques = torch.linalg.cross(body_vecs, forces, dim=1)

        # Get net torque by summing
        net_torque = torques.sum(axis=2).unsqueeze(2)

        return net_torque, torques

    def compute_accelerations(self, net_force: torch.Tensor, net_torque: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute accelerations based on net forces and net torques (and internal attribute values)
        :return: Linear acceleration and angular acceleration
        """
        pass

    def time_integration(self, lin_acc: torch.Tensor, ang_acc: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Method to step state forward

        :param lin_acc: linear acceleration
        :param ang_acc: angular acceleration
        :param dt: delta t
        :return: next state
        """
        pass

    def get_xyz_pos(self, curr_state: torch.Tensor) -> torch.Tensor:
        """
        Method to get the xyz coordinates from a state vector

        :param curr_state: Current state
        :return: xyz tensor
        """
        pass

    def get_curr_state(self) -> torch.Tensor:
        """
        Method to get current state
        :return: Current state tensor
        """
        pass

    def update_state(self, next_state: torch.Tensor) -> None:
        """
        Method to update internal state values
        :param next_state: Next state
        """
        pass

    def compute_aux_states(self, curr_state: torch.Tensor) -> None:
        """
        Given a state, compute and update all other internal attributes e.g given COM and orientation of rod,
        compute and update end points

        :param curr_state: Current state
        """
        pass

    def update_system_topology(self) -> None:
        """
        Method for updating the system topology (and site locations dictionary)
        """
        pass

    def compute_contact_deltas(self,
                               pre_next_state: torch.Tensor,
                               dt: Union[torch.Tensor, float]
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def resolve_contacts(self,
                         pre_next_state: torch.Tensor,
                         dt: Union[torch.Tensor, float],
                         delta_v,
                         delta_w,
                         toi) -> torch.Tensor:
        pass

    def start_hook(self, curr_state, external_forces, external_pts, dt):
        pass

    def post_forces_hook(self, curr_state, net_force, forces, acting_pts):
        pass

    def get_body_vecs(self, curr_state, acting_pts):
        curr_pos = self.get_xyz_pos(curr_state)
        body_vecs = acting_pts - curr_pos

        return body_vecs

    def print_state(self, state: torch.Tensor, time: float) -> Dict:
        raise NotImplementedError

    def apply_control(self, control_signals):
        pass

    def step(self,
             curr_state: torch.Tensor,
             dt: Union[torch.Tensor, float],
             external_forces: torch.Tensor = None,
             external_pts: torch.Tensor = None,
             control_signals: torch.Tensor = None) -> torch.Tensor:
        """
        Method to state current state and external forces/pts and step forward dynamics by dt. No internal happens
        after step forward.

        :param curr_state: Current state
        :param dt: delta t stepsize
        :param external_forces: Applied forces
        :param external_pts: Acting points of applied forces. Need to align with Applied forces

        :return: Next state
        """
        # If no external forces provided, assume 0 force

        if isinstance(dt, float):
            dt = torch.tensor([[[dt]]],
                              dtype=curr_state.dtype,
                              device=curr_state.device
                              )

        if external_forces is None or external_pts is None:
            size = (curr_state.shape[0], 3, 1)
            external_forces = torch.zeros(size, dtype=self.sys_precision, device=curr_state.device)
            external_pts = torch.zeros(size, dtype=self.sys_precision, device=curr_state.device)
        # Compute all other properties from state
        # self.compute_aux_states(curr_state)

        # Compute all (and net) forces
        net_force, forces, acting_pts = self.compute_forces(external_forces, external_pts)

        # Compute all (and net) torques
        body_vecs = self.get_body_vecs(curr_state, acting_pts)
        net_torque, _ = self.compute_torques(forces, body_vecs)

        # Compute the current linear and angular accelerations from net force and torque
        lin_acc, ang_acc = self.compute_accelerations(net_force, net_torque)

        # Compute next rod state (pos, lin vel, quat, ang vel)
        pre_next_state = self.time_integration(lin_acc, ang_acc, dt)

        # Resolve contacts
        delta_v, delta_w, toi = self.compute_contact_deltas(pre_next_state, dt)

        next_state = self.resolve_contacts(pre_next_state,
                                           dt,
                                           delta_v,
                                           delta_w,
                                           toi)
        return next_state


def rod_initializer(rod_type, rod_config, dtype=torch.float64):
    rod_config_no_type = {k: v for k, v in rod_config.items() if k != "rod_type"}
    rod_mapping_fn = {
        'rod': rods.RodState,
        'rod_sphere_end_caps': rods.RodSphericalEndCaps,
        'rod_sphere_end_caps_motors': rods.RodCylinderMotorsSphericalEndCaps,
        'rod_housing_end_caps_motors': rods.RodHousingMotorsSphericalEndCaps,
    }

    rod = rod_mapping_fn[rod_type].init_to_torch_tensors(
        rod_config_no_type,
        dtype
    )
    return rod