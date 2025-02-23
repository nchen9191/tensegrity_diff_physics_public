from typing import Dict, List

import torch

from simulators.abstract_simulator import rod_initializer
from state_objects.base_state_object import BaseStateObject
from state_objects.springs import get_spring, ActuatedCable, SpringState
from state_objects.system_topology import SystemTopology
from utilities import torch_quaternion
from utilities.tensor_utils import zeros


class TensegrityRobot:

    def __init__(self,
                 config: Dict,
                 sys_precision: torch.dtype):
        self.name = config['name']
        topology_dict = config['system_topology']
        self.system_topology = SystemTopology.init_to_torch(topology_dict['sites'],
                                                            topology_dict['topology'],
                                                            dtype=sys_precision)
        self.sys_precision = sys_precision
        self.rods = self._init_rods(config)
        self.springs = self._init_springs(config)

        self.pos = torch.hstack([rod.pos for rod in self.rods])
        self.linear_vel = torch.hstack([rod.linear_vel for rod in self.rods])
        self.quat = torch.hstack([rod.quat for rod in self.rods])
        self.ang_vel = torch.hstack([rod.ang_vel for rod in self.rods])

        self.actuated_cables, self.non_actuated_cables = {}, {}
        for k, s in self.springs.items():
            if isinstance(s, ActuatedCable):
                self.actuated_cables[k] = s
            else:
                self.non_actuated_cables[k] = s

        self.k_mat, self.c_mat, self.spring2rod_idxs, self.rod_end_pts \
            = self.build_spring_consts()

        self._init_sites()
        self.cable_map = config['act_cable_mapping'] \
            if 'act_cable_mapping' in config \
            else {s.name: s.name for s in self.actuated_cables.values()}

    def get_curr_state(self) -> torch.Tensor:
        return torch.hstack([
            torch.hstack([
                rod.pos,
                rod.quat,
                rod.linear_vel,
                rod.ang_vel,
            ])
            for rod in self.rods
        ])

    def _init_sites(self):
        for rod in self.rods:
            for site in rod.sites:
                world_frame_pos = self.system_topology.sites_dict[site].reshape(-1, 3, 1)
                body_frame_pos = rod.world_to_body_coords(world_frame_pos)
                rod.update_sites(site, body_frame_pos)

    def move_tensors(self, device: str):
        self.system_topology.move_tensors(device)
        for rod in self.rods:
            rod.move_tensors(device)

        for k, spring in self.springs.items():
            spring.move_tensors(device)

        self.k_mat = self.k_mat.to(device)
        self.c_mat = self.c_mat.to(device)

        return self

    def update_state(self, pos, lin_vel, quat, ang_vel):
        self.pos = pos
        self.linear_vel = lin_vel
        self.quat = quat
        self.ang_vel = ang_vel

        for i, rod in enumerate(self.rods):
            rod.update_state(
                pos[:, i * 3: (i + 1) * 3],
                lin_vel[:, i * 3: (i + 1) * 3],
                quat[:, i * 4: (i + 1) * 4],
                ang_vel[:, i * 3: (i + 1) * 3],
            )

    def update_act_lens(self, act_lens):
        for i, c in enumerate(self.actuated_cables.values()):
            c.actuation_length = act_lens[:, i: i + 1]

    def update_motor_speeds(self, motor_speeds):
        for i, c in enumerate(self.actuated_cables.values()):
            c.motor.motor_state.omega_t = motor_speeds[:, i: i + 1]

    def update_system_topology(self):
        for rod in self.rods:
            for site, rel_pos in rod.sites.items():
                world_pos = rod.body_to_world_coords(rel_pos)
                self.system_topology.update_site(site, world_pos)

    def _init_rods(self, config: Dict):
        rods = []
        for rod_config in config['rods']:
            rod_state = rod_initializer(rod_config['rod_type'],
                                        rod_config,
                                        self.sys_precision)
            rods.append(rod_state)

        return rods

    def _init_springs(self, config: Dict):
        springs = {}
        for spring_config in config['springs']:
            spring_cls = get_spring(spring_config['type'])
            config = {}
            for k, v in spring_config.items():
                if k != 'type':
                    config[k] = v

            spring = spring_cls.init_to_torch_tensors(**config,
                                                      sys_precision=self.sys_precision)
            springs[spring.name] = spring

        return springs

    def _find_rod_idxs(self, spring: SpringState):
        end_pt0, end_pt1 = spring.end_pts
        rod_idx0, rod_idx1 = None, None

        for i, rod in enumerate(self.rods):
            if end_pt0 in rod.sites:
                rod_idx0 = i
            elif end_pt1 in rod.sites:
                rod_idx1 = i

        return rod_idx0, rod_idx1

    def build_spring_consts(self):
        k_mat = torch.zeros((1, len(self.rods), len(self.springs)),
                            dtype=self.sys_precision)
        c_mat = torch.zeros((1, len(self.rods), len(self.springs)),
                            dtype=self.sys_precision)

        end_pt0_idxs, end_pt1_idxs = [], []
        rod_end_pts = [["None"] * len(self.springs) for _ in range(len(self.rods))]
        for j, spring in enumerate(self.springs.values()):
            k = spring.stiffness
            c = spring.damping

            rod_idx0, rod_idx1 = self._find_rod_idxs(spring)
            end_pt0_idxs.append(rod_idx0)
            end_pt1_idxs.append(rod_idx1)
            rod_end_pts[rod_idx0][j] = spring.end_pts[0]
            rod_end_pts[rod_idx1][j] = spring.end_pts[1]

            m, n = (rod_idx0, rod_idx1) if rod_idx1 > rod_idx0 \
                else (rod_idx1, rod_idx0)

            k_mat[0, m, j] = k
            k_mat[0, n, j] = -k

            c_mat[0, m, j] = c
            c_mat[0, n, j] = -c

        return k_mat, c_mat, (end_pt0_idxs, end_pt1_idxs), rod_end_pts

    def get_rest_lengths(self):
        batch_size = self.pos.shape[0]
        rest_lengths = []
        for s in self.springs.values():
            rest_length = s.rest_length
            rest_lengths.append(rest_length)
        return torch.concat(rest_lengths, dim=2)

    def get_spring_acting_pts(self):
        act_pts = torch.hstack([
            torch.concat(
                [self.system_topology.sites_dict[s]
                 for s in spring_list],
                dim=2
            )
            for spring_list in self.rod_end_pts
        ])

        return act_pts

    def compute_spring_forces(self):
        endpt_idxs0, endpt_idxs1 = self.spring2rod_idxs
        rods = list(self.rods)
        springs = list(self.springs.values())

        rod_pos = torch.concat([rod.pos for rod in rods], dim=2)
        rod_linvel = torch.concat([rod.linear_vel for rod in rods], dim=2)
        rod_angvel = torch.concat([rod.ang_vel for rod in rods], dim=2)

        spring_end_pts0 = torch.concat([self.system_topology.sites_dict[s.end_pts[0]]
                                        for s in springs], dim=2)
        spring_end_pts1 = torch.concat([self.system_topology.sites_dict[s.end_pts[1]]
                                        for s in springs], dim=2)
        spring_vecs = spring_end_pts1 - spring_end_pts0
        spring_lengths = spring_vecs.norm(dim=1, keepdim=True)
        spring_unit_vecs = spring_vecs / spring_lengths

        length_diffs = spring_lengths - self.get_rest_lengths()
        length_diffs = (torch.clamp_min(length_diffs, 0.0)
                        .repeat(1, len(rods), 1))

        vels0 = (
                rod_linvel[..., endpt_idxs0]
                + torch.cross(rod_angvel[..., endpt_idxs0],
                              spring_end_pts0 - rod_pos[..., endpt_idxs0],
                              dim=1)
        )
        vels1 = (
                rod_linvel[..., endpt_idxs1]
                + torch.cross(rod_angvel[..., endpt_idxs1],
                              spring_end_pts1 - rod_pos[..., endpt_idxs1],
                              dim=1)
        )
        rel_vels = torch.linalg.vecdot(
            vels1 - vels0,
            spring_unit_vecs,
            dim=1
        ).unsqueeze(1).repeat(1, len(rods), 1)

        stiffness_force_mags = self.k_mat * length_diffs
        damping_force_mags = self.c_mat * rel_vels
        force_mags = stiffness_force_mags + damping_force_mags

        rod_forces = torch.hstack([
            force_mags[:, i: i + 1] * spring_unit_vecs
            for i in range(len(rods))
        ])
        act_pts = self.get_spring_acting_pts()

        net_rod_forces = rod_forces.sum(dim=2, keepdim=True)

        return net_rod_forces, rod_forces, act_pts

    def compute_cable_length(self, cable: SpringState):
        e0, e1 = cable.end_pts
        e0, e1 = f"s{e0.split('_')[1]}", f"s{e0.split('_')[2]}"

        end_pt0 = self.system_topology.sites_dict[e0]
        end_pt1 = self.system_topology.sites_dict[e1]

        x_dir = end_pt1 - end_pt0
        length = x_dir.norm(dim=1, keepdim=True)
        x_dir = x_dir / length

        return length, x_dir

    def align_prin_axis_2d(self, new_prin_axis, new_com):
        robot_left_midpt = torch.concat([
            r.end_pts[0] for r in self.rods
        ], dim=-1).mean(dim=-1, keepdim=True)
        robot_right_midpt = torch.concat([
            r.end_pts[1] for r in self.rods
        ], dim=-1).mean(dim=-1, keepdim=True)

        robot_prin = robot_right_midpt - robot_left_midpt

        new_prin_axis[:, 2] = 0.
        robot_prin[:, 2] = 0.

        robot_prin = robot_prin / robot_prin.norm(dim=1, keepdim=True)
        new_prin = new_prin_axis / new_prin_axis.norm(dim=1, keepdim=True)

        rot_axis = torch.cross(robot_prin, new_prin, dim=1)
        rot_axis = rot_axis / rot_axis.norm(dim=1, keepdim=True)

        angle = torch.linalg.vecdot(robot_prin, new_prin, dim=1).unsqueeze(1)
        angle = torch.acos(torch.clamp(angle, -1, 1)) / 2.

        quat = torch.hstack([torch.cos(angle), torch.sin(angle) * rot_axis])

        curr_state = self.get_curr_state()
        lin_vel = curr_state.reshape(-1, 13, 1)[:, 7:10].reshape(curr_state.shape[0], -1, 1)
        ang_vel = curr_state.reshape(-1, 13, 1)[:, 10:].reshape(curr_state.shape[0], -1, 1)
        rod_pose = curr_state.reshape(curr_state.shape[0], -1, 13)[..., :7]
        com = rod_pose[..., :3].mean(dim=1).unsqueeze(-1)
        new_com[:, 2] = com[:, 2]

        rod_pose = rod_pose.transpose(1, 2)
        new_rod_pos = rod_pose[:, :3] - com
        new_rod_pos = torch_quaternion.rotate_vec_quat(quat, new_rod_pos) + new_com
        new_rod_pos = new_rod_pos.transpose(1, 2).reshape(curr_state.shape[0], -1, 1)

        new_rod_quat = torch_quaternion.quat_prod(quat, rod_pose[:, 3:7])
        new_rod_quat = new_rod_quat.transpose(1, 2).reshape(curr_state.shape[0], -1, 1)

        self.update_state(new_rod_pos, lin_vel, new_rod_quat, ang_vel)

        return self.get_curr_state()

    def init_by_endpts(self, rod_endpts: List):
        # with torch.no_grad():
        rigid_body = list(self.rods)[0]
        sphere_radius = rigid_body.sphere_radius
        motor_radius = rigid_body.motor_radius

        motor_offset = (rigid_body.end_pts[0]
                        - rigid_body.rigid_bodies[f'{rigid_body.name}_motor1'].end_pts[0]
                        ).norm(dim=1, keepdim=True)
        #
        # attachment_sites = self.compute_real_cable_attachments(rod_endpts,
        #                                                        sphere_radius,
        #                                                        motor_radius,
        #                                                        motor_offset)
        attachment_sites = self.compute_cable_attachments(rod_endpts, sphere_radius)

        for i, rod in enumerate(self.rods):
            end_pts = rod_endpts[i]
            lin_vel = zeros((1, 3, 1), ref_tensor=rod.linear_vel)
            ang_vel = zeros((1, 3, 1), ref_tensor=rod.ang_vel)
            rod.update_state_by_endpts(end_pts, lin_vel, ang_vel)

        for k, v in attachment_sites.items():
            self.system_topology.sites_dict[k] = v

        for i, e in enumerate(rod_endpts):
            self.system_topology.sites_dict[f"s{2 * i}"] = e[0]
            self.system_topology.sites_dict[f"s{2 * i + 1}"] = e[1]

        for rigid_body in self.rods:
            for site in rigid_body.sites.keys():
                world_frame_pos = self.system_topology.sites_dict[site].reshape(-1, 3, 1)
                body_frame_pos = rigid_body.world_to_body_coords(world_frame_pos)
                rigid_body.update_sites(site, body_frame_pos)

    def _get_rod_ends(self, rod_end_pts, sphere_radius, rod_idxs: List = None):
        if rod_idxs is None:
            rod_idxs = [i for i in range(len(rod_end_pts))]

        num_rods = len(rod_end_pts)
        endpts_idxs, floor_ends, top_ends = [], [], []
        for i, end_pts in enumerate(rod_end_pts):
            if end_pts[0][:, 2] < end_pts[1][:, 2]:
                floor_ends.append(end_pts[0])
                top_ends.append(end_pts[1])
                endpts_idxs.append(2 * rod_idxs[i])
                endpts_idxs.append(2 * rod_idxs[i] + 1)
            else:
                floor_ends.append(end_pts[1])
                top_ends.append(end_pts[0])
                endpts_idxs.append(2 * rod_idxs[i] + 1)
                endpts_idxs.append(2 * rod_idxs[i])

        # floor_ends = [min(end_pts, key=lambda e: e[:, 2]) for end_pts in rod_end_pts]
        # top_ends = [max(end_pts, key=lambda e: e[:, 2]) for end_pts in rod_end_pts]

        # Shift end pts
        shift_deltas = [floor_ends[i][:, 2] - sphere_radius
                        for i in range(num_rods)]
        shift_deltas = [torch.tensor([0, 0, s], dtype=self.sys_precision).reshape(1, 3, 1)
                        for s in shift_deltas]
        floor_ends = [floor_ends[i] - shift_deltas[i]
                      for i in range(num_rods)]
        top_ends = [top_ends[i] - shift_deltas[i]
                    for i in range(num_rods)]

        return floor_ends, top_ends, shift_deltas, endpts_idxs

    def _rotate_pts(self, prin_axis, r, end_pt):
        angle = torch.ones((1, 1, 1), dtype=self.sys_precision) * torch.pi / 3

        q = torch.hstack([torch.cos(angle), torch.sin(angle) * prin_axis])
        p2 = torch_quaternion.rotate_vec_quat(q, r)
        p3 = torch_quaternion.rotate_vec_quat(q, p2)

        p2 += end_pt
        p3 += end_pt

        return p2, p3

    def _compute_closest_pt_dir_on_circle(self, pt, circle_normal):
        r = pt - torch.linalg.vecdot(pt, circle_normal, dim=1) * circle_normal
        r /= r.norm(dim=1, keepdim=True)

        return r

    def _compute_closest_pt_dir(self, p0, p1, prin, end_pt):
        dists = [(end_pt - p0).norm(dim=1), (end_pt - p1).norm(dim=1)]
        min_rod_pt = p0 if dists[0] < dists[1] else p1 - end_pt
        r = self._compute_closest_pt_dir_on_circle(min_rod_pt, prin)

        return r

    def _compute_first_pt_dir(self, prin):
        z = torch.tensor([0, 0, 1], dtype=self.sys_precision).reshape(1, 3, 1)
        r = torch.cross(torch.cross(prin, z, dim=1), prin, dim=1)
        r /= r.norm(dim=1, keepdim=True)

        return r

    def _pt_matching(self, pts, match_idxs: List):
        pts_ = [[p.clone() for p in pt] for pt in pts]
        sites = {}

        for idxs in match_idxs:
            pts_set0, pts_set1 = pts_[idxs[0]], pts_[idxs[1]]
            dists = []
            for i, p0 in enumerate(pts_set0):
                for j, p1 in enumerate(pts_set1):
                    dist = (p1 - p0).norm(dim=1)
                    dists.append((dist, p0, p1, i, j))
            min_pair = min(dists, key=lambda x: x[0])
            sites[f's_{idxs[0]}_{idxs[1]}'] = min_pair[1]
            sites[f's_{idxs[1]}_{idxs[0]}'] = min_pair[2]
            del pts_set0[min_pair[3]]
            del pts_set1[min_pair[4]]

        return sites

    def compute_cable_attachments(self, end_pts, sphere_radius):
        match_idxs = [(0, 2), (0, 4), (2, 4), (1, 3),
                      (1, 5), (3, 5), (0, 3), (1, 4), (2, 5)]

        # sort rods
        # rod_idxs, end_pts = zip(*sorted(enumerate(end_pts),
        #                                 key=lambda x: (x[1][0][:, 2] + x[1][1][:, 2]) / 2))
        rod_lengths = [(e[1] - e[0]).norm(dim=1) for e in end_pts]

        # Get bottom 3 end pts
        floor_ends, top_ends, z_shifts, idxs = self._get_rod_ends(
            end_pts,
            sphere_radius,
            # rod_idxs
        )
        # floor_idxs = idxs[::2]
        # floor_pairs = [
        #     [(floor_idxs[0], floor_idxs[1]), (floor_idxs[0], floor_idxs[2])],
        #     [(floor_idxs[0], floor_idxs[1]), (floor_idxs[1], floor_idxs[2])],
        #     [(floor_idxs[0], floor_idxs[2]), (floor_idxs[1], floor_idxs[2])],
        # ]
        #
        # pivot = 0
        # for j, p in enumerate(floor_pairs):
        #     if p[0] in match_idxs and p[1] in match_idxs:
        #         pivot = j
        #         break
        #
        # non_pivots = [j for j in range(len(floor_pairs)) if j != pivot]
        # ordering = [pivot] + non_pivots
        # floor_ends = [floor_ends[j] for j in ordering]
        # top_ends = [top_ends[j] for j in ordering]
        # z_shifts = [z_shifts[j] for j in ordering]
        # end_pts = [end_pts[j] for j in ordering]
        # idxs = [k for i in range(len(floor_pairs)) for k in idxs[2 * i: 2 * (i + 1)]]

        # principal axis
        prins = [top_ends[i] - floor_ends[i] for i in range(len(end_pts))]
        prins = [p / p.norm(dim=1, keepdim=True) for p in prins]

        pts = [[] for _ in range(6)]
        for i in range(3):
            r = self._compute_first_pt_dir(prins[i]) if i == 0 else \
                self._compute_closest_pt_dir(pts[idxs[0]][1],
                                             pts[idxs[0]][2],
                                             prins[i],
                                             floor_ends[i])
            r *= sphere_radius
            p1 = r + floor_ends[i] + z_shifts[i]
            p2, p3 = self._rotate_pts(prins[i], r, floor_ends[i])

            p2 += z_shifts[i]
            p3 += z_shifts[i]

            p4, p5, p6 = [prins[i] * rod_lengths[i] + p for p in [p1, p2, p3]]

            pts[idxs[2 * i]] = [p1, p2, p3]
            pts[idxs[2 * i + 1]] = [p4, p5, p6]

        sites = self._pt_matching(pts, match_idxs)

        return sites
