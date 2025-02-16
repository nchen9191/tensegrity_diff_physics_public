import torch

from diff_physics_engine.simulators.abstract_simulator import rod_initializer
from diff_physics_engine.state_objects.base_state_object import BaseStateObject
from diff_physics_engine.state_objects.springs import get_spring, ActuatedCable
from diff_physics_engine.state_objects.system_topology import SystemTopology
from utilities import torch_quaternion, inertia_tensors
from utilities.misc_utils import MinDistTwoCircles
from utilities.tensor_utils import zeros


class TensegrityRobot(BaseStateObject):

    def __init__(self,
                 config,
                 sys_precision):
        super().__init__(config['name'])
        topology_dict = config['system_topology']
        self.system_topology = SystemTopology.init_to_torch(topology_dict['sites'],
                                                            topology_dict['topology'],
                                                            dtype=sys_precision)
        self.sys_precision = sys_precision
        self.rods = self._init_rods(config)
        self.springs = self._init_springs(config)

        self.pos = torch.hstack([rod.pos for rod in self.rods.values()])
        self.linear_vel = torch.hstack([rod.linear_vel for rod in self.rods.values()])
        self.quat = torch.hstack([rod.quat for rod in self.rods.values()])
        self.ang_vel = torch.hstack([rod.ang_vel for rod in self.rods.values()])

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
            for rod in self.rods.values()
        ])

    @property
    def potential_energy(self):
        rod_pe = torch.stack(
            [r.potential_energy for r in self.rods.values()],
            dim=2
        ).sum(dim=2, keepdim=True)

        spring_pe = torch.stack(
            [s.compute_potential_energy(
                self.system_topology.sites_dict[s.end_pts[0]],
                self.system_topology.sites_dict[s.end_pts[1]]
            ) for s in self.springs.values()],
            dim=2
        ).sum(dim=2, keepdim=True)

        total_pe = rod_pe + spring_pe
        return total_pe

    @property
    def kinetic_energy(self):
        rod_ke = torch.stack(
            [r.kinetic_energy for r in self.rods.values()],
            dim=2
        ).sum(dim=2, keepdim=True)

        return rod_ke

    def compute_state_to_energy(self, full_state):
        temp_sites = {}
        kes, pes = [], []
        for i, rod in enumerate(self.rods.values()):
            state = full_state[:, i * 13: (i + 1) * 13]
            pos, quat = state[:, :3], state[:, 3:7]
            lvel, avel = state[:, 7:10], state[:, 10:13]
            inertia = inertia_tensors.body_to_world_torch(
                torch_quaternion.quat_as_rot_mat(quat),
                rod.I_body
            )

            for k, v in rod.sites.items():
                s = torch_quaternion.rotate_vec_quat(quat, v) + pos
                temp_sites[k] = s

            lin_ke = 0.5 * rod.mass * (lvel ** 2).sum(dim=1, keepdim=True)
            ang_ke = 0.5 * torch.linalg.vecdot(avel, inertia @ avel, dim=1).unsqueeze(1)
            ke = lin_ke + ang_ke
            kes.append(ke)

            pe = pos[:, 2:3] * rod.mass * 9.81
            pes.append(pe)

        for c in self.springs.values():
            e0, e1, = c.end_pts
            s0, s1 = temp_sites[e0], temp_sites[e1]
            pe = c.compute_potential_energy(s0, s1)
            pes.append(pe)

        ke = sum(kes)
        pe = sum(pes)
        total_energy = ke + pe
        return total_energy

    def _init_sites(self):
        for rod in self.rods.values():
            for site in rod.sites:
                world_frame_pos = self.system_topology.sites_dict[site].reshape(-1, 3, 1)
                body_frame_pos = rod.world_to_body_coords(world_frame_pos)
                rod.update_sites(site, body_frame_pos)

    def move_tensors(self, device):
        self.system_topology.move_tensors(device)
        for k, rod in self.rods.items():
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

        for i, rod in enumerate(self.rods.values()):
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
        for rod in self.rods.values():
            for site, rel_pos in rod.sites.items():
                world_pos = rod.body_to_world_coords(rel_pos)
                self.system_topology.update_site(site, world_pos)

    def _init_rods(self, config):
        rods = {}
        for rod_config in config['rods']:
            rod_state = rod_initializer(rod_config['rod_type'],
                                        rod_config,
                                        self.sys_precision)
            rods[rod_state.name] = rod_state

        return rods

    def _init_springs(self, config):
        springs = {}
        for spring_config in config['springs']:
            spring_cls = get_spring(spring_config['type'])
            config = {k: v for k, v in spring_config.items() if k != 'type'}

            spring = spring_cls.init_to_torch_tensors(**config,
                                                      sys_precision=self.sys_precision)
            springs[spring.name] = spring

        return springs

    def _find_rod_idxs(self, spring):
        end_pt0, end_pt1 = spring.end_pts
        rod_idx0, rod_idx1 = None, None

        for i, rod in enumerate(self.rods.values()):
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
        rod_end_pts = [[None] * len(self.springs) for _ in range(len(self.rods))]
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
            if rest_length.shape[0] == 1:
                rest_length = rest_length.repeat(batch_size, 1, 1)
            rest_lengths.append(rest_length)
        return torch.concat(rest_lengths, dim=2)

    def get_spring_acting_pts(self):
        ref_tensor = None
        for s in self.rod_end_pts[0]:
            if s:
                ref_tensor = self.system_topology.sites_dict[s]
                break

        act_pts = torch.hstack([
            torch.concat(
                [self.system_topology.sites_dict[s]
                 if s else zeros(ref_tensor.shape, ref_tensor=ref_tensor)
                 for s in spring_list],
                dim=2
            )
            for spring_list in self.rod_end_pts
        ])

        return act_pts

    def compute_spring_forces(self):
        endpt_idxs0, endpt_idxs1 = self.spring2rod_idxs
        rods = list(self.rods.values())
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

    def compute_cable_length(self, cable):
        e0, e1 = cable.end_pts
        e0, e1 = f"s{e0.split('_')[1]}", f"s{e0.split('_')[2]}"

        end_pt0 = self.system_topology.sites_dict[e0]
        end_pt1 = self.system_topology.sites_dict[e1]

        x_dir = end_pt1 - end_pt0
        length = x_dir.norm(dim=1, keepdim=True)
        x_dir = x_dir / length

        end_pt0_ = self.system_topology.sites_dict[cable.end_pts[0]]
        end_pt1_ = self.system_topology.sites_dict[cable.end_pts[1]]

        x_dir_ = end_pt1_ - end_pt0_
        length_ = x_dir_.norm(dim=1, keepdim=True)

        sites_dict = self.system_topology.sites_dict
        for k, v in sites_dict.items():
            if len(k) < 3:
                continue
            endpt = sites_dict[f"s{k.split('_')[1]}"]
            dist = (endpt - v).norm(dim=1)
            lll=9

        return length, x_dir

    def align_prin_axis_2d(self, new_prin_axis, new_com):
        robot_left_midpt = torch.concat([
            r.end_pts[0] for r in self.rods.values()
        ], dim=-1).mean(dim=-1, keepdim=True)
        robot_right_midpt = torch.concat([
            r.end_pts[1] for r in self.rods.values()
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

    def init_by_endpts(self, rod_endpts):
        # with torch.no_grad():
        rigid_body = list(self.rods.values())[0]
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

            # poses, quats = [], []
            # for i, rod in enumerate(self.rods.values()):
            #     rod_sites = {k: self.system_topology.sites_dict[k] for k in rod.sites.keys()}
            #     endpts = rod_endpts[i]
            #     prin = endpts[1] - endpts[0]
            #     pos = (endpts[1] + endpts[0]) / 2
            #     quat = RodState.compute_init_quat_principal(prin)
            #     new_quat = self.compute_q_offset(attachment_sites,
            #                                      rod_sites,
            #                                      pos,
            #                                      quat,
            #                                      rod.pos,
            #                                      rod.quat)
            #     quats.append(new_quat)
            #     poses.append(pos)
            # poses = torch.hstack(poses)
            # quats = torch.hstack(quats)
            # lin_vel = zeros(poses.shape, ref_tensor=poses)
            # ang_vel = zeros(poses.shape, ref_tensor=poses)
            # self.update_state(poses, lin_vel, quats, ang_vel)

        for i, rod in enumerate(self.rods.values()):
            end_pts = rod_endpts[i]
            lin_vel = zeros((1, 3, 1), ref_tensor=rod.linear_vel)
        #     # lin_vel = zeros(rod.linear_vel.shape, ref_tensor=rod.linear_vel)
            ang_vel = zeros((1, 3, 1), ref_tensor=rod.ang_vel)
        #     # ang_vel = zeros(rod.ang_vel.shape, ref_tensor=rod.ang_vel)
            rod.update_state_by_endpts(end_pts, lin_vel, ang_vel)
        #
        for k, v in attachment_sites.items():
            self.system_topology.sites_dict[k] = v

        for i, e in enumerate(rod_endpts):
            self.system_topology.sites_dict[f"s{2 * i}"] = e[0]
            self.system_topology.sites_dict[f"s{2 * i + 1}"] = e[1]

        sites_dict = self.system_topology.sites_dict
        for k, v in sites_dict.items():
            if len(k) < 3:
                continue
            e = f"s{k.split('_')[1]}"
            endpt = sites_dict[e]
            dist = (endpt - v).norm(dim=1)
            lll = 9
        #
        for rigid_body in self.rods.values():
            for site in rigid_body.sites.keys():
                world_frame_pos = self.system_topology.sites_dict[site].reshape(-1, 3, 1)
                body_frame_pos = rigid_body.world_to_body_coords(world_frame_pos)
                rigid_body.update_sites(site, body_frame_pos)

    def compute_q_offset(self,
                         new_sites,
                         ref_sites,
                         pos_new,
                         quat_new,
                         pos_ref,
                         quat_ref):
        ref_sites = {k: v for k, v in ref_sites.items() if len(k) > 3}
        inv_quat_fn = lambda q, v: torch_quaternion.rotate_vec_quat(
            torch_quaternion.inverse_unit_quat(q), v)

        new_sites = {k: new_sites[k] for k in ref_sites.keys()}

        ref_ref_sites = torch.vstack([inv_quat_fn(quat_ref, v - pos_ref)
                                      for k, v in ref_sites.items()])
        ref_new_sites = torch.vstack([inv_quat_fn(quat_new, new_sites[k] - pos_new)
                                      for k in ref_sites.keys()])

        ref_sites1_norm = ref_ref_sites[:, :2].norm(dim=1, keepdim=True)
        ref_sites2_norm = ref_new_sites[:, :2].norm(dim=1, keepdim=True)
        angles = torch.linalg.vecdot(ref_ref_sites[:, :2], ref_new_sites[:, :2], dim=1).unsqueeze(1)
        angles = angles / ref_sites1_norm / ref_sites2_norm
        angles = torch.arccos(torch.clamp(angles, -1, 1))
        avg_angle = angles.mean(dim=0, keepdim=True)

        cross_prods = ref_ref_sites[:, 0:1] * ref_new_sites[:, 1:2] - ref_ref_sites[:, 1:2] * ref_new_sites[:, 0:1]
        rot_dir = torch.sign(cross_prods.mean(dim=0, keepdim=True))

        q = torch.tensor(
            [torch.cos(avg_angle / 2), 0, 0, rot_dir * torch.sin(avg_angle / 2)],
            dtype=pos_new.dtype, device=pos_new.device
        ).reshape(1, 4, 1)
        q_new = torch_quaternion.quat_prod(quat_new, q)

        return q_new

    def _get_rod_ends(self, rod_end_pts, sphere_radius, rod_idxs=None):
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

    def _pt_matching(self, pts, match_idxs):
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

    def _pt_matching2(self, pts, match_idxs):
        pts_ = [[p.clone() for p in pt] for pt in pts]
        sites = {}

        for idxs in match_idxs:
            pts0, pts1, pts2 = pts_[idxs[0]], pts_[idxs[1]], pts_[idxs[2]]
            combos = [
                [(pts0[0], pts1[0]), (pts1[1], pts2[0]), (pts2[1], pts0[1])],
                [(pts0[0], pts1[0]), (pts1[1], pts2[1]), (pts2[0], pts0[1])],
                [(pts0[0], pts1[1]), (pts1[0], pts2[1]), (pts2[0], pts0[1])],
                [(pts0[0], pts1[1]), (pts1[0], pts2[1]), (pts2[0], pts0[1])],
                [(pts0[1], pts1[0]), (pts1[1], pts2[0]), (pts2[1], pts0[0])],
                [(pts0[1], pts1[0]), (pts1[1], pts2[1]), (pts2[0], pts0[0])],
                [(pts0[1], pts1[1]), (pts1[0], pts2[0]), (pts2[1], pts0[0])],
                [(pts0[1], pts1[1]), (pts1[0], pts2[1]), (pts2[0], pts0[0])],
            ]
            dists = [(sum([(p0 - p1).norm() for p0, p1 in c]), c) for c in combos]

            min_pair = min(dists, key=lambda x: x[0])[1]
            sites[f's_{idxs[0]}_{idxs[1]}'] = min_pair[0][0]
            sites[f's_{idxs[1]}_{idxs[0]}'] = min_pair[0][1]
            sites[f's_{idxs[1]}_{idxs[2]}'] = min_pair[1][0]
            sites[f's_{idxs[2]}_{idxs[1]}'] = min_pair[1][1]
            sites[f's_{idxs[2]}_{idxs[0]}'] = min_pair[2][0]
            sites[f's_{idxs[0]}_{idxs[2]}'] = min_pair[2][1]

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

    # def _compute_attachments_real(self,
    #                               end_pt,
    #                               motor_pt,
    #                               prins,
    #                               motor_radius,
    #                               sphere_radius):
    #     prin_end_pt, prin_motor_pt = prins
    #
    #     normal = torch.cross(prins[1], prins[0], dim=1)
    #     normal /= normal.norm(dim=1, keepdim=True)
    #     angle = torch.tensor(torch.pi / 4, dtype=prins[0].dtype).reshape(1, 1, 1)
    #     q1 = torch.hstack([torch.cos(angle), normal * torch.sin(angle)])
    #     r = sphere_radius * torch_quaternion.rotate_vec_quat(q1, prin_end_pt)
    #
    #     pos_end_pts = [
    #         end_pt + r,
    #         end_pt - r
    #     ]
    #
    #     end_pt = min(pos_end_pts, key=lambda p: (p - motor_pt).norm(dim=1))
    #
    #     motor_pt_dir = self._compute_closest_pt_dir_on_circle(end_pt, prin_motor_pt)
    #     motor_pt = motor_pt + motor_radius * motor_pt_dir
    #
    #     return end_pt, motor_pt

    def _compute_attachments_real(self,
                                  end_pt,
                                  motor_pt,
                                  prins,
                                  motor_radius,
                                  sphere_radius):
        prin_end_pt, prin_motor_pt = prins

        minimizer = MinDistTwoCircles()
        _, _, angles, pts = minimizer.opt(1000,
                                          end_pt, sphere_radius, prin_end_pt,
                                          motor_pt, motor_radius, prin_motor_pt)

        return pts

    def compute_real_cable_attachments(self,
                                       end_pts,
                                       sphere_radius,
                                       motor_radius,
                                       motor_offset):
        if len(end_pts) == 3:
            end_pts = [e for endpts in end_pts for e in endpts]  # flatten

        attachment_pts = {}

        # principal axis
        prins = [end_pts[2 * i + 1] - end_pts[2 * i] for i in range(len(end_pts) // 2)]
        prins = [p / p.norm(dim=1, keepdim=True) for p in prins]

        short_cable_pts = []
        for i, j in [(0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 1)]:
            sgn = 1 if j % 2 == 0 else -1
            rod_prins = [prins[int(i // 2)], prins[int(j // 2)]]
            end_center = end_pts[i]
            motor_center = end_pts[j] + sgn * motor_offset * rod_prins[1]
            sphere_pt0, motor_pt = self._compute_attachments_real(end_center,
                                                                  motor_center,
                                                                  rod_prins,
                                                                  motor_radius,
                                                                  sphere_radius)

            angle = torch.tensor(torch.pi / 4, dtype=prins[0].dtype).reshape(1, 1, 1)
            q = torch.hstack([torch.cos(angle), rod_prins[0] * torch.sin(angle)])

            r0 = sphere_pt0 - end_center
            r1 = torch_quaternion.rotate_vec_quat(q, r0)
            r2 = torch_quaternion.rotate_vec_quat(q, r1)
            r3 = torch_quaternion.rotate_vec_quat(q, r2)
            sphere_pt1 = end_center + r1
            sphere_pt2 = end_center + r2
            sphere_pt3 = end_center + r3

            attachment_pts.update({
                f"s_{i}_b{j}": sphere_pt0,
                f"s_b{j}_{i}": motor_pt,
                f"s_{i}_{i + (3 if i < 3 else -3)}": sphere_pt2
            })

            short_cable_pts.append([sphere_pt1, sphere_pt3])

        match_idxs = [(0, 2, 4),
                      (1, 3, 5)]
        short_cable_sites = self._pt_matching2(short_cable_pts, match_idxs)
        attachment_pts.update(short_cable_sites)

        # for k in sorted(attachment_pts.keys()):
        #     v = attachment_pts[k]
        #     print(f"{k}: {v.flatten()[0]} {v.flatten()[1]} {v.flatten()[2]}")

        return attachment_pts

    def compute_stable_rest_lengths_(self):
        rods = [
            self.rigid_bodies['rod_01'],
            self.rigid_bodies['rod_23'],
            self.rigid_bodies['rod_45'],
        ]

        rod_sites = [
            ['s_1_3', 's_1_5', 's_0_2', 's_0_4', 's_0_3', 's_1_4'],
            ['s_3_5', 's_3_1', 's_2_0', 's_2_4', 's_2_5', 's_3_0'],
            ['s_5_3', 's_5_1', 's_4_0', 's_4_2', 's_5_2', 's_4_1'],
        ]
        idxs = [1, 2, 3, 4, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 22, 23, 24, 26]
        sphere_radius = rods[0].sphere_radius

        z = torch.tensor([0, 0, 1], dtype=self.sys_precision).reshape(1, 3, 1)
        stiffnesses = torch.tensor([[s.stiffness for s in self.springs.values()]],
                                   dtype=self.sys_precision)

        rod_endpts = [rod.end_pts for rod in rods]
        flood_ends = [min(e, key=lambda x: x[:, 2]) for e in rod_endpts]
        contacts = [e - sphere_radius * z for e in flood_ends]

        torque_g = torch.hstack([
            torch.cross(-rod.mass * 9.81 * z, rod.pos - c, dim=1)
            for rod, c in zip(rods, contacts)
        ])

        dxs, x_hats = zip(*[self.compute_cable_length(s) for s in self.springs.values()])
        dx_dict = {s.end_pts[0]: dx for s, dx in zip(self.springs.values(), dxs)}
        dx_dict.update({s.end_pts[1]: dx for s, dx in zip(self.springs.values(), dxs)})
        dx_tensor = zeros((27, 1, 1), ref_tensor=sphere_radius)
        dx_tensor[idxs] = torch.vstack([
            torch.vstack([dx_dict[s] for s in sites])
            for sites in rod_sites
        ])

        x_hats_dict = {s.end_pts[0]: x_hat for s, x_hat in zip(self.springs.values(), x_hats)}
        x_hats_dict.update({s.end_pts[1]: -x_hat for s, x_hat in zip(self.springs.values(), x_hats)})
        x_hats_tensor = torch.vstack([
            torch.vstack([x_hats_dict[s] for s in sites])
            for sites in rod_sites
        ])

        body_vecs = torch.vstack([
            torch.vstack([self.system_topology.sites_dict[s] - contacts[i] for s in sites])
            for i, sites in enumerate(rod_sites)
        ])

        v = zeros((27, 3, 1), ref_tensor=sphere_radius)
        v[idxs] = torch.cross(x_hats_tensor, body_vecs, dim=1)

        d_mat = torch.hstack([
            v[:9], v[9:18], v[18:]
        ]).T * stiffnesses

        c = stiffnesses.T.repeat(3, 1).unsqueeze(-1) * dx_tensor * v
        c_ = torch.vstack([
            c[:9].sum(dim=0).reshape(-1, 1),
            c[9:18].sum(dim=0).reshape(-1, 1),
            c[18:].sum(dim=0).reshape(-1, 1)
        ])

        d = torch.linalg.solve(d_mat.squeeze(0), (c_ + torque_g.squeeze(0)))

        return d

    def compute_stable_rest_lengths(self):
        rods = [
            self.rigid_bodies['rod_01'],
            self.rigid_bodies['rod_23'],
            self.rigid_bodies['rod_45'],
        ]

        rod_sites = [
            ['s_1_3', 's_1_5', 's_0_2', 's_0_4', 's_0_3', 's_1_4'],
            ['s_3_5', 's_3_1', 's_2_0', 's_2_4', 's_2_5', 's_3_0'],
            ['s_5_3', 's_5_1', 's_4_0', 's_4_2', 's_5_2', 's_4_1'],
        ]
        sites = [[self.system_topology.sites_dict[s] for s in rod_site] for rod_site in rod_sites]
        sphere_radius = rods[0].sphere_radius

        z = torch.tensor([0, 0, 1], dtype=self.sys_precision).reshape(1, 3, 1)
        stiffnesses = torch.tensor([[s.stiffness for s in self.springs.values()]],
                                   dtype=self.sys_precision)

        rod_endpts = [rod.end_pts for rod in rods]
        flood_ends = [min(e, key=lambda x: x[:, 2]) for e in rod_endpts]
        contacts = [e - sphere_radius * z for e in flood_ends]

        torque_g = torch.hstack([
            torch.cross(-rod.mass * 9.81 * z, rod.pos - c, dim=1)
            for rod, c in zip(rods, contacts)
        ]).squeeze()

        dxs, x_hats = zip(*[self.compute_cable_length(s) for s in self.springs.values()])
        dxs = [dx.squeeze() for dx in dxs]

        d_mat = zeros((9, 9), ref_tensor=torque_g)
        d_mat[:3, 1] = torch.cross(x_hats[1], (sites[0][0] - contacts[0]), dim=1).squeeze()
        d_mat[:3, 2] = torch.cross(x_hats[2], (sites[0][1] - contacts[0]), dim=1).squeeze()
        d_mat[:3, 3] = torch.cross(x_hats[3], (sites[0][2] - contacts[0]), dim=1).squeeze()
        d_mat[:3, 4] = torch.cross(x_hats[4], (sites[0][3] - contacts[0]), dim=1).squeeze()
        d_mat[:3, 7] = torch.cross(x_hats[7], (sites[0][4] - contacts[0]), dim=1).squeeze()
        d_mat[:3, 8] = torch.cross(x_hats[8], (sites[0][5] - contacts[0]), dim=1).squeeze()

        d_mat[3:6, 0] = torch.cross(x_hats[0], (sites[1][0] - contacts[1]), dim=1).squeeze()
        d_mat[3:6, 1] = torch.cross(-x_hats[1], (sites[1][1] - contacts[1]), dim=1).squeeze()
        d_mat[3:6, 3] = torch.cross(-x_hats[3], (sites[1][2] - contacts[1]), dim=1).squeeze()
        d_mat[3:6, 5] = torch.cross(x_hats[5], (sites[1][3] - contacts[1]), dim=1).squeeze()
        d_mat[3:6, 6] = torch.cross(x_hats[6], (sites[1][4] - contacts[1]), dim=1).squeeze()
        d_mat[3:6, 7] = torch.cross(-x_hats[7], (sites[1][5] - contacts[1]), dim=1).squeeze()

        d_mat[6:9, 0] = torch.cross(-x_hats[0], (sites[2][0] - contacts[2]), dim=1).squeeze()
        d_mat[6:9, 2] = torch.cross(-x_hats[2], (sites[2][1] - contacts[2]), dim=1).squeeze()
        d_mat[6:9, 4] = torch.cross(-x_hats[4], (sites[2][2] - contacts[2]), dim=1).squeeze()
        d_mat[6:9, 5] = torch.cross(-x_hats[5], (sites[2][3] - contacts[2]), dim=1).squeeze()
        d_mat[6:9, 6] = torch.cross(-x_hats[6], (sites[2][4] - contacts[2]), dim=1).squeeze()
        d_mat[6:9, 8] = torch.cross(-x_hats[8], (sites[2][5] - contacts[2]), dim=1).squeeze()

        d_mat *= stiffnesses

        c = zeros((9, 1), ref_tensor=torque_g)
        c[:3, 0] = (d_mat[:3, 1] * dxs[1]
                    + d_mat[:3, 2] * dxs[2]
                    + d_mat[:3, 3] * dxs[3]
                    + d_mat[:3, 4] * dxs[4]
                    + d_mat[:3, 7] * dxs[7]
                    + d_mat[:3, 8] * dxs[8]) + torque_g[:3]

        c[3:6, 0] = (d_mat[3:6, 0] * dxs[0]
                     + d_mat[3:6, 1] * dxs[1]
                     + d_mat[3:6, 3] * dxs[3]
                     + d_mat[3:6, 5] * dxs[5]
                     + d_mat[3:6, 6] * dxs[6]
                     + d_mat[3:6, 7] * dxs[7]) + torque_g[3:6]

        c[6:, 0] = (d_mat[6:9, 0] * dxs[0]
                    + d_mat[6:9, 2] * dxs[2]
                    + d_mat[6:9, 4] * dxs[4]
                    + d_mat[6:9, 5] * dxs[5]
                    + d_mat[6:9, 6] * dxs[6]
                    + d_mat[6:9, 8] * dxs[8]) + torque_g[6:]

        d = torch.matmul(torch.linalg.inv(d_mat), c)
        # d = torch.max(d, torch.vstack(dxs))

        # forces = stiffnesses.T * (torch.stack(dxs).unsqueeze(-1) - d) * torch.vstack(x_hats).squeeze(-1)
        # print(forces)

        return torch.vstack(dxs)

    def grad_descent_rest_lengths(self):
        springs = [s for s in self.springs.values()]
        params = torch.nn.ParameterList([torch.rand(1) * 1.2 + 0.8 for _ in range(9)])
        # + [torch.rand(1) * 4000 for _ in range(1)])
        # print([r.detach().item() for r in params])
        # params = torch.nn.ParameterList([s._rest_length for s in springs[:]])
        #                                 + [springs[0]._rest_length, springs[6]._rest_length]
        #                                 + [springs[0].stiffness, springs[6].stiffness])

        # params = torch.nn.ParameterList([springs[0]._rest_length, springs[0].stiffness,
        #                                  springs[6]._rest_length, springs[6].stiffness])
        max_lengths = []
        for i, s in enumerate(springs[:]):
            s._rest_length = params[i]
            length, _ = self.compute_cable_length(s)
            max_lengths.append(length.detach().item())

        # for s in springs[:6]:
        #     s._rest_length = params[-4]
        #     s.stiffness = params[-2]

        # for s in springs[6:9]:
        #     s._rest_length = params[-3]
        #     s.stiffness = params[-1]

        short_passive_max_length = min([self.compute_cable_length(s) for s in springs[:6]],
                                       key=lambda x: x[0])[0].item()

        long_passive_max_length = min([self.compute_cable_length(s) for s in springs[6:9]],
                                      key=lambda x: x[0])[0].item()

        optimizer = torch.optim.Adam(params, lr=0.01)
        start_state = self.get_curr_state()
        best_max_vel = 100.0

        # print(start_state.flatten())

        for i in range(3000):
            # if i == 500:
            #     optimizer = torch.optim.Adam(params, lr=0.001)
            if i == 400:
                optimizer = torch.optim.Adam(params, lr=0.001)
            curr_state = start_state.clone()
            for _ in range(1):
                curr_state = self.step(curr_state, 0.001)
            # print(curr_state.flatten())
            start_vel = torch.vstack([start_state[0, 7:10], start_state[0, 20:23], start_state[0, 33:36]])
            curr_vel = torch.vstack([curr_state[0, 7:10], curr_state[0, 20:23], curr_state[0, 33:36]])

            loss = ((curr_vel - start_vel) ** 2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for j, max_length in enumerate(max_lengths):
                params[j].data.clamp_(0.8, max_length)
            # params[-4].data.clamp_(0.8, short_passive_max_length)
            # params[-3].data.clamp_(0.8, long_passive_max_length)
            # params[-2].data.clamp_(1000, 15000)
            # params[-1].data.clamp_(1000, 15000)

            if torch.abs(curr_vel).max().detach().item() < best_max_vel:
                best_max_vel = torch.abs(curr_vel).max().item()
                final_params = [r.detach().item() for r in params]
            print(i, torch.abs(curr_vel).sum().item(), torch.abs(curr_vel).max().item())

        print(best_max_vel, final_params)
