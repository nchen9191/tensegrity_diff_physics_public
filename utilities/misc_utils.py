import json
import shutil
from pathlib import Path

import torch

from utilities import torch_quaternion


def get_num_precision(precision: str) -> torch.dtype:
    if precision.lower() == 'float':
        return torch.float
    elif precision.lower() == 'float16':
        return torch.float16
    elif precision.lower() == 'float32':
        return torch.float32
    elif precision.lower() == 'float64':
        return torch.float64
    else:
        print("Precision unknown, defaulting to float16")
        return torch.float


def seq_to_train_data(json_path, output_path):
    with Path(json_path).open('r') as j:
        seq_data = json.load(j)

    seq_data.sort(key=lambda d: d['time'])

    train_data = []
    for i in range(len(seq_data) - 1):
        seq_curr = seq_data[i]
        seq_next = seq_data[i + 1]

        dt = seq_next['time'] - seq_curr['time']

        train_data.append({"curr": seq_curr, "next": seq_next, "dt": dt})

    with Path(output_path).open('w') as j:
        json.dump(train_data, j)


def multi_step_data_gen(raw_data, indices, num_steps, dtype):
    # batch_x = [{
    #     "curr_state": torch.tensor(raw_data[k]['pos'] + raw_data[k]['vel'], dtype=dtype),
    #     "external_forces": torch.zeros(3, dtype=dtype),
    #     "external_pts": torch.zeros(3, dtype=dtype),
    #     "dt": raw_data[k + 1]['time'] - raw_data[k]['time']
    # } for k in indices]

    curr_state = torch.tensor([raw_data[k]['pos'] + raw_data[k]['vel'] for k in indices], dtype=dtype)
    ext_forces = torch.zeros((len(indices), 3, 1))
    ext_pts = ext_forces.clone()
    dt = torch.tensor([raw_data[k + 1]['time'] - raw_data[k]['time'] for k in indices], dtype=dtype)

    batch_x = {
        "curr_state": curr_state.reshape(len(indices), -1, 1),
        "external_forces": ext_forces,
        "external_pts": ext_pts,
        "dt": dt.reshape(len(indices), -1, 1)
    }

    # batch_y = torch.vstack([torch.tensor([
    #     d['end_pt1'] + d['end_pt2']
    #     for d in raw_data[k + 1:k + num_steps + 1]], dtype=dtype)
    #     for k in indices]).reshape(6, num_steps, -1)

    batch_y = torch.vstack(
        [torch.tensor([
            d['end_pt1'] + d['end_pt2']
            for d in raw_data[k + num_steps:k + num_steps + 1]], dtype=dtype)
            for k in indices]
    ).unsqueeze(2)

    return batch_x, batch_y


def rod_mjc_to_dp_data(data_jsons, rod_names):
    from diff_physics_engine.state_objects.rigid_object import RigidBody

    for i in range(len(data_jsons)):
        rod_poses = []
        for j, rod_name in enumerate(rod_names):
            endpt1 = data_jsons[i][rod_name + "_end_pt1"]
            endpt2 = data_jsons[i][rod_name + "_end_pt2"]

            rod_pos = [(endpt2[k] + endpt1[k]) / 2 for k in range(len(endpt1))]

            prin_axis = torch.tensor([[endpt2[k] - endpt1[k] for k in range(len(endpt1))]], dtype=torch.float64)
            q = RigidBody.compute_init_quat_principal(prin_axis.reshape(1, 3, 1))

            rod_poses.extend(rod_pos)
            rod_poses.extend(q.flatten().numpy().tolist())

        data_jsons[i]["pos"] = rod_poses

    return data_jsons


def save_curr_code(code_dir, output_dir):
    code_dir = Path(code_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)
    for p in code_dir.iterdir():
        if p.suffix in [".py", ".json", ".xml"]:
            shutil.copy(p, output_dir / p.name)
        elif p.is_dir() and not p.name.startswith("."):
            save_curr_code(p, output_dir / p.name)


def compute_num_steps(time_gap, dt, tol=1e-6):
    num_steps = int(time_gap // dt)
    gap = time_gap - dt * num_steps
    num_steps += 0 if gap < tol else 1
    # num_steps += int(math.ceil(time_gap - dt * num_steps))

    return num_steps


class MinDistTwoCircles(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.ParameterList([torch.tensor(0.0), torch.tensor(0.0)])

    def compute_circle_axes(self, normal):
        normal /= normal.norm(dim=1, keepdim=True)
        z = torch.tensor([0, 0, 1],
                         dtype=torch.float64
                         ).reshape(1, 3, 1).repeat(normal.shape[0], 1, 1)
        boo = (torch.linalg.vecdot(normal, z, dim=1) < 1e-2).flatten()
        z[boo, 1] = 1.0
        z[boo, 2] = 0.0

        x = torch.cross(normal, z, dim=1)
        x /= x.norm(dim=1, keepdim=True)

        y = torch.cross(normal, x, dim=1)
        y /= y.norm(dim=1, keepdim=True)

        return x, y

    def compute_circle_pt(self, c, x_hat, y_hat, r, t):
        t = t.reshape(-1, 1, 1).repeat(1, 3, 1)
        pt = c + r * torch.cos(t) * x_hat + r * torch.sin(t) * y_hat

        return pt

    def dist(self, p1, p2):
        return (p1 - p2).norm(dim=1, keepdim=True)

    def init_guess(self, c1, r1, x1, y1, z1, c2, r2, x2, y2, z2):
        v = c2 - c1
        v /= v.norm(dim=1, keepdim=True)

        v1 = v - torch.linalg.vecdot(v, z1, dim=1).unsqueeze(1) * z1
        v1 /= v1.norm(dim=1, keepdim=True)
        pt1 = c1 + r1 * v1
        pt1_x = torch.linalg.vecdot(x1, v1, dim=1).unsqueeze(1)
        pt1_y = torch.linalg.vecdot(y1, v1, dim=1).unsqueeze(1)
        t1 = torch.atan2(pt1_y, pt1_x)

        v2 = (-v) - torch.linalg.vecdot(-v, z2, dim=1) * z2
        v2 /= v2.norm(dim=1, keepdim=True)
        pt2 = c2 + r2 * v2
        pt2_x = torch.linalg.vecdot(x2, v2, dim=1).unsqueeze(1)
        pt2_y = torch.linalg.vecdot(y2, v2, dim=1).unsqueeze(1)
        t2 = torch.atan2(pt2_y, pt2_x)

        return pt1, pt2, t1, t2

    def init_vars(self, c1, r1, z1, c2, r2, z2):
        self.c1, self.r1, self.z1 = c1, r1, z1
        self.c2, self.r2, self.z2 = c2, r2, z2

        self.x1, self.y1 = self.compute_circle_axes(z1)
        self.x2, self.y2 = self.compute_circle_axes(z2)

        init_guesses = self.init_guess(c1, r1, self.x1, self.y1, z1, c2, r2, self.x2, self.y2, z2)
        # self.t[0], self.t[1] = (torch.rand((1, 1)) - 0.5) * 2 * torch.pi, (torch.rand((1, 1)) - 0.5) * 2 * torch.pi
        self.t[0], self.t[1] = init_guesses[2], init_guesses[3]

    def opt(self, num_iter, c1, r1, z1, c2, r2, z2):
        self.init_vars(c1, r1, z1, c2, r2, z2)

        p1, p2 = None, None
        best_dist, best_iter = 99999, 0
        lr = 0.1
        for j, n in enumerate([num_iter, num_iter]):
            lr /= 10
            optimizer = torch.optim.Adam(lr=lr, params=self.t)

            for i in range(n):
                p1 = self.compute_circle_pt(self.c1, self.x1, self.y1, self.r1, self.t[0])
                p2 = self.compute_circle_pt(self.c2, self.x2, self.y2, self.r2, self.t[1])
                d = self.dist(p1, p2)

                if d.detach().item() < best_dist:
                    best_dist = d.detach().item()
                    best_iter = j * num_iter + i

                d.backward()

                optimizer.step()
                optimizer.zero_grad()

        angles = [self.t[0].data, self.t[1].data]
        pts = [p1.detach(), p2.detach()]

        return best_dist, best_iter, angles, pts


class MinSumCableLengths(torch.nn.Module):

    def __init__(self, rod_names, cable_edges, weights=None):
        super().__init__()
        self.rod_names = rod_names
        self.cable_edges = cable_edges

    # def min_sum_lens(self,
    #                  num_iter,
    #                  prev_end_pts_dict,
    #                  curr_end_pts_dict,
    #                  prev_sites_dict):
    #     self.angle_deltas = torch.nn.Parameter(
    #         torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64).reshape([3, 1, 1])
    #     )
    #     self.optimizer = torch.optim.Adam(lr=0.01, params=self.parameters())
    #
    #     quats, prev_poses, curr_poses, curr_prins\
    #         = self.compute_quat_pos(prev_end_pts_dict, curr_end_pts_dict)
    #
    #     new_sites = None
    #     for i in range(num_iter):
    #         new_sites = self.compute_new_sites(prev_sites_dict,
    #                                            quats,
    #                                            curr_prin_dict,
    #                                            prev_pos_dict,
    #                                            curr_pos_dict)
    #
    #         # dists = []
    #         # for j, (s0, s1) in enumerate(self.cable_edges_idxs):
    #         #     site0, site1 = new_sites[s0], new_sites[s1]
    #         #     d = self.weights[j] * (site1 - site0).norm(dim=1, keepdim=True)
    #         #     dists.append(d)
    #         dists = (new_sites[self.cable_edges_idxs[:, 0]] - new_sites[self.cable_edges_idxs[:, 1]]).norm(dim=1)
    #
    #         total_dist = sum(dists) / sum(self.weights)
    #         total_dist.backward()
    #
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()
    #
    #     # print(total_dist.detach().item(), [a.detach().item() for a in self.angle_deltas.values()])
    #     # angles = {k: prev_angles_dict[k] + v.detach() for k, v in self.angle_deltas.items()}
    #     # return angles
    #     return new_sites

    def compute_new_sites(self,
                          prev_sites,
                          quats,
                          curr_prins,
                          prev_poses,
                          curr_poses):
        n = prev_sites.shape[0] // len(self.rod_names)
        quats = quats.repeat(1, n, 1).reshape(-1, 4, 1)
        curr_prins = curr_prins.repeat(1, n, 1).reshape(-1, 3, 1)
        prev_poses = prev_poses.repeat(1, n, 1).reshape(-1, 3, 1)
        curr_poses = curr_poses.repeat(1, n, 1).reshape(-1, 3, 1)

        ang_dels = (self.angle_deltas / 2).repeat(1, n, 1).reshape(-1, 1, 1)
        q2 = torch.hstack([torch.cos(ang_dels), curr_prins * torch.sin(ang_dels)])

        new_sites = torch_quaternion.rotate_vec_quat(quats, prev_sites - prev_poses)
        new_sites = torch_quaternion.rotate_vec_quat(q2, new_sites) + curr_poses

        return new_sites

    def compute_quat_pos(self, prev_end_pts_dict, curr_end_pts_dict):
        quats, prev_poses, curr_poses, curr_prins = [], [], [], []
        for k in prev_end_pts_dict.keys():
            prev_end_pts = prev_end_pts_dict[k]
            curr_end_pts = curr_end_pts_dict[k]

            prev_pos = (prev_end_pts[1] + prev_end_pts[0]) / 2
            curr_pos = (curr_end_pts[1] + curr_end_pts[0]) / 2

            prev_prin = (prev_end_pts[1] - prev_end_pts[0])
            prev_prin /= prev_prin.norm(dim=1, keepdim=True)
            curr_prin = (curr_end_pts[1] - curr_end_pts[0])
            curr_prin /= curr_prin.norm(dim=1, keepdim=True)

            n = torch.cross(prev_prin, curr_prin, dim=1)
            n /= n.norm(dim=1, keepdim=True)

            ang = torch.linalg.vecdot(prev_prin, curr_prin, dim=1).unsqueeze(1)
            ang = torch.acos(ang) / 2
            q = torch.hstack([torch.cos(ang), n * torch.sin(ang)])

            prev_poses.append(prev_pos)
            curr_poses.append(curr_pos)
            quats.append(q)
            curr_prins.append(curr_prin)

        prev_poses = torch.vstack(prev_poses)
        curr_poses = torch.vstack(curr_poses)
        quats = torch.vstack(quats)
        curr_prins = torch.vstack(curr_prins)

        return quats, prev_poses, curr_poses, curr_prins

    def _compute_new_sites(self, prev_end_pts_dict, curr_end_pts_dict, prev_sites_dict):
        new_sites = {}
        for k in prev_end_pts_dict.keys():
            prev_sites = prev_sites_dict[k]

            prev_end_pts = prev_end_pts_dict[k]
            curr_end_pts = curr_end_pts_dict[k]

            prev_pos = (prev_end_pts[1] + prev_end_pts[0]) / 2
            curr_pos = (curr_end_pts[1] + curr_end_pts[0]) / 2

            prev_prin = (prev_end_pts[1] - prev_end_pts[0])
            prev_prin /= prev_prin.norm(dim=1, keepdim=True)
            curr_prin = (curr_end_pts[1] - curr_end_pts[0])
            curr_prin /= curr_prin.norm(dim=1, keepdim=True)

            n = torch.cross(prev_prin, curr_prin, dim=1)
            n /= n.norm(dim=1, keepdim=True)

            ang = torch.linalg.vecdot(prev_prin, curr_prin, dim=1).unsqueeze(1)
            ang = torch.acos(ang) / 2
            q = torch.hstack([torch.cos(ang), n * torch.sin(ang)])

            ang_del = (self.angle_deltas[k] / 2).reshape(1, 1, 1)
            q2 = torch.hstack([torch.cos(ang_del), curr_prin * torch.sin(ang_del)])

            for s in prev_sites.keys():
                # if s not in self.site_names:
                #     continue

                prev_site = prev_sites[s]
                new_site = torch_quaternion.rotate_vec_quat(q, prev_site - prev_pos)
                new_site = torch_quaternion.rotate_vec_quat(q2, new_site) + curr_pos
                new_sites[s] = new_site

        return new_sites


class MinDistConstraints(MinSumCableLengths):

    def __init__(self, rod_names, cable_edges, weights=None):
        super().__init__(rod_names, cable_edges, weights)
        self.loss_fn = torch.nn.MSELoss()
        self.rest_lengths = [
            1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
            1.75, 1.75, 1.75
        ]
        self.cable_edges = [
            ("s_3_5", "s_5_3"),
            ("s_1_3", "s_3_1"),
            ("s_1_5", "s_5_1"),
            ("s_0_2", "s_2_0"),
            ("s_0_4", "s_4_0"),
            ("s_2_4", "s_4_2"),
            ("s_2_5", "s_5_2"),
            ("s_0_3", "s_3_0"),
            ("s_1_4", "s_4_1"),
        ]
        self.rod_cables = {
            "0": [
                ("s_0_2", "s_2_0"),
                ("s_0_4", "s_4_0"),
                ("s_0_3", "s_3_0")
            ],
            "1": [
                ("s_1_3", "s_3_1"),
                ("s_1_5", "s_5_1"),
                ("s_1_4", "s_4_1"),
            ],
            "2": [
                ("s_2_4", "s_4_2"),
                ("s_2_5", "s_5_2"),
                ("s_2_0", "s_0_2"),
            ],
            "3": [
                ("s_3_5", "s_5_3"),
                ("s_3_1", "s_1_3"),
                ("s_3_0", "s_0_3"),
            ],
            "4": [
                ("s_4_0", "s_0_4"),
                ("s_4_2", "s_2_4"),
                ("s_4_1", "s_1_4"),
            ],
            "5": [
                ("s_3_5", "s_5_3"),
                ("s_1_5", "s_5_1"),
                ("s_2_5", "s_5_2")
            ]
        }

        # self.loss_fn = torch.nn.SmoothL1Loss(beta=0.1)

    def optimize(self,
                 num_iter,
                 prev_end_pts_dict,
                 curr_end_pts_dict,
                 prev_sites_dict,
                 dist_dict,
                 dt):
        gt_dists = torch.vstack([
            dist_dict[e0 + "_" + e1] for e0, e1 in self.cable_edges
        ])

        num_rods = len(self.rod_names)
        prev_sites, mapping = [], {}
        i = 0
        for k, v in prev_sites_dict.items():
            for kk, vv in v.items():
                prev_sites.append(vv)
                mapping[kk] = i
                i += 1
        prev_sites = torch.vstack(prev_sites)

        cable_idxs = torch.tensor([[mapping[e0], mapping[e1]] for e0, e1 in self.cable_edges])

        self.angle_deltas = torch.nn.Parameter(
            torch.tensor([0.0] * num_rods, dtype=torch.float64).reshape(num_rods, 1, 1)
        )
        self.optimizer = torch.optim.Adam(lr=0.01, params=self.parameters())

        quats, prev_poses, curr_poses, curr_prins \
            = self.compute_quat_pos(prev_end_pts_dict, curr_end_pts_dict)

        new_sites = None
        best_loss = 99999
        for i in range(num_iter):
            new_sites = self.compute_new_sites(prev_sites,
                                               quats,
                                               curr_prins,
                                               prev_poses,
                                               curr_poses)

            # dists = []
            # for j, (s0, s1) in enumerate(self.cable_edges_idxs):
            #     site0, site1 = new_sites[s0], new_sites[s1]
            #     d = self.weights[j] * (site1 - site0).norm(dim=1, keepdim=True)
            #     dists.append(d)

            x = new_sites[cable_idxs[:, 0]] - new_sites[cable_idxs[:, 1]]
            dists = x.norm(dim=1, keepdim=True)
            dirs = x / dists
            reg = 0.0001 / dt * self.angle_deltas.abs().mean()
            reg2 = 0.001 * ((self.angle_deltas[0] - self.angle_deltas[1]).abs()
                            + (self.angle_deltas[0] - self.angle_deltas[2]).abs()
                            + (self.angle_deltas[2] - self.angle_deltas[1]).abs())
            reg
            # reg3 = 0.01 * (dists.sum() - 6 * 1.4 - 3 * 1.75)

            loss = self.loss_fn(dists, gt_dists) + reg + reg2
            loss.backward()

            if loss.detach().item() < best_loss:
                best_loss = loss.detach().item()
                new_sites_dict = {}
                for k, v in prev_sites_dict.items():
                    new_sites_dict[k] = {}
                    for kk in v.keys():
                        i = mapping[kk]
                        new_sites_dict[k][kk] = new_sites[i: i + 1].detach().clone()

            self.optimizer.step()
            self.optimizer.zero_grad()

        # new_sites_dict = {}
        # for k, v in prev_sites_dict.items():
        #     new_sites_dict[k] = {}
        #     for kk in v.keys():
        #         i = mapping[kk]
        #         new_sites_dict[k][kk] = new_sites[i: i + 1].detach()

        return new_sites_dict, best_loss

    def _optimize(self,
                  num_iter,
                  prev_end_pts_dict,
                  curr_end_pts_dict,
                  prev_sites_dict,
                  dist_dict,
                  step=0):
        gt_distss = torch.vstack([
            dist_dict[e0 + "_" + e1] for e0, e1 in self.cable_edges
        ])

        num_rods = len(self.rod_names)
        prev_sites, mapping = [], {}
        i = 0
        order = []
        for k, v in prev_sites_dict.items():
            for kk, vv in v.items():
                order.append(kk)
                prev_sites.append(vv)
                mapping[kk] = i
                i += 1
        prev_sites = torch.vstack(prev_sites)

        cable_idxs = torch.tensor([[mapping[e0], mapping[e1]] for e0, e1 in self.cable_edges])

        quats, prev_poses, curr_poses, curr_prins \
            = self.compute_quat_pos(prev_end_pts_dict, curr_end_pts_dict)

        self.angle_deltas = torch.nn.ParameterDict({
            k: torch.tensor(0.0, dtype=torch.float64) for k in self.rod_names
        })
        self.optimizer = torch.optim.Adam(lr=0.01, params=self.parameters())

        new_sites = None
        for i in range(num_iter):
            new_sites = self.compute_new_sites(prev_end_pts_dict,
                                               curr_end_pts_dict,
                                               prev_sites_dict)

            new_sites_tensor = torch.vstack([new_sites[k] for k in order])
            self.angle_deltass = torch.vstack([a.reshape(1, 1, 1) for a in self.angle_deltas.values()])
            new_sites2 = self._compute_new_sites(prev_sites,
                                                 quats,
                                                 curr_prins,
                                                 prev_poses,
                                                 curr_poses)

            dists = []
            gt_dists = []
            for j, (s0, s1) in enumerate(self.cable_edges):
                key = s0 + "_" + s1
                gt_d = dist_dict[key]

                site0, site1 = new_sites[s0], new_sites[s1]
                d = (site1 - site0).norm(dim=1, keepdim=True)

                dists.append(d)
                gt_dists.append(gt_d)

            dists = torch.vstack(dists)
            gt_dists = torch.vstack(gt_dists)

            distss = (new_sites2[cable_idxs[:, 0]] - new_sites2[cable_idxs[:, 1]]).norm(dim=1, keepdim=True)

            reg = (0.01) * torch.vstack(list(self.angle_deltas.values())).abs().mean()

            loss = self.loss_fn(dists, gt_dists) + reg
            loss.backward()

            # print(loss.detach().item())

            self.optimizer.step()
            self.optimizer.zero_grad()

        # print(loss.detach().item(), [a.detach().item() for a in self.angle_deltas.values()])
        # angles = {k: prev_angles_dict[k] + v.detach() for k, v in self.angle_deltas.items()}
        # return angles
        new_sites_dict = {}
        for k, v in prev_sites_dict.items():
            new_sites_dict[k] = {}
            for kk in v.keys():
                new_sites_dict[k][kk] = new_sites[kk].detach()

        return new_sites_dict

    def guassian_noise(self, mean, std):
        torch.normal(mean, std)


def plot_learning_curve(loss_txt_path: Path):
    import matplotlib.pyplot as plt

    trains, vals, rollouts = [], [], []
    with loss_txt_path.open("r") as fp:
        for i, line in enumerate(fp.readlines()):
            try:
                split = line.split(",")
                epoch = int(split[0][6:])
                train = float(split[1][30:-1])
                val = float(split[2][2:-1])
                rollout = float(split[3][2:-2])

                trains.append([i, train])
                vals.append([i, val])
                rollouts.append([i, rollout])
            except:
                ignore = 0

    vals = [[v[0], v[1]] for v in vals if v[1] != -1]
    rollouts = [[v[0], v[1]] for v in rollouts if v[1] not in [-1, -99, -999]]

    plt.figure()
    x, y = zip(*trains[:-7])
    plt.plot(x, y, label='train')

    x, y = zip(*vals)
    plt.plot(x, y, label='val')

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train/Val Learning Curves")
    # plt.show()
    plt.savefig(loss_txt_path.parent / "train_val_lc.png")

    plt.figure()
    x, y = zip(*rollouts[:])
    plt.plot(x, y, label='rollout')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Rollout Learning Curves")
    plt.yscale('log')
    # plt.show()
    plt.savefig(loss_txt_path.parent / "rollout_lc.png")
    print(y)


if __name__ == '__main__':
    path = Path('/Users/nelsonchen/Documents/Rutgers-CS-PhD/Research/tensegrity/data_sets/tensegrity_real_datasets'
                '/mjc_syn_models/6d5d_normflow_v2/loss.txt')
    plot_learning_curve(path)
