import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Union

import torch
import tqdm
from torch.nn.modules.loss import _Loss

from diff_physics_engine.simulators.tensegrity_simulator import TensegrityRobotSimulator
from diff_physics_engine.state_objects.base_state_object import BaseStateObject
from diff_physics_engine.state_objects.rods import RodState
from utilities.misc_utils import save_curr_code


class AbstractRealTensegrityTrainingEngine(BaseStateObject):

    def __init__(self,
                 training_config: Dict,
                 criterion: _Loss,
                 dt: Union[float, torch.Tensor]):
        """

        :param trainable_params: Dict of torch.nn.Parameter(s) that can be updated by backprop
        :param other_params: Dict of torch.Tensors, other params needed for simulator that are not trainable
        :param simulator: function with step() to take in current state and external forces/pts to produce
                          next state
        :param optim_params: Optimizer parameters for Adam
        :param criterion: Loss function
        """
        super().__init__("tensegrity simulator")

        self.sim_config = training_config['sim_config']
        if isinstance(training_config['sim_config'], str):
            with open(training_config['sim_config'], "r") as j:
                self.sim_config = json.load(j)

        self.simulator = TensegrityRobotSimulator.init_from_config_file(self.sim_config)

        train_params, params_ranges = self._build_and_set_trainable_params(
            training_config["trainable_params"],
            torch.float64
        )

        self.trainable_params = torch.nn.ParameterDict(train_params)
        self.sim_trainable_params_ranges = params_ranges

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          **training_config['optimizer_params'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=1.0)
        self.criterion = criterion

        self.data_json, self.target_gaits = self.load_trajectory_data(training_config['data_path'])
        self.gt_end_pts = self.get_endpts()
        self.dt = dt

        self.num_gaits = training_config['num_gaits']
        self.start_state = None

        self.output_dir = Path(training_config['output_path'])
        self.output_dir.mkdir(exist_ok=True)

        self.device = "cpu"  # default
        self.best_loss = 9999

        self.save_code()
        self.init_simulator()
        # self.simulator.grad_descent_rest_lengths()

    def save_code(self, save_code_flag=True):
        if save_code_flag:
            code_dir_name = "tensegrity_diff_physics_engine"
            curr_code_dir = os.getcwd()
            code_output = Path(self.output_dir, code_dir_name)
            save_curr_code(curr_code_dir, code_output)

    def _build_and_set_trainable_params(self,
                                        trainable_params_dict: Dict,
                                        dtype: torch.dtype = torch.float64):
        """
        Method to build trainable_params dict

        :param trainable_params_dict:
        :param dtype: torch dtype precision

        :return: Dict of trainable params
        """
        trainable_params, params_ranges = {}, {}
        for param_dict in trainable_params_dict:
            param_name = param_dict['param_name']
            key = param_name

            init_val = torch.rand(1, dtype=dtype)

            if "range" in param_dict:
                params_ranges[key] = param_dict['range']
                min_val, max_val = param_dict['range']
                init_val = init_val * (max_val - min_val) + min_val

            if 'init_val' in param_dict:
                init_val = torch.tensor(param_dict['init_val'], dtype=dtype)

            torch_param = torch.nn.Parameter(init_val.type(dtype))
            trainable_params[key] = torch_param

        return trainable_params, params_ranges

    def set_sim_trainable_params(self) -> None:
        """
        See parent. In SpringRodSimulator, there is a set_params function to be used to set parameters
        """
        for key, val in self.trainable_params.items():
            if key == 'simulator_parameters':
                continue
            split = key.split("|")
            self.set_attribute(split, val)

    def init_simulator(self):
        # self.simulator = TensegrityRobotSimulator.init_from_config_file(self.sim_config)
        self.simulator.reset_actuation()
        # self.set_sim_trainable_params()

        start_endpts = self.gt_end_pts[0]
        start_endpts = [
            [start_endpts[0].detach(), start_endpts[1].detach()],
            [start_endpts[2].detach(), start_endpts[3].detach()],
            [start_endpts[4].detach(), start_endpts[5].detach()],
        ]

        self.simulator.init_by_endpts(start_endpts)
        # self.simulator2.init_by_endpts(start_endpts)
        self.start_state = self.simulator.get_curr_state()
        # if self.start_state is None:
        #     self.start_state = self.simulator.run_until_stable(max_time=30, dt=0.001, tol=1e-2)

    def get_endpts(self):
        with torch.no_grad():
            dtype = torch.float64
            # dtype = self.simulator.sys_precision
            end_pts = [
                [
                    torch.tensor(d['rod_01_end_pt1'], dtype=dtype).reshape(1, 3, 1),
                    torch.tensor(d['rod_01_end_pt2'], dtype=dtype).reshape(1, 3, 1),
                    torch.tensor(d['rod_23_end_pt1'], dtype=dtype).reshape(1, 3, 1),
                    torch.tensor(d['rod_23_end_pt2'], dtype=dtype).reshape(1, 3, 1),
                    torch.tensor(d['rod_45_end_pt1'], dtype=dtype).reshape(1, 3, 1),
                    torch.tensor(d['rod_45_end_pt2'], dtype=dtype).reshape(1, 3, 1)
                ] for d in self.data_json
            ]
        return end_pts

    def load_trajectory_data(self, path):
        with Path(path, "processed_data.json").open('r') as fp:
            data_json = json.load(fp)

        with Path(path, "target_gaits.json").open('r') as fp:
            target_gaits_json = json.load(fp)

        return data_json, target_gaits_json

    def move_tensors(self, device):
        self.device = device
        self.to(device)
        # self.other_params = {k: v.to(device) for k, v in self.other_params.items()}
        self.simulator.move_tensors(device)

    def compute_loss(self, pred_endpts) -> torch.Tensor:
        """
        Method to do custom loss computation if needed

        :param pred_y: Predicted y-value
        :param gt_y: Ground truth y-value
        :return: Loss tensor
        """

        gt_endpts = torch.vstack(
            [torch.hstack(self.gt_end_pts[v['idx'] - 1])
             for v in self.target_gaits[1:self.num_gaits + 1]]
            + ([torch.hstack(self.gt_end_pts[-1])]
               if self.num_gaits + 1 > len(self.target_gaits) else [])
        )

        loss = self.criterion(pred_endpts, gt_endpts)
        return loss

    def _batch_compute_end_pts(self, batch_state: torch.Tensor, sim=None) -> torch.Tensor:
        """
        Compute end pts for entire batch

        :param batch_state: batch of states
        :return: batch of endpts
        """
        if sim is None:
            sim = self.simulator

        end_pts = []
        for i, rod in enumerate(sim.rigid_bodies.values()):
            state = batch_state[:, i * 13: i * 13 + 7]
            principal_axis = RodState.compute_principal_axis(state[:, 3:7])
            end_pts.extend(RodState.compute_end_pts_from_state(state, principal_axis, rod.length))

        return torch.concat(end_pts, dim=1)

    def forward(self) -> torch.Tensor:
        """
        Forward method in torch.nn.Module

        :param batch_data: input data for batch processing
        :return:
        """
        # torch.set_printoptions(8)
        self.init_simulator()
        # self.simulator.reset_actuation()
        curr_state = self.start_state.clone()
        states, pred_key_endpts = [], []
        states.append({"time": 0.0, "state": curr_state.detach().flatten().numpy().tolist()})
        global_steps = 0
        for target_gait in tqdm.tqdm(self.target_gaits[:self.num_gaits]):
            for pid in self.simulator.pids.values():
                pid.RANGE = torch.tensor(target_gait['info']['RANGE'] / 100, dtype=curr_state.dtype)
            self.simulator.reset_pids()
            target_gait_dict = {f"spring_{i}": g
                                for i, g in enumerate(target_gait['target_gait'])}
            controls = torch.ones(len(target_gait_dict), dtype=self.simulator.sys_precision)
            step = 0
            while (controls != 0.0).any():
                # for _ in range(num_steps[k]):
                step += 1
                global_steps += 1
                curr_state, controls = self.simulator.step_with_target_gait(
                    curr_state,
                    self.dt,
                    target_gait_dict=target_gait_dict
                )
                states.append({
                    'time': self.dt * global_steps,
                    'state': curr_state.detach().flatten().numpy().tolist()
                })

                controls = torch.hstack(controls).detach()
            pred_key_endpts.append(self._batch_compute_end_pts(curr_state))

        pred_key_endpts = torch.vstack(pred_key_endpts)

        return states, pred_key_endpts

    def backward(self, loss: torch.Tensor) -> None:
        """
        Run back propagation with loss tensor

        :param loss: torch.Tensor
        """
        if loss.grad_fn is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
            self.optimizer.step()
            self.optimizer.zero_grad()

        for param_name, param in self.trainable_params.items():
            if param_name in self.sim_trainable_params_ranges:
                ranges = self.sim_trainable_params_ranges[param_name]
                param.data.clamp_(ranges[0], ranges[1])

        self.optimizer.zero_grad()

    def run_one_epoch(self):
        """
        Run one epoch over dataloader

        :return: Average loss float
        """

        # Run forward over batch of data to get loss
        states, pred_endpts = self.forward()
        loss = self.compute_loss(pred_endpts)

        return loss

    def train_epoch(self, epoch_num, data_name="") -> float:
        """
        Run one epoch cycle over training data and validation data. If val_data_loader not provided,
        will default to using train_data_loader as validation data.

        :param train_data_loader: torch DataLoader
        :param val_data_loader: torch DataLoader

        :return: train_loss (float), val_loss (float)
        """
        # If val_data_loader not provided, will default to using train_data_loader as validation data.
        # Compute train loss
        train_loss = self.run_one_epoch()
        train_loss_ = train_loss.detach().item()

        self.log_status(train_loss_, epoch_num, data_name)

        # If gradient updates required, run backward pass
        self.backward(train_loss)

        return train_loss_

    def log_status(self, train_loss: float, epoch_num: int, data_name="") -> None:
        """
        Method to print training status to console

        :param train_loss: Training loss
        :param epoch_num: Current epoch
        """
        # latest_values = {k: v[-1] for k, v in param_values.items()}
        param_values = {k: v.detach().item()
                        for k, v in self.trainable_params.items()
                        if k != "simulator_parameters"}
        loss_file = Path(self.output_dir, "loss.txt")
        msg = f'Epoch {epoch_num}, Data: {data_name} "Train Loss": {train_loss}, "param_values": {param_values}\n'

        try:
            with loss_file.open('a') as fp:
                fp.write(msg)
        except:
            with loss_file.open('w') as fp:
                fp.write(msg)

        print(msg)
        print()

    def run(self, num_epochs: int) -> Tuple[List, Dict]:
        """
        Method to run entire training

        :param num_epochs: Number of epochs to train
        :return: List of train losses, List of val losses, dictionary of trainable params and list of losses
        """
        # Initialize simulator
        # self.init_simulator()

        # Initialize by running evaluation over train and validation loss
        # with torch.no_grad():
        #     init_train_loss = self.run_one_epoch(grad_required=False)

        # Initialize storing objects
        # train_losses, val_losses = [0], [0]
        init_loss = self.run_one_epoch(1, grad_required=False)
        train_losses = [init_loss]
        # param_values_dict = {}
        param_values_dict = {k: [v.detach().item()] for k, v in self.trainable_params.items()}

        # Run training over num_epochs
        for n in range(num_epochs):
            # Run single epoch training and evaluation
            train_loss = self.train_epoch(n + 1)
            # self.scheduler.step()

            # Store current epoch's values
            # train_losses.append(train_loss)

            for k, v in self.trainable_params.items():
                param_values_dict[k].append(v.detach().item())

        return train_losses, param_values_dict


class RealTensegrityTrainingEngine(AbstractRealTensegrityTrainingEngine):

    def __init__(self,
                 training_config: Dict,
                 criterion: _Loss,
                 dt: Union[float, torch.Tensor]):
        super(
            AbstractRealTensegrityTrainingEngine,
            self
        ).__init__("tensegrity gnn simulator")

        self.config = training_config
        self.sim_config = training_config['sim_config']
        if isinstance(training_config['sim_config'], str):
            with open(training_config['sim_config'], "r") as j:
                self.sim_config = json.load(j)

        self.load_simulator(training_config)
        # self.curr_state_dict = self.simulator.state_dict()

        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          **training_config['optimizer_params'])
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
        #                                                         gamma=1.0)

        train_paths = training_config['train_data_paths']
        random.shuffle(train_paths)

        self.train_data_jsons, self.train_target_gaits \
            = self.load_trajectory_data(train_paths)
        self.train_data_names = [Path(p).name for p in train_paths]

        self.val_data_jsons, self.val_target_gaits \
            = self.load_trajectory_data(training_config['val_data_paths'])
        self.val_data_names = [Path(p).name for p in training_config['val_data_paths']]

        self.data_json = self.train_data_jsons[0]
        self.target_gaits = self.train_target_gaits[0]

        self.gt_end_pts = self.get_endpts()
        self.dt = dt
        self.curr_dataset_name = None

        self.num_gaits = training_config['num_gaits'] if 'num_gaits' in training_config else -1
        self.start_states_dict = {
            n: torch.tensor(d[0]['stable_start'], dtype=torch.float64).reshape(1, -1, 1)
            for n, d in zip(self.train_data_names + self.val_data_names,
                            self.train_data_jsons + self.val_data_jsons)
            if 'stable_start' in d[0]
        }
        self.act_len_dict = {
            n: [torch.tensor(a, dtype=torch.float64) for a in d[0]['init_act_lens']]
            for n, d in zip(self.train_data_names + self.val_data_names,
                            self.train_data_jsons + self.val_data_jsons)
            if 'init_act_lens' in d[0]
        }

        self.output_dir = Path(training_config['output_path'])
        Path(self.output_dir).mkdir(exist_ok=True)

        self.device = "cpu"  # default
        self.best_loss = 1e9

        self.save_code()

        # self.init_simulator()

    def load_trajectory_data(self, paths):
        data_jsons, target_gaits_jsons = [], []
        for path in paths:
            with Path(path, "processed_data.json").open('r') as fp:
                data_jsons.append(json.load(fp))

            with Path(path, "target_gaits.json").open('r') as fp:
                target_gaits_jsons.append(json.load(fp))

        return data_jsons, target_gaits_jsons

    def assign_traj_data(self, data_json, target_gait):
        self.data_json = data_json
        self.target_gaits = target_gait
        self.gt_end_pts = self.get_endpts()
        self.num_gaits = len(self.target_gaits)

    def get_pid_params(self, name):
        if "ccw" in name:
            min_length = 1.0
            range_ = 1.0
            tol = 0.1
        elif "rolling" in name:
            min_length = 0.8
            range_ = 1.0
            tol = 0.1
        elif "cw" in name:
            min_length = 0.7
            range_ = 1.2
            tol = 0.1
        else:
            min_length = 1.0
            range_ = 1.0
            tol = 0.1

        return min_length, range_, tol

    def assign_pid_params(self, name):
        min_length, range_, tol = self.get_pid_params(name)

        self.data_json[0]['pid'] = {
            "min_length": min_length,
            "RANGE": range_,
            "tol": tol,
            "motor_speed": 0.7
        }
    def init_simulator(self):
        # self.simulator.detach()
        # self.simulator = TensegrityGNNResidualSimulator(**self.sim_config)
        # self.simulator.load_state_dict(self.curr_state_dict)
        self.simulator.curr_graph = None
        self.simulator.reset_actuation()

        cables = self.simulator.tensegrity_robot.actuated_cables.values()
        dtype = self.simulator.sys_precision

        start_endpts = self.gt_end_pts[0]
        for e in start_endpts:
            e[:, 2] += 0.05

        start_endpts_ = [
            [start_endpts[0].detach(), start_endpts[1].detach()],
            [start_endpts[2].detach(), start_endpts[3].detach()],
            [start_endpts[4].detach(), start_endpts[5].detach()],
        ]
        self.simulator.init_by_endpts(start_endpts_)
        for i, cable in enumerate(cables):
            e0, e1 = cable.end_pts
            e0, e1 = start_endpts[int(e0.split("_")[1])], start_endpts[int(e0.split("_")[2])]
            cable_length = (e1 - e0).norm(dim=1, keepdim=True)

            cable.actuation_length = cable._rest_length - cable_length
            cable.motor.speed = torch.tensor(self.data_json[0]['pid']['motor_speed'], dtype=dtype)
            cable.motor.motor_state.omega_t = torch.tensor(0., dtype=dtype)

        for pid in self.simulator.pids.values():
            pid_params = self.data_json[0]['pid']
            pid.min_length = torch.tensor(pid_params['min_length'], dtype=dtype)
            pid.RANGE = torch.tensor(pid_params['RANGE'], dtype=dtype)
            pid.tol = torch.tensor(pid_params['tol'], dtype=dtype)
            pid.k_p = 10
        # self.start_state = self.simulator.get_curr_state().clone()
        if self.curr_dataset_name not in self.start_states_dict:
            # self.simulator.max_correction_norm = 1e-12
            robot = deepcopy(self.simulator.tensegrity_robot)
            sim = TensegrityRobotSimulator(robot,
                                           self.simulator.gravity,
                                           self.sim_config['contact_params'],
                                           self.simulator.sys_precision)

            for c in sim.tensegrity_robot.actuated_cables.values():
                c.motor.speed = torch.tensor(0.1, dtype=torch.float64)

            for p in sim.pids.values():
                p.tol = torch.tensor(0.01, dtype=torch.float64)
                p.min_length = torch.tensor(1.0, dtype=torch.float64)
                p.RANGE = torch.tensor(1.0, dtype=torch.float64)
                p.k_p = 10

            sim.init_by_endpts(start_endpts_)

            # sim.grad_descent_rest_lengths()
            curr_state, all_states = sim.run_until_stable(max_time=30, dt=0.001, tol=1e-2)
            frames = [{'time': i * 0.01, 'pos': s.reshape(-1, 13)[:, :7].flatten().numpy()}
                      for i, s in enumerate(all_states)]

            # visualizer = MuJoCoVisualizer()
            # visualizer.set_xml_path(Path("mujoco_physics_engine/xml_models/3prism_real_upscaled_vis.xml"))
            # visualizer.data = frames
            # visualizer.set_camera("camera")
            # visualizer.visualize(Path(f"/Users/nelsonchen/Desktop/vid.mp4"), 0.01)

            target_gaits = {}
            for i, s in enumerate(sim.tensegrity_robot.actuated_cables.values()):
                e0, e1 = s.end_pts
                e0, e1 = int(e0.split("_")[1]), int(e0.split("_")[2])
                cable_length = (start_endpts[e1] - start_endpts[e0]).norm(dim=1, keepdim=True)

                gait = (cable_length.item() - pid_params['min_length']) / pid_params['RANGE']
                target_gaits[f'spring_{i}'] = gait

            states, _ = sim.run_target_gait(0.001, curr_state, target_gaits)
            sim.update_state(states[-1])

            print(states[-1].flatten().numpy().tolist())

            tensegrity_left_mid = torch.vstack([e[0] for e in start_endpts_]).mean(dim=0, keepdim=True)
            tensegrity_right_mid = torch.vstack([e[1] for e in start_endpts_]).mean(dim=0, keepdim=True)
            tensegrity_prin = tensegrity_right_mid - tensegrity_left_mid
            tensegrity_prin /= tensegrity_prin.norm(dim=1, keepdim=True)
            tensegrity_com = torch.vstack(start_endpts).mean(dim=0, keepdim=True)
            start_state_ = sim.tensegrity_robot.align_prin_axis_2d(tensegrity_prin, tensegrity_com)
            start_state_ = start_state_.reshape(-1, 13, 1)

            # all_states.extend(states)
            #
            # all_states = [{'time': 0.001 * j, 'pos': s.reshape(-1, 13)[:, :7].flatten().numpy().tolist()}
            #               for j, s in enumerate(all_states)][::10]
            # with Path(self.output_dir, f"{self.curr_dataset_name}_init_frames.json").open('w') as fp:
            #     json.dump(all_states, fp)

            # start_state_ = states[-1].reshape(-1, 13, 1)
            self.start_state = torch.hstack([start_state_[:, :7],
                                             torch.zeros_like(start_state_[:, 7:])
                                             ]).reshape(1, -1, 1)
            self.act_lengths = [c.actuation_length.detach()
                                for c in sim.tensegrity_robot.actuated_cables.values()]
            for i, c in enumerate(self.simulator.tensegrity_robot.actuated_cables.values()):
                c.actuation_length = self.act_lengths[i].detach().clone()
            print(self.start_state.flatten().numpy().tolist())
            print([a.item() for a in self.act_lengths])

            curr_endpts = [e for r in sim.tensegrity_robot.rods.values() for e in r.end_pts]
            print(((torch.vstack(start_endpts) - torch.vstack(curr_endpts)) ** 2).mean().item())

            self.start_states_dict[self.curr_dataset_name] = self.start_state.detach().clone()
            self.act_len_dict[self.curr_dataset_name] = self.act_lengths
        else:
            self.simulator.reset_actuation()
            self.start_state = self.start_states_dict[self.curr_dataset_name].detach().clone()
            for i, c in enumerate(self.simulator.tensegrity_robot.actuated_cables.values()):
                c.actuation_length = self.act_len_dict[self.curr_dataset_name][i].detach().clone()
        # self.simulator.max_correction_norm = 1e-3

    def forward(self) -> Tuple[torch.Tensor, List]:
        """
        Forward method in torch.nn.Module

        :param batch_data: input data for batch processing
        :return:
        """
        self.init_simulator()
        # self.simulator.reset_actuation()
        curr_state = self.start_state.clone()
        states, pred_key_endpts = [], []
        # states = []
        # pred_key_endpts = torch.zeros((self.num_gaits, 18, 1), dtype=curr_state.dtype)
        total_steps = 0
        curr_time = 0.0

        for i in tqdm.tqdm(list(range(self.num_gaits))):
            target_gait = self.target_gaits[i]
            for pid in self.simulator.pids.values():
                pid.RANGE = torch.tensor(target_gait['info']['RANGE'] / 100, dtype=curr_state.dtype)

            self.simulator.reset_pids()
            target_gait_dict = {
                f"spring_{i}": gait
                for i, gait in enumerate(target_gait['target_gait'])
            }
            gait_time = self._get_gait_time(
                target_gait['idx'],
                (self.target_gaits[i + 1]['idx']
                 if i < len(self.target_gaits) - 1
                 else len(self.data_json) - 1)
            )
            # max_steps = 10
            max_steps = int(math.ceil(gait_time / self.dt))
            controls = torch.ones(len(target_gait_dict), dtype=self.simulator.sys_precision)
            step = 0
            while (controls != 0.0).any():
                # for _ in range(max_steps):
                total_steps += 1
                curr_state, controls = self.simulator.step_with_target_gait(
                    curr_state,
                    self.dt,
                    target_gait_dict=target_gait_dict
                )
                # states.append(curr_state.detach())

                controls = torch.hstack(controls).detach()
            # print(step)
            # pred_key_endpts[i] = self._batch_compute_end_pts(curr_state)
            pred_key_endpts.append(self._batch_compute_end_pts(curr_state))
            # print(pred_key_endpts[-1].squeeze())

        pred_key_endpts = torch.vstack(pred_key_endpts)
        # print(total_steps)

        return states, pred_key_endpts

    def _get_gait_time(self, start_idx, end_idx):
        t0 = self.data_json[start_idx]['time']
        t1 = self.data_json[end_idx]['time']
        return t1 - t0

    def run(self, num_epochs: int) -> Tuple[List, Dict]:
        train_losses = self.compute_init_loss()

        param_values_dict = {k: v.detach().item()
                             for k, v in self.trainable_params.items()
                             if k != "simulator_parameters"}
        # Run training over num_epochs
        avg_train_loss = 0.0
        for n in range(num_epochs):
            idx = n % len(self.train_data_jsons)
            self.curr_dataset_name = self.train_data_names[idx]
            self.assign_traj_data(self.train_data_jsons[idx],
                                  self.train_target_gaits[idx])
            self.assign_pid_params(self.train_data_names[idx])
            # print(self.train_data_names[idx])
            # Run single epoch training and evaluation
            train_loss = self.train_epoch(n + 1, self.train_data_names[idx])
            avg_train_loss += train_loss
            # self.log_status(train_loss, n + 1, "Average Train")
            # self.scheduler.step()

            if idx == len(self.train_data_jsons) - 1:
                avg_train_loss /= len(self.train_data_jsons)
                self.log_status(avg_train_loss, -1, "Average Train")
                self.eval()
                avg_train_loss = 0

            # Store current epoch's values
            train_losses.append(train_loss)

        return train_losses, param_values_dict

    def eval(self):
        with torch.no_grad():
            val_loss = 0.0
            for i in range(len(self.val_data_jsons)):
                self.curr_dataset_name = self.val_data_names[i]
                self.assign_traj_data(self.val_data_jsons[i], self.val_target_gaits[i])
                self.assign_pid_params(self.val_data_names[i])
                # print(self.val_data_names[i])
                loss = self.run_one_epoch()
                loss_ = loss.item()
                val_loss += loss_
                self.log_status(loss_, -1, "Val " + self.val_data_names[i])
            val_loss /= len(self.val_data_jsons)

        self.log_status(val_loss, -1, "Avg Val")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(
                self.simulator,
                self.output_dir / "best_model.pt"
            )

        return val_loss

    def load_simulator(self, training_config):
        train_params, params_ranges = self._build_and_set_trainable_params(
            training_config["trainable_params"],
            torch.float64
        )

        self.simulator = TensegrityRobotSimulator.init_from_config_file(self.sim_config)

        self.trainable_params = torch.nn.ParameterDict(train_params)
        self.sim_trainable_params_ranges = params_ranges
        self.set_sim_trainable_params()

    def compute_init_loss(self):
        with torch.no_grad():
            total_loss = 0
            for i in range(len(self.train_data_jsons)):
                self.curr_dataset_name = self.train_data_names[i]
                self.assign_traj_data(self.train_data_jsons[i],
                                      self.train_target_gaits[i])
                self.assign_pid_params(self.train_data_names[i])
                loss = self.run_one_epoch()
                # print(self.train_data_names[i])
                self.log_status(loss.item(), 0, self.train_data_names[i])
                total_loss += loss

            total_loss /= len(self.train_data_jsons)
            self.log_status(total_loss, 0, "Average Train")

            # total_loss = 0
            # for i in range(len(self.val_data_jsons)):
            #     self.curr_dataset_name = self.val_data_names[i]
            #     self.assign_traj_data(self.val_data_jsons[i],
            #                           self.val_target_gaits[i])
            #     self.assign_pid_params(self.val_data_names[i])
            #     loss = self.run_one_epoch(0, grad_required=False)
            #     # print(self.train_data_names[i])
            #     self.log_status(loss.item(), 0, self.val_data_names[i])
            #     total_loss += loss
            #
            # total_loss /= len(self.val_data_jsons)
            self.eval()
            # self.log_status(total_loss, 0, "Average Val")

            train_losses = []

        return train_losses
