import torch

from experiments.simulation_runner import run_traj


if __name__ == '__main__':
    with torch.no_grad():
        run_traj()
