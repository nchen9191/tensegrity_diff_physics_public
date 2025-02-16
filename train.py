import json

import torch

from diff_physics_engine.training.real_tensegrity_trainer import RealTensegrityTrainingEngine


def main():
    config_file_path = "diff_physics_engine/training/configs/real_3bar_sys_id_config.json"
    with open(config_file_path, 'r') as fp:
        config_file = json.load(fp)

    trainer = RealTensegrityTrainingEngine(config_file,
                                           torch.nn.MSELoss(),
                                           0.01)
    trainer.run(200)


if __name__ == '__main__':
    torch.set_anomaly_enabled(False)
    main()
