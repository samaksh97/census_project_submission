import os
import tempfile

import hydra
import mlflow
from omegaconf import DictConfig

_ALL_STEP = [
    'data_cleaning',
    'training',
]


@hydra.main(config_name='hydra_config')
def run(config: DictConfig):

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _ALL_STEP

    with tempfile.TemporaryDirectory() as _:

        root_path = hydra.utils.get_original_cwd()

        if "data_cleaning" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(root_path, "src/data_clean"),
                "main",
                parameters={
                    "input_data_path": config["data_process"]["input_data_path"],  # NOQA: E501
                    "output_data_path": config['data_process']["output_data_path"]  # NOQA: E501
                },
            )
        if "training" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(root_path, "src/training"),
                "main",
                parameters={
                    "input_data_path": config["train"]["train_data_path"],
                    "output_model_path": config['train']["output_model_path"]
                },
            )


if __name__ == "__main__":
    run()
