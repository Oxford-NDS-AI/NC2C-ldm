import numpy as np
import os
import ntpath
import time
import wandb

class Visualiser_wandb():
    def __init__(self, project_name, group=None, config=None):
        """
        Initializes the wandb project.

        Parameters:
        - project_name: str. The name of the wandb project.
        - entity: str, optional. The entity (team) under which the project will be saved. Default is None.
        - config: dict, optional. A dictionary of hyperparameters and their values. Default is None.
        """
        self.run = wandb.init(project=project_name, group=group, config=config, settings=wandb.Settings(start_method="fork"))
        self.config = config

    def log_metrics(self, metrics:dict, step=None):
        """
        Logs metrics to wandb.

        Parameters:
        - images: list of PIL.Image or numpy arrays. The images to log.
        - captions: list of str, optional. Captions for each image. Default is None.
        - step: int, optional. The step number at which to log the images. Useful for time series data.
        """
        wandb.log(
            data = metrics,
            step = step
        )
        
    def log_images(self, images, captions, step=None):
        wandb.log({"examples": [wandb.Image(image, caption=caption) for image, caption in zip(images, captions)]}, step=step)

    def finish(self):
        """
        Marks the end of the wandb run.
        """
        wandb.finish()