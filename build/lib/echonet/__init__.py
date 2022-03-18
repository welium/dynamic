"""
The echonet package contains code for loading echocardiogram videos, and
functions for training and testing segmentation and ejection fraction
prediction models.
"""

import click

from .__version__ import __version__
from .config import CONFIG as config
from . import datasets
from . import utils

@click.group()
def main():
    """Entry point for command line interface."""


del click

main.add_command(utils.new.segmentation.run)
main.add_command(utils.new.video.run)
main.add_command(utils.original.segmentation.run)
main.add_command(utils.original.video.run)

__all__ = ["__version__", "config", "datasets", "main", "utils"]
