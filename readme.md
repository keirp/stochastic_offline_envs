# Stochastic Offline RL Tasks

This codebase contains only the essential code for running the stochastic offline RL environments from the paper "You Can't Count on Luck: Why Decision Transformers Fail in Stochastic Environments" ([arXiv](https://arxiv.org/abs/2205.15967)).

Tasks:
- Connect 4 vs. a stochastic opponent
- 2048 (but ends at 128 to make it feasible as an offline RL task)
- Simplistic Gambling environment as a sanity check

## Installation Instructions

- Install the gym-2048 package that is included.
- Install dependencies with `pip install -r requirements.txt`
- Run `pip install -e .` to install the `stochastic_offline_envs` package.
- Run the script `download_datasets.py` to download the datasets used in the paper.
- This repo has been tested with Gym version `0.19.0`.

## Usage Instructions
Please see `notebooks/env_example.ipynb` for details about how to load the environments and offline datasets.