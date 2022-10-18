# Stochastic Offline RL Tasks

This is a preview codebase containing only the essential code for running the offline RL environments from the paper "You Can't Count on Luck: Why Decision Transformers Fail in Stochastic Environments" ([arXiv](https://arxiv.org/abs/2205.15967)).

Tasks:
- Connect 4 vs. a stochastic opponent
- 2048 (but ends at 128 to make it feasible as an offline RL task)
- Simplistic Gambling environment as a sanity check

## Installation Instructions

- Install the gym-2048 package that is included.
- Run `pip install -e .` to install the `esper_envs` package.
- This repo has been tested with Gym version `0.19.0`.

## Usage Instructions
Please see `notebooks/env_example.ipynb` for details about how to load the environments and offline datasets.

## Known Issues / TODO
- This codebase does not include the generation code for the environment. That will come in the official release.