import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    init_pos = [0.0, 0.0, 1.5]  # Initial position of the robot
    init_ori = [0.0, 0.0, 0.0, 1.0]  # Initial orientation of the robot
    env_cfg.init_state.pos = init_pos
    env_cfg.init_state.rot = init_ori

    env_cfg.env.test = True

    # prepare environment
    if MOVE_CAMERA:
        args.num_envs=1
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        # print(env.base_pos)
        if MOVE_CAMERA:
            position = env.base_pos[0].clone()
            position[2] += 2
            position[0] += 2
            position[1] += 2
            # print(f'position: {position}')
            lookat = env.base_pos[0]
            # print(f'lookat', lookat)
            env.set_camera(position, lookat)
            # pos = [10, 0, 6]  # [m]
            # lookat = [11., 5, 3.]  # [m]
            # env.set_camera(pos, position)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
