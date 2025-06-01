# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *


from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, print_welcome_message
from legged_gym.utils.Zlog import zzs_basic_graph_logger

if os.path.exists("./legged_gym/envs/CustomEnvironments"):
    from legged_gym.envs.CustomEnvironments import *

import numpy as np
import torch

import inspect


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 3.0
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    stop_state_log = 300  # number of steps before plotting states
    robot_index = [0, 3]  # which robot is used for logging
    joint_index = 0  # which joint is used for logging

    if USE_ZZS_LOGGER:
        logger = zzs_basic_graph_logger(
            dt=env.dt,
            max_episode_length=stop_state_log,
            action_dim=env.num_actions,
            observation_dim=env.num_obs,
            dof_names=env.dof_names,
            num_agents=len(robot_index),
        )
    else:
        logger = Logger(env.dt)

    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    file_path = inspect.getfile(env.__class__)
    directory = os.path.dirname(file_path)
    print(f"Playing {args.task} from {directory}")

    # run policy
    print("Running policy...")
    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    args.task,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            if type(logger) is zzs_basic_graph_logger:
                measured_height = env.root_states[:, 2].unsqueeze(1) - env.measured_heights
                measured_height = measured_height[robot_index, 0]
                if hasattr(env, "ref_dof_pos"):
                    logger.log_state("dof_ref", env.ref_dof_pos[robot_index, :].detach().cpu().numpy())
                logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, :].detach().cpu().numpy() * env.cfg.control.action_scale,
                        "dof_pos": env.dof_pos[robot_index, :].detach().cpu().numpy(),
                        "dof_vel": env.dof_vel[robot_index, :].detach().cpu().numpy(),
                        "dof_torque": env.torques[robot_index, :].detach().cpu().numpy(),
                        "command_x": env.commands[robot_index, 0].detach().cpu().numpy(),
                        "command_y": env.commands[robot_index, 1].detach().cpu().numpy(),
                        "command_yaw": env.commands[robot_index, 2].detach().cpu().numpy(),
                        "base_vel_x": env.base_lin_vel[robot_index, 0].detach().cpu().numpy(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].detach().cpu().numpy(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].detach().cpu().numpy(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].detach().cpu().numpy(),
                        "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                        "base_height": measured_height.detach().cpu().numpy(),
                        # "base_angle_x": env.base_euler_xyz[robot_index, 0].detach().cpu().numpy(),
                        # "base_angle_y": env.base_euler_xyz[robot_index, 1].detach().cpu().numpy(),
                        # "base_angle_z": env.base_euler_xyz[robot_index, 2].detach().cpu().numpy(),
                    }
                )
            else:
                if isinstance(robot_index, list):
                    print("logger do NOT support robot_index as list input, change to robot 0 instead!")
                    robot_index = 0
                logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                        "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                        "dof_torque": env.torques[robot_index, joint_index].item(),
                        "command_x": env.commands[robot_index, 0].item(),
                        "command_y": env.commands[robot_index, 1].item(),
                        "command_yaw": env.commands[robot_index, 2].item(),
                        "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                        "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    }
                )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    USE_ZZS_LOGGER = True
    args = get_args()
    print_welcome_message()
    play(args)
