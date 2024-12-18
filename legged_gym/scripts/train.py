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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *

if os.path.exists("./legged_gym/envs/CustomEnvironments"):
    from legged_gym.envs.CustomEnvironments import *

from legged_gym.utils import get_args, task_registry
from legged_gym.utils import ExperimentLogger
from legged_gym.utils import train_batch
from legged_gym.utils import print_welcome_message
from legged_gym.utils.helpers import launch_tensorboard, cp_env
from legged_gym.utils.helpers import class_to_dict
import torch


def train(args):
    logdir = ExperimentLogger.generate_logdir(args.task)
    exp_msg = {}
    # NOTE: the train batch function is not implemented yet!
    if args.train_batch <= 1:  # require experiment commit message for non batched run or first time of batched run
        exp_msg = ExperimentLogger.commit_experiment(logdir, args)  # force to commit expriment message

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.train_batch != 0:
        args.headless = True  # force headless mode for batched runs
        # prepare environment
        train_batch_len = train_batch.check_env_batch_config(env_cfg)

        # check if batch size is too large
        if args.train_batch > train_batch_len:
            print(f"Batch size {args.train_batch} is too large, maximum batch size is {train_batch_len}")
            os._exit(0)

        env_cfg = train_batch.parse_env_batch_config(env_cfg, args.train_batch - 1)

    if args.launch_tensorboard:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task)
        launch_tensorboard(log_root)

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=logdir
    )
    # exp_msg["env_cfg"] = class_to_dict(env_cfg)
    # exp_msg["train_cfg"] = class_to_dict(train_cfg)
    # ExperimentLogger.save_hyper_params(logdir, env_cfg, train_cfg)

    if args.backup_env:
        cp_env(env, ppo_runner.log_dir)

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True, experiment_log=exp_msg
    )


if __name__ == "__main__":
    args = get_args()
    print_welcome_message()
    train(args)
