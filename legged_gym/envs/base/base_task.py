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

import sys
from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.utils.Zlog import zzs_basic_graph_logger
from isaacgym.torch_utils import quat_apply
import numpy as np
import torch


# Base class for RL tasks
class BaseTask:
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.device_id = self.sim_device_id
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
            print("running simulation on cuda device: " + self.sim_device + str(self.sim_device_id))
        else:
            print("running simulation on cpu.")
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # set viewer
        self.set_viewer()

        # action_test
        self.action_test_countdown_period = 300
        self.action_test_countdown = -1
        self.action_test_random_robot_index = 0
        self.action_test_logger = zzs_basic_graph_logger(
            sim_params.dt, self.action_test_countdown_period, self.num_actions, self.num_obs, self.dof_names
        )

    def set_viewer(self):
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        self.viewer_move = gymapi.Vec3(0, 0, 0)

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "record_frames")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "move_forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "move_backward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "move_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "move_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "move_down")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "move_up")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "move_up")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT_SHIFT, "move_down")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "action_test")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "camera_position")

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    print("FUNCTION NOT IMPLEMENTED YET!")  # TODO: add record method
                elif evt.action == "move_forward":
                    self.viewer_move.z = evt.value * 0.2
                elif evt.action == "move_backward":
                    self.viewer_move.z = -evt.value * 0.2
                elif evt.action == "move_left":
                    self.viewer_move.x = evt.value * 0.2
                elif evt.action == "move_right":
                    self.viewer_move.x = -evt.value * 0.2
                elif evt.action == "move_up":
                    self.viewer_move.y = evt.value * 0.2
                elif evt.action == "move_down":
                    self.viewer_move.y = -evt.value * 0.2
                elif evt.action == "action_test" and evt.value > 0 and self.action_test_countdown < 0:
                    self.action_test_countdown = self.action_test_countdown_period
                    self.action_test_random_robot_index = np.random.randint(self.num_envs)
                    print(f"begin action test for random robot with id {self.action_test_random_robot_index}")
                elif evt.action == "camera_position" and evt.value > 0:
                    viewer_pos = self.gym.get_viewer_camera_transform(self.viewer, None)
                    viewer_trans_tensor = torch.tensor([viewer_pos.p.x, viewer_pos.p.y, viewer_pos.p.z])
                    viewer_quat_tensor = torch.tensor([viewer_pos.r.x, viewer_pos.r.y, viewer_pos.r.z, viewer_pos.r.w])
                    z = torch.tensor([0.0, 0.0, 1.0])
                    look_at = quat_apply(viewer_quat_tensor, z)
                    print("viewer position:", viewer_trans_tensor)
                    print("viewer look at:", look_at)

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)

                if (self.viewer_move.x != 0.0) or (self.viewer_move.y != 0.0) or (self.viewer_move.z != 0.0):
                    viewer_pos = self.gym.get_viewer_camera_transform(self.viewer, None)
                    viewer_trans_tensor = torch.tensor([viewer_pos.p.x, viewer_pos.p.y, viewer_pos.p.z])
                    viewer_quat_tensor = torch.tensor([viewer_pos.r.x, viewer_pos.r.y, viewer_pos.r.z, viewer_pos.r.w])
                    z = torch.tensor([0.0, 0.0, 1.0])
                    look_at = quat_apply(viewer_quat_tensor, z)
                    move_torch = torch.tensor([self.viewer_move.x, 0, self.viewer_move.z])
                    offset = quat_apply(viewer_quat_tensor, move_torch)
                    offset[2] = self.viewer_move.y + offset[2]
                    new_pos = offset + viewer_trans_tensor
                    new_look = look_at + new_pos
                    new_pos_gym = gymapi.Vec3(new_pos[0], new_pos[1], new_pos[2])
                    new_look_gym = gymapi.Vec3(new_look[0], new_look[1], new_look[2])
                    self.gym.viewer_camera_look_at(self.viewer, None, new_pos_gym, new_look_gym)

                if self.action_test_countdown >= 0:
                    self.process_action_test()

            else:
                self.gym.poll_viewer_events(self.viewer)

    def process_action_test(self):
        if self.action_test_countdown > 0:
            # print progress every 50 steps
            if self.action_test_countdown % 50 == 0:
                print(
                    f"action test percentage: {((self.action_test_countdown_period - self.action_test_countdown)/self.action_test_countdown_period * 100):.2f} %"
                )
            self.action_test_countdown -= 1
            robot_index = self.action_test_random_robot_index
            measured_height = self.root_states[:, 2].unsqueeze(1) - self.measured_heights
            measured_height = measured_height[robot_index, 0]

            if hasattr(self, "predict_lin_vel"):
                self.action_test_logger.log_state(
                    "pred_lin_vel_x", self.predict_lin_vel[robot_index, 0].detach().cpu().numpy()
                )
                self.action_test_logger.log_state(
                    "pred_lin_vel_y", self.predict_lin_vel[robot_index, 1].detach().cpu().numpy()
                )
                self.action_test_logger.log_state(
                    "pred_lin_vel_z", self.predict_lin_vel[robot_index, 2].detach().cpu().numpy()
                )

            if hasattr(self, "ref_dof_pos"):
                self.action_test_logger.log_state("dof_ref", self.ref_dof_pos[robot_index, :].detach().cpu().numpy())
            self.action_test_logger.log_states(
                {
                    "dof_pos_target": self.actions[robot_index, :].detach().cpu().numpy()
                    * self.cfg.control.action_scale,
                    "dof_pos": self.dof_pos[robot_index, :].detach().cpu().numpy(),
                    "dof_vel": self.dof_vel[robot_index, :].detach().cpu().numpy(),
                    "dof_torque": self.torques[robot_index, :].detach().cpu().numpy(),
                    "command_x": self.commands[robot_index, 0].detach().cpu().numpy(),
                    "command_y": self.commands[robot_index, 1].detach().cpu().numpy(),
                    "command_yaw": self.commands[robot_index, 2].detach().cpu().numpy(),
                    "base_vel_x": self.base_lin_vel[robot_index, 0].detach().cpu().numpy(),
                    "base_vel_y": self.base_lin_vel[robot_index, 1].detach().cpu().numpy(),
                    "base_vel_z": self.base_lin_vel[robot_index, 2].detach().cpu().numpy(),
                    "base_vel_yaw": self.base_ang_vel[robot_index, 2].detach().cpu().numpy(),
                    "contact_forces_z": self.contact_forces[robot_index, self.feet_indices, 2].cpu().numpy(),
                    "base_height": measured_height.detach().cpu().numpy(),
                    "base_angle_x": self.base_euler_xyz[robot_index, 0].detach().cpu().numpy(),
                    "base_angle_y": self.base_euler_xyz[robot_index, 1].detach().cpu().numpy(),
                    "base_angle_z": self.base_euler_xyz[robot_index, 2].detach().cpu().numpy(),
                }
            )
        elif self.action_test_countdown == 0:
            print("action test finished")
            self.action_test_logger.plot_states()
            self.action_test_countdown = -1
            self.action_test_logger.reset()
        else:
            pass  # do nothing if self.action_test_countdown < 0
