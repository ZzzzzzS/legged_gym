import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class zzs_basic_graph_logger:
    def __init__(self, dt, max_episode_length, action_dim, observation_dim, dof_names, num_agents=1):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.max_episode_length = max_episode_length
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.num_episodes = 0
        self.plot_process = None
        self.dof_names = dof_names
        self.num_agents = num_agents
        self.__init_buffer()

    def __init_buffer(self):
        self.buffer = {
            "dof_pos_target": np.zeros((self.max_episode_length, self.num_agents, self.action_dim)),
            "dof_pos": np.zeros((self.max_episode_length, self.num_agents, self.action_dim)),
            "dof_ref": np.zeros((self.max_episode_length, self.num_agents, self.action_dim)),
            "dof_vel": np.zeros((self.max_episode_length, self.num_agents, self.action_dim)),
            "dof_torque": np.zeros((self.max_episode_length, self.num_agents, self.action_dim)),
            "command_x": np.zeros((self.max_episode_length, self.num_agents)),
            "command_y": np.zeros((self.max_episode_length, self.num_agents)),
            "command_yaw": np.zeros((self.max_episode_length, self.num_agents)),
            "base_vel_x": np.zeros((self.max_episode_length, self.num_agents)),
            "base_vel_y": np.zeros((self.max_episode_length, self.num_agents)),
            "base_vel_z": np.zeros((self.max_episode_length, self.num_agents)),
            "pred_lin_vel_x": np.zeros((self.max_episode_length, self.num_agents)),
            "pred_lin_vel_y": np.zeros((self.max_episode_length, self.num_agents)),
            "pred_lin_vel_z": np.zeros((self.max_episode_length, self.num_agents)),
            "base_vel_yaw": np.zeros((self.max_episode_length, self.num_agents)),
            "contact_forces_z": np.zeros((self.max_episode_length, self.num_agents, 2)),
            "base_height": np.zeros((self.max_episode_length, self.num_agents)),
            "base_angle_x": np.zeros((self.max_episode_length, self.num_agents)),
            "base_angle_y": np.zeros((self.max_episode_length, self.num_agents)),
            "base_angle_z": np.zeros((self.max_episode_length, self.num_agents)),
        }
        self.buffer_idx = 0

    def log_state(self, key, value):
        if key in self.buffer:
            self.buffer[key][self.buffer_idx, :] = value
        else:
            print(f"Warning: key {key} not in buffer")
            self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)
        self.buffer_idx += 1

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()
        del self.buffer
        self.__init_buffer()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def __plot_action(self, robot_index):
        nb_rows = self.action_dim
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.suptitle(f"Robot {robot_index} action")
        time = np.linspace(0, self.max_episode_length * self.dt, self.max_episode_length)
        for i in range(self.action_dim):
            a = axs[i, 0]
            a.plot(time, self.buffer["dof_pos_target"][:, robot_index, i], label="target")
            a.plot(time, self.buffer["dof_pos"][:, robot_index, i], label="measured")
            if "dof_ref" in self.buffer.keys():
                a.plot(time, self.buffer["dof_ref"][:, robot_index, i], label="measured")
            a.set(xlabel="time [s]", ylabel="Position [rad]", title=f"DOF Position {self.dof_names[i]}")
            a.legend()
            a = axs[i, 1]
            a.plot(time, self.buffer["dof_pos_target"][:, robot_index, i], label="target")
            a.plot(time, self.buffer["dof_vel"][:, robot_index, i], label="measured")
            a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title=f"Joint Velocity {self.dof_names[i]}")
            a.legend()
            a = axs[i, 2]
            a.plot(time, self.buffer["dof_torque"][:, robot_index, i], label="measured")
            a.set(xlabel="time [s]", ylabel="Torque [Nm]", title=f"Joint Torque {self.dof_names[i]}")
            a.legend()

    def __plot_info(self, robot_index):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.suptitle(f"Robot {robot_index} info")
        time = np.linspace(0, self.max_episode_length * self.dt, self.max_episode_length)
        a = axs[0, 0]
        a.plot(time, self.buffer["command_x"][:, robot_index], label="commanded")
        a.plot(time, self.buffer["base_vel_x"][:, robot_index], label="measured")
        if "pred_lin_vel_x" in self.buffer.keys():
            a.plot(time, self.buffer["pred_lin_vel_x"][:, robot_index], label="predicted")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        a.legend()
        a = axs[0, 1]
        a.plot(time, self.buffer["command_y"][:, robot_index], label="commanded")
        a.plot(time, self.buffer["base_vel_y"][:, robot_index], label="measured")
        if "pred_lin_vel_y" in self.buffer.keys():
            a.plot(time, self.buffer["pred_lin_vel_y"][:, robot_index], label="predicted")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity y")
        a.legend()
        a = axs[0, 2]
        a.plot(time, self.buffer["command_yaw"][:, robot_index], label="commanded")
        a.plot(time, self.buffer["base_vel_yaw"][:, robot_index], label="measured")
        a.set(xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base velocity yaw")
        a.legend()
        a = axs[1, 0]
        a.plot(time, self.buffer["base_vel_z"][:, robot_index], label="measured")
        if "pred_lin_vel_z" in self.buffer.keys():
            a.plot(time, self.buffer["pred_lin_vel_z"][:, robot_index], label="predicted")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity z")
        a.legend()
        a = axs[1, 1]
        forces = self.buffer["contact_forces_z"][:, robot_index]
        for i in range(forces.shape[1]):
            a.plot(time, forces[:, i], label=f"force {i}")
        a.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        a.legend()
        a = axs[1, 2]
        a.plot(time, self.buffer["base_height"][:, robot_index], label="measured")
        a.set(xlabel="time [s]", ylabel="height [m]", title="Base height")
        a.legend()
        a = axs[2, 0]
        a.plot(time, self.buffer["base_angle_x"][:, robot_index], label="measured")
        a.set(xlabel="time [s]", ylabel="angle [rad]", title="base_angle_x")
        a.legend()
        a = axs[2, 1]
        a.plot(time, self.buffer["base_angle_y"][:, robot_index], label="measured")
        a.set(xlabel="time [s]", ylabel="angle [rad]", title="base_angle_y")
        a.legend()
        a = axs[2, 2]
        a.plot(time, self.buffer["base_angle_z"][:, robot_index], label="measured")
        a.set(xlabel="time [s]", ylabel="angle [rad]", title="base_angle_z")
        a.legend()

    def _plot(self):
        for robot_index in range(self.num_agents):
            self.__plot_action(robot_index)
            self.__plot_info(robot_index)
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
