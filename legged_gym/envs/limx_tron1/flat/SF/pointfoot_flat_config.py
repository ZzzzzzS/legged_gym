from legged_gym.envs import PointFootRoughCfg, PointFootRoughCfgPPO


class PointFootFlatCfg(PointFootRoughCfg):
    class env(PointFootRoughCfg.env):
        num_privileged_obs = 33
        num_propriceptive_obs = 33 # base_ang_vel: 3, projected_gravity: 3, dof_pos: 8, dof_vel: 8, actions: 8, commands: 3
        num_actions = 8

    class terrain(PointFootRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights_critic = False

    class commands(PointFootRoughCfg.commands):
        num_commands = 3
        heading_command = False
        resampling_time = 5.

        class ranges(PointFootRoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            heading = [0.0, 0.0]
            lin_vel_y = [-1.0, 1.0] # [-1.0, 1.0]
            ang_vel_yaw = [-0.5, 0.5]

    class init_state(PointFootRoughCfg.init_state):
        # pos = [0.0, 0.0, 0.8] # origin
        pos = [0.0, 0.0, 0.7 + 0.1664] # [0.0, 0.0, 0.7 + 0.1664]  # x,y,z [m]
        rot = [0.0, -0.1, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0,
        }

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]
        touch_down_vel = 0.0

        class ranges:
            frequencies = [1.0, 1.5]
            offsets = [0.5, 0.5]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.5, 0.15]

    class control(PointFootRoughCfg.control):
        control_type = "P_AND_V"  # P: position, V: velocity, T: torques.
        # P_AND_V: some joints use position control
        # and others use vecocity control.
        # PD Drive parameters:
        stiffness = {
            "abad_L_Joint": 45,
            "hip_L_Joint": 45,
            "knee_L_Joint": 45,
            "abad_R_Joint": 45,
            "hip_R_Joint": 45,
            "knee_R_Joint": 45,
            "ankle_L_Joint": 45,
            "ankle_R_Joint": 45,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 1.5,
            "hip_L_Joint": 1.5,
            "knee_L_Joint": 1.5,
            "abad_R_Joint": 1.5,
            "hip_R_Joint": 1.5,
            "knee_R_Joint": 1.5,
            "ankle_L_Joint": 0.8,
            "ankle_R_Joint": 0.8,
        }  # [N*m*s/rad]
        # action scale: target angle = actionscale * action + defaultangle
        # action_scale_pos is the action scale of joints that use position control
        # action_scale_vel is the action scale of joints that use velocity control
        action_scale_pos = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PointFootRoughCfg.asset):
        foot_name = "ankle"
        foot_radius = 0.0
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        replace_cylinder_with_capsule = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(PointFootRoughCfg.domain_rand):
        friction_range = [0.2, 1.6]
        added_mass_range = [-0.5, 2]

    class rewards(PointFootRoughCfg.rewards):
        class scales:
            # base class
            keep_balance = 0.0 # off

            # tracking related rewards
            tracking_lin_vel_x = 1.5
            tracking_lin_vel_y = 1.5
            tracking_ang_vel = 1
            base_height = -20.0
            stand_still = -1.0
            # survival = 0.1

            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            torques = -0.00008
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -2.0
            collision = -10
            action_smooth = -0.01
            orientation = -5.0
            feet_distance = -200
            feet_regulation = -0.05
            tracking_contacts_shaped_force = -2.0 * 0 # off
            tracking_contacts_shaped_vel = -2.0 * 0 # off
            tracking_contacts_shaped_height = -2.0 * 0 # off
            feet_contact_forces = -0.002
            ankle_torque_limits = -0.1
            power = -2e-3

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.56
        feet_height_target = 0.10
        min_feet_distance = 0.19
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005

        about_landing_threshold = 0.07


class PointFootFlatCfgPPO(PointFootRoughCfgPPO):
    class policy(PointFootRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32] # [256, 128, 64, 32]
        critic_hidden_dims = [128, 64, 32] # [256, 128, 64, 32]

    class runner(PointFootRoughCfgPPO.runner):
        experiment_name = 'pointfoot_flat'
        max_iterations = 10000
