from legged_gym.envs import PointFootRoughCfg, PointFootRoughCfgPPO


class PointFootFlatCfg(PointFootRoughCfg):
    class env(PointFootRoughCfg.env):
        num_privileged_obs = 27 + 2 + 2
        num_propriceptive_obs = 27 + 2 + 2 # plus 2 dof vel(wheels) and 2 actions(wheels)
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
            heading = [-1.0, 1.0]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-3.14, 3.14]
    
    class init_state(PointFootRoughCfg.init_state):
        # pos = [0.0, 0.0, 0.8] # origin
        pos = [0.0, 0.0, 0.7 + 0.1664]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }   
    
    class control(PointFootRoughCfg.control):
        control_type = "P_AND_V" # P: position, V: velocity, T: torques. 
                                 # P_AND_V: some joints use position control 
                                 # and others use vecocity control.
        # PD Drive parameters:
        stiffness = {
            "abad_L_Joint": 40,
            "hip_L_Joint": 40,
            "knee_L_Joint": 40,
            "abad_R_Joint": 40,
            "hip_R_Joint": 40,
            "knee_R_Joint": 40,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 1.8,
            "hip_L_Joint": 1.8,
            "knee_L_Joint": 1.8,
            "abad_R_Joint": 1.8,
            "hip_R_Joint": 1.8,
            "knee_R_Joint": 1.8,
            "wheel_L_Joint": 0.5,
            "wheel_R_Joint": 0.5,
        }  # [N*m*s/rad]
        # action scale: target angle = actionscale * action + defaultangle
        # action_scale_pos is the action scale of joints that use position control
        # action_scale_vel is the action scale of joints that use velocity control
        action_scale_pos = 0.25
        action_scale_vel = 8
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4       

    class asset(PointFootRoughCfg.asset):
        foot_name = "wheel"
        foot_radius = 0.127
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]       
        replace_cylinder_with_capsule = False       
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    
    class domain_rand(PointFootRoughCfg.domain_rand):
        friction_range = [0.2, 1.6]
        added_mass_range = [-0.5, 2]
        
    class rewards(PointFootRoughCfg.rewards):
        class scales(PointFootRoughCfg.rewards.scales):
            # base class
            action_rate = -0.01
            ang_vel_xy = 0.0 # off
            base_height = -20.0
            collision = -10.0
            dof_acc = -2.5e-07
            dof_pos_limits = -2.0
            dof_vel = 0.0 # off
            feet_air_time = 0.0 # off
            feet_contact_forces = 0.0 # off
            feet_stumble = 0.0 # off
            lin_vel_z = 0.0 # off
            no_fly = 1.0
            orientation = -5.0 # 很重要，不加的话会导致存活时间下降
            stand_still = -1.0
            termination = 0.0 # off
            torque_limits = 0.0 # off
            torques = -2.5e-05
            tracking_ang_vel = 5.0 # off
            tracking_lin_vel = 10.0
            unbalance_feet_air_time = 0.0 # off
            unbalance_feet_height = 0.0 # off
            feet_distance = 5 # -100
            survival = 0.1
            # new added
            wheel_adjustment = 1.0 # 1.0 off
            inclination = 0.0 # off

        base_height_target = 0.65
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        min_feet_distance = 0.29
        max_feet_distance = 0.32
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        base_height_target = 0.65 + 0.1664
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001

class PointFootFlatCfgPPO(PointFootRoughCfgPPO):
    class policy(PointFootRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner(PointFootRoughCfgPPO.runner):
        experiment_name = 'pointfoot_flat'
        max_iterations = 10000
