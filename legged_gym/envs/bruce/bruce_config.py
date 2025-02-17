from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BruceRoughCfg(LeggedRobotCfg):
    class init_state( LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.48] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #    'hip_yaw_r' : 0. ,   
        #    'hip_roll_r' : 0,               
        #    'hip_pitch_r' : 0.111,         
        #    'knee_pitch_r' : -0.264,       
        #    'ankle_pitch_r' : 0.154,     
        #    'hip_yaw_l' : 0., 
        #    'hip_roll_l' : 0, 
        #    'hip_pitch_l' : 0.111,                                       
        #    'knee_pitch_l' : -0.264,                                             
        #    'ankle_pitch_l' : 0.154,                                     
        # }
        pos = [0.0, 0.0, 0.48] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #    'hip_yaw_r' : 0. ,   
        #    'hip_roll_r' : 0,               
        #    'hip_pitch_r' : 0.442,         
        #    'knee_pitch_r' : -1.02,       
        #    'ankle_pitch_r' : 0.578,     
        #    'hip_yaw_l' : 0., 
        #    'hip_roll_l' : 0, 
        #    'hip_pitch_l' : 0.442,                                       
        #    'knee_pitch_l' : -1.02,                                             
        #    'ankle_pitch_l' : 0.578,                                     
        # }

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'hip_yaw_r' : 0. ,   
           'hip_roll_r' : 0,               
           'hip_pitch_r' : 0,         
           'knee_pitch_r' : 0,       
           'ankle_pitch_r' : 0,     
           'hip_yaw_l' : 0., 
           'hip_roll_l' : 0, 
           'hip_pitch_l' : 0.,                                       
           'knee_pitch_l' : 0,                                             
           'ankle_pitch_l' : 0.,                                     
        }
    
    class env(LeggedRobotCfg.env):
        # 3 + 3 + 3 + 10 + 10 + 10 + 2 = 41
        num_observations = 41
        num_privileged_obs = 44
        num_actions = 10
      

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.3

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        # stiffness = {'hip_yaw': 150,
        #              'hip_roll': 150,
        #              'hip_pitch': 150,
        #              'knee': 200,
        #              'ankle': 40,
        #              'torso': 300,
        #              'shoulder': 150,
        #              "elbow":100,
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 2,
        #              'hip_roll': 2,
        #              'hip_pitch': 2,
        #              'knee': 4,
        #              'ankle': 2,
        #              'torso': 6,
        #              'shoulder': 2,
        #              "elbow":2,
        #              }  # [N*m/rad]  # [N*m*s/rad]

        # stiffness = {'hip_yaw': 15,
        #              'hip_roll': 15,
        #              'hip_pitch': 15,
        #              'knee': 30,
        #              'ankle': 0.1,
        #              }  # [N*m/rad]
        stiffness = {'hip_yaw': 0,
                     'hip_roll': 0,
                     'hip_pitch': 0,
                     'knee': 0,
                     'ankle': 0,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 0,
                     'hip_roll': 0,
                     'hip_pitch': 0,
                     'knee': 0,
                     'ankle': 0,
                     }
        # damping = {  'hip_yaw': 0.1,
        #              'hip_roll': 0.1,
        #              'hip_pitch': 0.1,
        #              'knee': 0.12,
        #              'ankle': 0.0016,
        #              }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bruce/urdf/bruce3.urdf'
        name = "bruce"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        # terminate_after_contacts_on = ["base_link", "knee", "hip"]
        terminate_after_contacts_on = ["base_link"]
        disable_gravity = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        # fix_base_link = True
  

    # class noise( LeggedRobotCfg.noise ):
    #     add_noise = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            feet_air_time = 0.0
            collision = -1.0
            action_rate = -0.01
            torques = 0.0
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18
            termination = -1.0

class BruceRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'bruce'

  
