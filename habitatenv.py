import os
import random
import math
import pickle

import transformations_tf as tft

import numpy as np
import magnum as mn
from PIL import Image
from settings import default_sim_settings, make_cfg
from habitat_sim.scene import SceneNode

from utils.frame_utils import read_gen
import matplotlib.pyplot as plt

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_magnum,
    quat_to_magnum,
)

class HabitatEnv():
    def __init__(self, folder, init_state, depth_type):
        scene_glb = folder + "/" + os.path.basename(folder).capitalize() + ".glb"
        self._cfg = make_cfg(scene_glb)
        self.init_common(init_state)
        self.depth_type = depth_type
        agent_node = self._sim.agents[0].scene_node
        self.agent_object_id = self._sim.add_object(1, agent_node)
        self._sim.set_object_motion_type(
            habitat_sim.physics.MotionType.KINEMATIC, self.agent_object_id
        )
        assert (
        self._sim.get_object_motion_type(self.agent_object_id)
        == habitat_sim.physics.MotionType.KINEMATIC
        )
        # Saving Start Frame
        observations = self._sim.get_sensor_observations()
        self.save_color_observation(observations, 0, 0, folder)
        self.save_depth_observation(observations, 0, 0, folder)

        self.noise = False
        self.translation_noise = pickle.load(open("actuation_noise_fwd.pkl", 'rb'))
        self.rotation_left_noise = pickle.load(open("actuation_noise_left.pkl", 'rb'))
        self.rotation_right_noise = pickle.load(open("actuation_noise_right.pkl", 'rb'))
       
    def init_common(self, init_state):
        self._sim = habitat_sim.Simulator(self._cfg)
        random.seed(default_sim_settings["seed"])
        self._sim.seed(default_sim_settings["seed"])
        start_state = self.init_agent_state(default_sim_settings["default_agent"], init_state)
        return start_state
        
    def init_agent_state(self, agent_id, init_state):
        start_state = habitat_sim.agent.AgentState()
        x, y, z, w, p, q, r = init_state
        start_state.position = np.array([x, y, z]).astype('float32')
        start_state.rotation = np.quaternion(w,p,q,r)
        agent = self._sim.initialize_agent(agent_id, start_state)
        start_state = agent.get_state()
        return start_state
    
    def get_agent_pose(self):
        agent = self._sim._default_agent
        state = agent.get_state()
        position = state.position
        rotation = state.rotation
        pose = [position[0], position[1], position[2], rotation.w, rotation.x, rotation.y, rotation.z]
        return pose

    def save_color_observation(self, obs, frame, step, folder):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save(folder + "/results/test.rgba.%05d.%05d.png" % (frame, step))
        color_img = read_gen(folder + "/results/test.rgba.%05d.%05d.png" % (frame, step))
        if self.depth_type == 'FLOW':
            if frame == 1:
                prev_color_img = read_gen(folder + "/results/test.rgba.%05d.%05d.png" % (frame-1, step-1))
                return color_img, prev_color_img
            elif frame > 1:
                prev_color_img = read_gen(folder + "/results/test.rgba.%05d.%05d.png" % (frame-1, step))
                return color_img, prev_color_img
            else:
                return color_img, color_img
        elif self.depth_type == 'TRUE':
            return color_img
    
    def save_depth_observation(self, obs, frame, step, folder):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        depth_img.save(folder + "/results/test.depth.%05d.%05d.png" % (frame, step))
        depth_img = plt.imread(folder + "/results/test.depth.%05d.%05d.png" % (frame, step))
        return depth_img

    def agent_controller(self, agent, velocity):
        vel_control = self._sim.get_object_velocity_control(self.agent_object_id)
        print("Normal Velocity:", velocity)
        if self.noise:
            '''
            t_noise = self.translation_noise.sample()[0][0] # 1 x 3 []
            velocity[0] += t_noise[0]
            velocity[1] += t_noise[0]
            velocity[2] += t_noise[0]
            velocity[3] += (t_noise[1] - t_noise[2])
            velocity[4] += (t_noise[1] - t_noise[2])
            velocity[5] += (t_noise[1] - t_noise[2])

            noise_left = self.rotation_left_noise.sample()[0][0]
            noise_right = self.rotation_right_noise.sample()[0][0]
            r_noise = noise_left - noise_right

            velocity[0] += r_noise[0]
            velocity[1] += r_noise[0]
            velocity[2] += r_noise[0]
            velocity[3] += (r_noise[1] - r_noise[2])
            velocity[4] += (r_noise[1] - r_noise[2])
            velocity[5] += (r_noise[1] - r_noise[2])
            '''
            noise = np.random.normal(0, 0.1, 6)
            velocity += noise
            #0.05, 0.1

            print("Noise Velocity:", velocity)
        vel_control.linear_velocity = np.array(velocity[0:3])
        vel_control.angular_velocity = np.array(velocity[3:])
        vel_control.controlling_lin_vel = True
        vel_control.controlling_ang_vel = True
        # step with world time
        self._sim.step_physics(0.00416)
        vel_control.lin_vel_is_local = True
        vel_control.ang_vel_is_local = True
    
    def example(self, vel, frame=1,folder=''):
        '''
        vel : n x 6 velocity vector
        '''
        vel[:, 2] = -vel[:, 2] # conventions Z axis
        vel[:, 1] = -vel[:, 1] # conventions Y axis

        vel[:, 5] = -vel[:, 5] # conventions Z axis
        vel[:, 4] = -vel[:, 4] # conventions Y axis
        #vel[:, 0] = -vel[:, 0] # conventions

        agent_id = default_sim_settings["default_agent"]
        agent = self._sim._default_agent

        color_img = None
        depth_img = None
        for i in range(vel.shape[0]):
            state = agent.get_state()
            self.agent_controller(agent, vel[i])
            observations = self._sim.get_sensor_observations()
            if self.depth_type == 'FLOW':
                color_img , prev_color_img = self.save_color_observation(observations, frame, i + 1, folder)
            elif self.depth_type == 'TRUE':
                color_img = self.save_color_observation(observations, frame, i + 1, folder)
            depth_img = self.save_depth_observation(observations, frame, i + 1, folder)
        
        if self.depth_type == 'TRUE':
            return color_img, depth_img
        elif self.depth_type == 'FLOW':
            return color_img, prev_color_img , depth_img

    def end_sim(self):
        self._sim.close()
        del self._sim
