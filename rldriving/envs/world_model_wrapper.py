import gym
import os
import tensorflow as tf
import numpy as np
from gym import spaces
from rldriving.vae.cvae import CVAE
from rldriving.memory.memory import Memory


results_dir = '../'
path_to_vision = '../vae/models/'

memory_params = {
    'input_dim': 13,  # latent_size + num_actions
    'output_dim': 8,  # latent size
    'num_timesteps': 1,
    'batch_size': 100,
    'epochs': 1,  # 1
    'batch_per_epoch':1,
    'lstm_nodes': 256,
    'num_mix': 5,
    'grad_clip': 1.0,
    'initial_learning_rate': 0.001,
    'end_learning_rate': 0.00001,
    'load_model': True,
    'results_dir': os.path.join(results_dir, 'memory')
}

vae_params = {
    'latent_dim': 8
}

class WorldModelWrapper(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    def __init__(self, env):
        super(WorldModelWrapper, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(269, ), dtype=np.float32)
        self.env = env
    
        # Load vision
        vision_name = 'vae_model_7500K_8'
        self.vision = CVAE(**vae_params)
        self.vision.load(path_to_vision, vision_name)
        
        # Load memory
        self.memory = Memory(**memory_params)
        
        # Initialise LSTM hidden state
        self.state = self.memory.lstm.get_zero_hidden_state(np.zeros(13).reshape(1, 1, 13))
    
    def apply_vision_memory(self, observation, reset_mode=False, action=None):
        
        # Transform observations to appropriate formats
        # Lidar
        lidar = np.array(observation['lidar']).astype('float32') + 1.0 / 2.0
        lidar = np.reshape(lidar, (1, 32, 32, 1))
        # Steering
        steering = np.array(observation['steering']).astype('float32').reshape(1, 1)
        # Velocity
        velocity = np.array(observation['a']).astype('float32').reshape(1, 1)
        # Acceleration
        acceleration = np.array(observation['a']).astype('float32').reshape(1, 1)
        
        # Action
        if reset_mode:
            action=self.env.action_space.sample()
            action = np.array(action).astype('float32').reshape(1, 2)
        if action is not None:
            action = np.array(action).astype('float32').reshape(1, 2)
        
        # Apply vision to get an encoded latent space vector Z
        mu, logvar = self.vision.encode(lidar)
        z = self.vision.reparameterize(mu, logvar)
        
        # Concatenate other observations into one vector
        x = tf.concat((z, steering, velocity, acceleration, action), axis=1)
        
        # Run LSTM and Gaussian Mixture
        y, h_state, c_state = self.memory(x, self.state, temperature=1.0)
        self.state = [h_state, c_state]
        
        # Modify initial observation
        observation = tf.concat([x, h_state], axis=1).numpy().reshape((-1, ))
        
        return observation
        
    
    def step(self, action):       
        # Make a step in the original gym-metacar environment
        observation, reward, done, info = self.env.step(action)
        # Apply vision and memory
        return self.apply_vision_memory(observation, action=action), reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        observation['a'] = 0
        observation['steering'] = 0
        return self.apply_vision_memory(observation, reset_mode=True)  # reward, done, info can't be included
    
    def render(self, mode='human'):
        env.render()
    
    def close (self):
        env.close()