import gym
import h5py
import ray
import time


try:
    import gym_metacar
except:
    import sys
    sys.path.append(".")
    sys.path.append("..")
    import gym_metacar
from gym_metacar.wrappers import *


@ray.remote
def rollout(i, step_limit, num_episodes, lidar_shape):
    epsilon = 0.5
    print("Creating environment ", i)
    env = gym.make("metacar-level3-continuous-v0")
    env = ClipRewardsWrapper(env)
    env = StepLimitTerminateWrapper(env, step_limit)
    
    print('Creating H5 files for observations and actions for environment ', i)
    # Create a file to write observations of the lidar into
    with h5py.File('env_' + str(i) + '_lidar.h5', 'w', libver='latest') as fo:
        fo.create_dataset('lidar', data=np.zeros((0, lidar_shape)), compression="gzip", chunks=True, maxshape=(None, 32))
    # Create a file to write actions into
    with h5py.File('env_' + str(i) + '_actions.h5', 'w', libver='latest') as fa:
        fa.create_dataset('actions', data=np.zeros((0, 2)), compression="gzip", chunks=True, maxshape=(None, 2))
    # Create a file to write steering angles
    with h5py.File('env_' + str(i) + '_steering.h5', 'w', libver='latest') as fa:
        fa.create_dataset('steering', data=np.zeros((0, 1)), compression="gzip", chunks=True, maxshape=(None, 1))
    # Create a file to write acceleration angles
    with h5py.File('env_' + str(i) + '_acceleration.h5', 'w', libver='latest') as fa:
        fa.create_dataset('acceleration', data=np.zeros((0, 1)), compression="gzip", chunks=True, maxshape=(None, 1))
    # Create a file to write acceleration angles
    with h5py.File('env_' + str(i) + '_velocity.h5', 'w', libver='latest') as fa:
        fa.create_dataset('velocity', data=np.zeros((0, 1)), compression="gzip", chunks=True, maxshape=(None, 1))

    for episode in range(num_episodes):
        print('Running episode {} in environment {}'.format(episode, i))
        env.reset()
        done = False
        step = 0
        lidar = []
        actions = []
        steering = []
        velocity = []
        acceleration = []
        if np.random.uniform(size=1) < epsilon:
            if step < np.floor(step_limit / 2):
                while done == False:
                    action = [1, env.action_space.sample()[1]]
                    obs, _, done, info = env.step(action)
                    step += 1
                    done = done
                    lidar.append(np.array(obs['lidar']))
                    steering.append(np.array(obs['steering']).reshape(1, 1))
                    velocity.append(np.array(obs['v']).reshape(1, 1))
                    acceleration.append(np.array(obs['a']).reshape(1, 1))
                    actions.append(np.array(action).reshape(1, 2))
                    
            else:
                while done == False:
                    action = [-1, env.action_space.sample()[1]]
                    obs, _, done, info = env.step(action)
                    step += 1
                    done = done
                    lidar.append(np.array(obs['lidar']))
                    steering.append(np.array(obs['steering']).reshape(1, 1))
                    velocity.append(np.array(obs['v']).reshape(1, 1))
                    acceleration.append(np.array(obs['a']).reshape(1, 1))
                    actions.append(np.array(action).reshape(1, 2))
        else:
            if step < np.floor(step_limit / 2):
                while done == False:
                    action = [-1, env.action_space.sample()[1]]
                    obs, _, done, info = env.step(action)
                    step += 1
                    done = done
                    lidar.append(np.array(obs['lidar']))
                    steering.append(np.array(obs['steering']).reshape(1, 1))
                    velocity.append(np.array(obs['v']).reshape(1, 1))
                    acceleration.append(np.array(obs['a']).reshape(1, 1))
                    actions.append(np.array(action).reshape(1, 2))
            else:
                while done == False:
                    action = [1, env.action_space.sample()[1]]
                    obs, _, done, info = env.step(action)
                    step += 1
                    done = done
                    lidar.append(np.array(obs['lidar']))
                    steering.append(np.array(obs['steering']).reshape(1, 1))
                    velocity.append(np.array(obs['v']).reshape(1, 1))
                    acceleration.append(np.array(obs['a']).reshape(1, 1))
                    actions.append(np.array(action).reshape(1, 2))
        
        lidar = np.concatenate(lidar, axis=0)
        steering = np.concatenate(steering, axis=0)
        velocity = np.concatenate(velocity, axis=0)
        acceleration = np.concatenate(acceleration, axis=0)
        actions = np.concatenate(actions, axis=0)
        
        print('Appending all observations and actions to files for environment {} and episode {}'.format(i, episode))
        with h5py.File('env_' + str(i) + '_lidar.h5', 'a') as fl:
            fl["lidar"].resize((fl["lidar"].shape[0] + lidar.shape[0]), axis = 0)
            fl["lidar"][-lidar.shape[0]:] = lidar
            
        with h5py.File('env_' + str(i) + '_steering.h5', 'a') as fs:
            fs["steering"].resize((fs["steering"].shape[0] + steering.shape[0]), axis = 0)
            fs["steering"][-steering.shape[0]:] = steering
        
        with h5py.File('env_' + str(i) + '_velocity.h5', 'a') as fv:
            fv["velocity"].resize((fv["velocity"].shape[0] + velocity.shape[0]), axis = 0)
            fv["velocity"][-velocity.shape[0]:] = velocity
        
        with h5py.File('env_' + str(i) + '_acceleration.h5', 'a') as fac:
            fac["acceleration"].resize((fac["acceleration"].shape[0] + acceleration.shape[0]), axis = 0)
            fac["acceleration"][-acceleration.shape[0]:] = acceleration

        with h5py.File('env_' + str(i) + '_actions.h5', 'a') as fa:
            fa["actions"].resize((fa["actions"].shape[0] + actions.shape[0]), axis = 0)
            fa["actions"][-actions.shape[0]:] = actions
    
    return None
                
    
def parallel_rollout(num_env, step_limit, num_episodes, lidar_shape):
# note: maybe run this twice to warmup the system
    ray.init()
    ray.get([rollout.remote(i, step_limit, num_episodes, lidar_shape) for i in range(num_env)])
    ray.shutdown()
    return None


def main():
    start = time.time()
    parallel_rollout(num_env=32, step_limit=1000, num_episodes=1000, lidar_shape=32)
    print('Rollouts have been finished')
    print('Elapsed time in seconds: {}'.format(time.time() - start))
    
if __name__ == "__main__":
    main()