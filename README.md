# PPO with World Models for gym-metacar environment

![alt-text](https://github.com/stanislav-chekmenev/rl-driving/blob/master/assets/wm_ppo_policy.gif)

A combination of proximal policy optimization (PPO) and World Models is used to train a self-driving car in gym-metacar environment. More about gym-metacar can be found here: https://github.com/AI-Guru/gym-metacar

# Prerequisites

To install gym-metacar environment run:

```pyhton
pip install git+https://github.com/AI-Guru/gym-metacar
```

You should have chromedriver installed. 

### Mac

```python
brew cask install google-chrome
brew cask install chromedriver
```

### Linux

```python
apt install chromium-chromedriver
apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1
```

# Installation

Firstly, please clone the repo:

```python
git clone https://github.com/stanislav-chekmenev/rl-driving
```

### RLdriving

To install rldriving package:

```python
cd rldriving
pip install -e .
```
### Baselines

You will need baselines for the control part if you want to do training. I modified some source code of the default tensorflow 2.0 implementation of baselines, as well as created my custom neural nets for the control part where I used PPO algorithm. So you should simply do the following:

```python
cd baselines
pip install -e .
```

### Environment specifications

You can skip this step totally. It only describes how to change the resolution of the lidar, so you could use the pretrained models and run the scripts without any changes. The default lidar resolution is 5x5.

I used a lidar resolution of 32x32, so to change the default one you would need to change the specification of the environment. The script with the environment should be in the directory of your environment:

```python
path_to_environment/lib/python3.6/site-packages/gym_metacar/envs
```

Then you should insert these 2 lines to the appropriate positions in the section for observation space.

```python
lidar_space = spaces.Box(low=-1, high=1, shape=(32, 32), dtype=np.float32)
linear_space = spaces.Box(low=-1, high=1, shape=(32 * 32 + 1,), dtype=np.float32)
```

And finally you should add this line to the second if/else clause in the trigger environment initialization section:

```python
script += 'env.setAgentLidar({pts: 32, width: 5, height: 5, pos: -1.5});' + "\n"
```

You can find more info on the pages of the metacar environment: https://github.com/thibo73800/metacar

# Pretrained agent

The package comes with a pretrained agent, a complete World Model where for the control part PPO is used. Also, you can find the first two parts of the World Model, which are a variational autoencoder (VAE) and a LSTM with a Gaussian Mixture (Memory). The models are stored in the models directory for each part.

### Driving

You can use the agent to enjoy driving and, of course, as a first step for sampling from the environment for further retraining.

A neural network for the policy is in drive directory and is contained in the script wm_ppo_policy.py. This is exactly the same net, as I used for training, which can be found in baselines/baselines/common/models. It's called world_model_mlp.  

For driving:

```python 
cd drive
python drive.py
```

Perhaps, you would need to change the path to the agent inside the script. 

Enjoy!



