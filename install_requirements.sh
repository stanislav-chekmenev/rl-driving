#!/bin/bash

conda create --name rl_env python=3.6  
conda init bash
conda activate rl_env 
git clone https://github.com/openai/gym.git
cd gym 
pip install -e . 
cd .. 
git clone https://github.com/benelot/pybullet-gym.git 
cd pybullet-gym 
pip install -e .
cd .. 
pip install -r requirements.txt
printf "********************************** \n  The installation is successful\n  Please activate now the environment, using conda activate rl_env \n**********************************\n"




