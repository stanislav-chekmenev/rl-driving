from setuptools import setup, find_packages

setup(
    name='rldriving',
    version='0.0.1',
    author="Dr. Stanislav Chekmenev",
    author_email="stanislav.chekmenev@gmail.com",
    description="World Models and PPO2 for gym_metacar",
    long_description="World Models and PPO2 for gym_metacar",
    long_description_content_type="text/markdown",
    url="https://github.com/stanislav-chekmenev/rl-driving",
    install_requires=['h5py==2.10.0', 'imageio==2.6.1', 'ipython==7.9.0', 'matplotlib==3.1.1', 'mpi4py==3.0.3', 
                      'numpy==1.17.4', 'pillow==6.2.1', 'ray==0.7.6', 'setproctitle==1.1.10', 'tensorflow==2.7.2'],
    packages=find_packages()
)
