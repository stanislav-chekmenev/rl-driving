from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import imageio
import h5py

from IPython import display

from rldriving.vae.cvae import *

# Function for loading samples
def read_observations(directory: str, envs: list, num_episodes: int, episode_len: int, lidar_res: int):
    
    data = np.zeros((0, lidar_res), dtype='float32')
    for env in envs:
        with h5py.File(os.path.join(directory, 'env_' + str(env) + '_observations_50.h5'), 'r') as f:
            # Throw away the first observation in each file, first 32 rows,
            # since it was used there to creat the files in the beginning
            data = np.concatenate([data, f['lidar'][lidar_res:].astype('float32')], axis=0)
        with h5py.File(os.path.join(directory, 'env_' + str(env) + '_observations_200.h5'), 'r') as f:
            data = np.concatenate([data, f['lidar'][lidar_res:].astype('float32')], axis=0)
    assert data.shape[0] == len(envs) * num_episodes * episode_len * lidar_res
    return data.reshape(-1, lidar_res, lidar_res, 1) 

# Load samples
envs_train = [i for i in range(31)]
envs_test = [32]
train_samples = read_observations('../samples', envs=envs_train, num_episodes=250, episode_len=1000, lidar_res=32)
test_samples = read_observations('../samples', envs=envs_test, num_episodes=250, episode_len=1000, lidar_res=32)


# Rescale lidar values
train_samples = (train_samples + 1.) / 2.
test_samples = (test_samples + 1.) / 2.

# Specify dataset sizes & batch size
TRAIN_BUF = train_samples.shape[0]
BATCH_SIZE = 256

TEST_BUF = 10000

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_samples).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_samples[:100000]).shuffle(TEST_BUF).batch(BATCH_SIZE)
del train_samples, test_samples

# Create writer for tensorboard
writer = tf.summary.create_file_writer("summaries/")
    
# Training parameters
epochs = 300
latent_dim = 32
num_examples_to_generate = 16
# Create model
model = CVAE(latent_dim)

# Optimizer and losses
optimizer = tf.keras.optimizers.Adam(1e-4)
kl_tolerance = 0.5

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def get_loss(model, batch, epoch, name):
        means, logvars = model.encode(batch)
        latent = model.reparameterize(means, logvars)
        generated = model.decode(latent)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(batch - generated), axis=[1, 2, 3])
        )

        kl_loss = - 0.5 * tf.reduce_sum(
            1 + logvars - tf.square(means) - tf.exp(logvars),
            axis=1
        )

        kl_loss = tf.reduce_mean(
            tf.maximum(kl_loss, kl_tolerance * latent_dim)
        )
        with writer.as_default():
            tf.summary.scalar(name + '_reconstruction_loss', reconstruction_loss, step=epoch)
            tf.summary.scalar(name + '_kl_loss', kl_loss, step=epoch)
            writer.flush()
        return {
            'reconstruction-loss': reconstruction_loss,
            'kl-loss': kl_loss
        }
    
@tf.function
def backward(model, batch, optimizer, epoch):
    """ images to loss to new weights"""
    with tf.GradientTape() as tape:
        losses = get_loss(model, batch, epoch=epoch, name='train')
        gradients = tape.gradient(sum(losses.values()), model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return losses

# Random sample for image generation
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])

# Image generation
def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(16,16))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))

# Training
best = -1e20
wait = 0
patience = 10
best_weights = model.get_weights()

generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        backward(model, train_x, optimizer, epoch=epoch)

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        losses = get_loss(model, test_x, epoch=epoch, name='test')
        loss(sum(losses.values()))
    elbo = -loss.result()
           
    if np.greater(elbo, best):
        best = elbo
        wait = 0
        best_weights = model.get_weights()
    else:
        wait += 1
        if wait >= patience:
            model.stop_training = True
            print('Restoring model weights from the end of the best epoch.')
            model.set_weights(best_weights)
            print('Saving model weights')
            model.save('models')
            
    end_time = time.time()
    
    generate_and_save_images(model, epoch, random_vector_for_generation)
    
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))

print('Saving model weights')
model.save('models')