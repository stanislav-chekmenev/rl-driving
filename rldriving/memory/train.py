from rldriving.vae.cvae import *
from rldriving.memory.memory import Memory

import time
import h5py

BATCH_SIZE = 16
NUM_BATCHES = int(np.ceil(lidar_train.shape[0] / BATCH_SIZE))
NUM_EPOCHS = 1
SAVE_EVERY = 10

def read_observations(directory: str, envs: list, lidar_res: int):
    
    lidar = np.zeros((0, lidar_res), dtype='float32')
    actions = np.zeros((0, 2), dtype='float32')
    velocity = np.zeros((0, 1), dtype='float32')
    acceleration = np.zeros((0, 1), dtype='float32')
    steering = np.zeros((0, 1), dtype='float32')
    
    for env in envs:
        with h5py.File(os.path.join(directory, 'env_' + str(env) + '_lidar.h5'), 'r') as fl:
            lidar = np.concatenate([lidar, fl['lidar'][:].astype('float32')], axis=0)
            
        with h5py.File(os.path.join(directory, 'env_' + str(env) + '_steering.h5'), 'r') as fs:
            steering = np.concatenate([steering, fs['steering'][:].astype('float32')], axis=0)
            
        with h5py.File(os.path.join(directory, 'env_' + str(env) + '_velocity.h5'), 'r') as fv:
            velocity = np.concatenate([velocity, fv['velocity'][:].astype('float32')], axis=0)
            
        with h5py.File(os.path.join(directory, 'env_' + str(env) + '_acceleration.h5'), 'r') as fa:
            acceleration = np.concatenate([acceleration, fa['acceleration'][:].astype('float32')], axis=0)
            
        with h5py.File(os.path.join(directory, 'env_' + str(env) + '_actions.h5'), 'r') as fac:
            actions = np.concatenate([actions, fac['actions'][:].astype('float32')], axis=0)
        
    
    right_shape = (lidar.shape[0]/lidar_res)
    assert right_shape == steering.shape[0]
    assert right_shape == actions.shape[0]
    assert right_shape == velocity.shape[0]
    assert right_shape == acceleration.shape[0]
    
    return lidar.reshape(-1, 1000, lidar_res, lidar_res, 1), \
            steering.reshape(-1, 1000, 1), \
            velocity.reshape(-1, 1000, 1), \
            acceleration.reshape(-1, 1000, 1), \
            actions.reshape(-1, 1000, 2)

envs_train = [i for i in range(31)]
envs_test = [31]
lidar_train, steering_train, velocity_train, acceleration_train, actions_train = read_observations('samples', envs=envs_train, lidar_res=32)
lidar_test, steering_test, velocity_test, acceleration_test, actions_test = read_observations('samples', envs=envs_test, lidar_res=32)

# Rescale lidar values
lidar_train = (lidar_train + 1.) / 2.
lidar_test = (lidar_test + 1.) / 2.

# Load the VAE
vae_model = CVAE(latent_dim=8)
vae_model.load("../vae/models", 'vae_model_7500K_8')

lidar_train_dataset = tf.data.Dataset.from_tensor_slices(lidar_train).batch(BATCH_SIZE, drop_remainder=True)
lidar_test_dataset = tf.data.Dataset.from_tensor_slices(lidar_test).batch(BATCH_SIZE, drop_remainder=True)

#Create the test set
batch_num = 0
X_test = np.zeros((NUM_BATCHES, BATCH_SIZE, 999, 13)).astype('float32')
Y_test = np.zeros((NUM_BATCHES, BATCH_SIZE, 999, 8)).astype('float32')
for batch in lidar_test_dataset:
        # Split the steering, velocity, etc.
        begin = batch_num * BATCH_SIZE
        end = begin + BATCH_SIZE
        steering_test_batch = steering_test[begin:end]
        velocity_test_batch = velocity_test[begin:end]
        acceleration_test_batch = acceleration_test[begin:end]
        actions_test_batch = actions_test[begin:end]

        assert batch.shape[0] == BATCH_SIZE
        
        mu_test = np.zeros((BATCH_SIZE, 1000, 8)).astype('float32')
        logvar_test = np.zeros((BATCH_SIZE, 1000, 8)).astype('float32')
        
        for idx in range(batch.shape[0]):
            mu_test[idx], logvar_test[idx] = vae_model.encode(batch[idx])
        
        # Sample a latent_space vector
        epsilon_test = tf.random.normal(shape=mu_test.shape)
        z_test = mu_test + epsilon_test * tf.exp(logvar_test * .5)

        # Concatenate other observations and make x, y pairs for training
        x_test = tf.concat((z_test[:, :-1, :], 
                   steering_test_batch[:, :-1, :], 
                   velocity_test_batch[:, :-1, :], 
                   acceleration_test_batch[:, :-1, :],
                   actions_test_batch[:, :-1, :]), axis=2)

        # Create output
        y_test = z_test[:, 1:, :]
        
        assert x_test.shape[0] == y_test.shape[0]
        assert y_test.shape[1] == 999
        assert x_test.shape[2] == 13
        assert y_test.shape[2] == 8
        
        X_test[batch_num] = x_test
        Y_test[batch_num] = y_test
        
        batch_num += 1
        print('Batch_num: {} processed'.format(batch_num - 1))
        
# Create writer for tensorboard
writer = tf.summary.create_file_writer("summaries/")
batch_train_loss = []
batch_test_loss = []
model = Memory(epochs=NUM_EPOCHS, batch_per_epoch=NUM_BATCHES)

# Train
def main():
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        batch_num = 0
        for batch in lidar_train_dataset:
            # Split the steering, velocity, etc. + the test set
            begin = batch_num * BATCH_SIZE
            end = begin + BATCH_SIZE
            steering_train_batch = steering_train[begin:end]
            velocity_train_batch = velocity_train[begin:end]
            acceleration_train_batch = acceleration_train[begin:end]
            actions_train_batch = actions_train[begin:end]

            assert batch.shape[0] == BATCH_SIZE

            mu_train = np.zeros((BATCH_SIZE, 1000, 8)).astype('float32')
            logvar_train = np.zeros((BATCH_SIZE, 1000, 8)).astype('float32')

            for idx in range(batch.shape[0]):
                mu_train[idx], logvar_train[idx] = vae_model.encode(batch[idx])

            # Sample a latent_space vector
            epsilon_train = tf.random.normal(shape=mu_train.shape)
            z_train = mu_train + epsilon_train * tf.exp(logvar_train * .5)

            # Concatenate other observations and make x, y pairs for training
            x_train = tf.concat((z_train[:, :-1, :], 
                       steering_train_batch[:, :-1, :], 
                       velocity_train_batch[:, :-1, :], 
                       acceleration_train_batch[:, :-1, :],
                       actions_train_batch[:, :-1, :]), axis=2)

            # Create output
            y_train = z_train[:, 1:, :]

            # Get x_test and y_test
            x_test = X_test[batch_num]
            y_test = Y_test[batch_num]

            assert x_train.shape[0] == y_train.shape[0]
            assert y_train.shape[1] == 999
            assert x_train.shape[2] == 13
            assert y_train.shape[2] == 8

            assert x_test.shape[0] == y_test.shape[0]
            assert y_test.shape[1] == 999
            assert x_test.shape[2] == 13
            assert y_test.shape[2] == 8

            # Get first hidden state
            train_state = model.lstm.get_zero_hidden_state(x_train)
            test_state = model.lstm.get_zero_hidden_state(x_test)
            # Get train loss
            train_loss = model.train_op(x_train, y_train, train_state)
            # Get test loss
            test_out, _, _ = model.lstm(x_test, test_state)
            test_loss = model.mixture.get_loss(test_out, y_test)
            # Append losses
            batch_train_loss.append(train_loss)
            batch_test_loss.append(test_loss)
            # Write the losses
            batch_num += 1
            with writer.as_default():
                tf.summary.scalar('batch_train_loss', train_loss, step=batch_num-1)
                tf.summary.scalar('batch_test_loss', test_loss, step=batch_num-1)
                writer.flush()

            print('Epoch: {}, batch_num: {}, batch_train_loss: {}, batch_test_loss {}'.format(epoch, batch_num, train_loss.numpy(), test_loss.numpy()))
            if batch_num % SAVE_EVERY == 0:
                model.save(os.getcwd())



        end_time = time.time()
        print('Time elapsed for epoch {}: {}'.format(epoch, end_time - start_time))
        model.save(os.getcwd())

    
if __name__ == "__main__":
    main()