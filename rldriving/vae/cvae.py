import tensorflow as tf
import os


# Create VAE class
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer(input_shape=(32, 32, 1)),
              tf.keras.layers.Conv2D(
                  filters=32, kernel_size=3, strides=(2, 2), activation='relu', name='conv1I'),
              tf.keras.layers.Conv2D(
                  filters=64, kernel_size=3, strides=(2, 2), activation='relu', name='conv2I'),
              tf.keras.layers.Conv2D(
                  filters=128, kernel_size=3, strides=(2, 2), activation='relu', name='conv3I'),
              tf.keras.layers.Flatten(),
              # No activation
              tf.keras.layers.Dense(latent_dim + latent_dim),
          ]
        )

        self.generative_net = tf.keras.Sequential(
            [
              tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
              tf.keras.layers.Dense(units=3*3*128, activation=tf.nn.relu, name='dense1G'),
              tf.keras.layers.Reshape(target_shape=(1, 1, 3*3*128), name='reshape1G'),
              tf.keras.layers.Conv2DTranspose(
                  filters=128,
                  kernel_size=3,
                  strides=(2, 2),
                  activation='relu',
                  name='conv1G'),
              tf.keras.layers.Conv2DTranspose(
                  filters=64,
                  kernel_size=3,
                  strides=(2, 2),
                  activation='relu',
                  name='conv2G'),
              tf.keras.layers.Conv2DTranspose(
                  filters=32,
                  kernel_size=3,
                  strides=(2, 2),
                  activation='relu',
                  name='conv3G'),
              # No activation
              tf.keras.layers.Conv2DTranspose(
                  filters=1, kernel_size=4, strides=(2, 2), padding="VALID", name='conv4G'),
            ]
        )
      
    def save(self, filepath):
        """ only model weights """
        filepath = os.path.join(filepath)
        os.makedirs(filepath, exist_ok=True)
        print('saving model to {}'.format(filepath))
        self.save_weights('{}/vae_model_7500K.h5'.format(filepath))
        
    def load(self, filepath):
        """ only model weights """
        filepath = os.path.join(filepath)
        print('loading model from {}'.format(filepath))
        self.load_weights('{}/vae_model_7500K.h5'.format(filepath))

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits