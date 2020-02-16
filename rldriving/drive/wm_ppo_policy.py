import tensorflow as tf
import os
import numpy as np

class WMPPODriver(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        
        initializer = tf.keras.initializers.Orthogonal(np.sqrt(2))
        config = initializer.get_config()
        initializer = tf.keras.initializers.Orthogonal.from_config(config)
        self.mlp = tf.keras.Sequential(
            [   tf.keras.layers.InputLayer(input_shape=(input_shape, )),
                tf.keras.layers.Dense(units=16, kernel_initializer=initializer, name='d1', activation='tanh'),
                tf.keras.layers.Dense(units=32, kernel_initializer=initializer, name='d2', activation='tanh'),
                tf.keras.layers.Dense(units=2, kernel_initializer=initializer, name='d3', activation=None)
            ]
        )
            
    def __call__(self, x):
        policy = self.mlp(x)
        return policy.numpy()
        
    def load(self, filepath, name):
        """ only model weights """
        filepath = os.path.join(filepath)
        print('loading model from {}'.format(filepath))
        self.load_weights('{}/{}.h5'.format(filepath, name))
        