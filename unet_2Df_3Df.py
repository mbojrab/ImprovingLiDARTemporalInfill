import datetime
import numpy as np
import tensorflow as tf
import uuid


def select_channels(x, channels):
    if len(channels) != x.shape[-1] or np.any(np.array(channels) != np.arange(x.shape[-1])):
        _c = tf.unstack(x, axis=-1)
        return tf.stack([_c[c] for c in channels], axis=-1)
    else:
        return x

class _SoftPlus(tf.keras.layers.Layer):
    def __init__(self, beta=10.):
        tf.keras.layers.Layer.__init__(self, name='softmax')
        self._beta = beta
    def call(self, x):
        return tf.math.log(tf.ones_like(x) + tf.math.exp(x))


class _LayerSequence(tf.keras.layers.Layer):
    def __init__(self, name):
        tf.keras.layers.Layer.__init__(self, name=name)
        self._seq = []

    def call(self, features):
        out = features
        for layer in self._seq:
            out = layer(out)
        return out

def resize(kernel_size, strides):
    if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        kernel_size = [3] + list(kernel_size)
    if isinstance(kernel_size, int):
        kernel_size = [3, kernel_size, kernel_size]
    if isinstance(strides, int):
        strides = [1, strides, strides]
    return kernel_size, strides

class _GatedConv3D(_LayerSequence):

    def __init__(self, kernels, kernel_size, strides, l2_reg, activation=None, name=''):
        _LayerSequence.__init__(self, name=name)
        kernel_size, strides = resize(kernel_size, strides)
        self._seq.extend([tf.keras.layers.Conv3D(filters=kernels, kernel_size=kernel_size, padding='same',
                                                 strides=strides, kernel_initializer='he_uniform',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                 activation=activation),
                          tf.keras.layers.Conv3D(filters=kernels, kernel_size=kernel_size, padding='same',
                                                 strides=strides, kernel_initializer='uniform',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                 activation=tf.nn.sigmoid)])

    def call(self, features):
        return self._seq[0](features) * self._seq[1](features)


class _GatedConv2D(_LayerSequence):

    def __init__(self, kernels, kernel_size, strides, l2_reg, activation=None, name=''):
        _LayerSequence.__init__(self, name=name)
        self._seq.extend([tf.keras.layers.Conv2D(filters=kernels, kernel_size=kernel_size, padding='same',
                                                 strides=strides, kernel_initializer='he_uniform',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                 activation=activation),
                          tf.keras.layers.Conv2D(filters=kernels, kernel_size=kernel_size, padding='same',
                                                 strides=strides, kernel_initializer='uniform',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                 activation=tf.nn.sigmoid)])

    def call(self, features):
        return self._seq[0](features) * self._seq[1](features)


class _BlockSequence(_LayerSequence):

    def __init__(self, kernels, kernel_size, strides=1, l2_reg=0.0, activation=None, name=''):
        _LayerSequence.__init__(self, name=name)

        # build the convolution block --
        # NOTE: Deconvolution is performed in a separate convolutional layer in order to remedy the artifacting
        #       resulting from upsampled convolutions. Performing a second convolution on its heals, ensures the
        #       artifacts are learned and undone. This is a common approach to this documented issue.
        kernel_size_3d, strides_3d = resize(kernel_size, strides)
        self._seq.extend([tf.keras.layers.Conv3DTranspose(filters=kernels, kernel_size=kernel_size_3d,
                                                          padding='same', strides=strides_3d,
                                                          kernel_initializer='he_uniform',
                                                          bias_initializer='he_uniform',
                                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                          bias_regularizer=tf.keras.regularizers.l2(l2_reg))
                          if strides != 1 else
                          _GatedConv3D(kernels, kernel_size, strides, l2_reg, activation)] +
                          [tf.keras.layers.BatchNormalization(),
                           activation,
                           _GatedConv3D(kernels, 5, 1, l2_reg, None),
                           tf.keras.layers.BatchNormalization(),
                           activation,
                           _GatedConv3D(kernels, 5, 1, l2_reg, None),
                           tf.keras.layers.BatchNormalization(),
                           activation])


class _FinalBlock(_LayerSequence):

    def __init__(self, kernels, kernel_size, strides=1, l2_reg=0.0, name=''):
        _LayerSequence.__init__(self, name=name)

        kernel_size_3d, strides_3d = resize(kernel_size, strides)
        self._seq.extend([tf.keras.layers.Conv3DTranspose(filters=kernels, kernel_size=kernel_size_3d,
                                                          padding='same', strides=strides_3d,
                                                          kernel_initializer='he_uniform',
                                                          bias_initializer='he_uniform',
                                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                          activation=tf.keras.layers.LeakyReLU()),
                          _GatedConv3D(kernels, 5, 1, l2_reg, activation=tf.keras.layers.LeakyReLU())])


class _Encoder(_LayerSequence):

    NAME = 'Encoder'

    def __init__(self, channels, kernels, kernel_size, strides=None, l2_reg=0.0, name=None):
        _LayerSequence.__init__(self, name=self.NAME if name is None else name)

        self._channels = channels
        assert len(kernels) == len(kernel_size), 'The kernel and filter sizes must match to be valid'
        if strides is not None:
            assert len(kernels) == len(strides), 'The length of strides does not match. This is invalid'
        else:
            strides = [(1, 1) for _ in kernels]
        self._seq.extend([_GatedConv2D(k, ks, s, l2_reg, tf.keras.layers.LeakyReLU())
                          for ii, (k, ks, s) in enumerate(zip(kernels, kernel_size, strides))])

    def call(self, features):
        self._residuals = [tf.expand_dims(select_channels(features, self._channels), axis=0)]
        for layer in self._seq:
            self._residuals.append(layer(self._residuals[-1]))
        return self._residuals[::-1]


class _Decoder(_LayerSequence):

    NAME = 'Decoder'

    def __init__(self, num_channels, kernels, kernel_size, strides=None, l2_reg=0.0):
        _LayerSequence.__init__(self, name=self.NAME)
        assert len(kernels) == len(kernel_size), 'The kernel and filter sizes must match to be valid'
        if strides is not None:
            assert len(kernels) == len(strides), 'The length of strides does not match. This is invalid'
        else:
            strides = [(1, 1) for _ in kernels]
        self._seq.extend([_BlockSequence(k, ks, s, l2_reg, tf.keras.layers.LeakyReLU(), f'Block_{ii}')
                          for ii, (k, ks, s) in enumerate(zip(kernels[1:], kernel_size[:-1], strides[:-1]))])

        # add matching decode layer
        self._seq.append(_FinalBlock(kernels=kernels[-1], kernel_size=kernel_size[-1], strides=strides[-1],
                                     l2_reg=l2_reg))

        # add the pointwise squashing layer
        self.squash_layer = _GatedConv3D(num_channels, 1, 1, l2_reg, _SoftPlus())

    def call(self, features):
        def squasher(o):
            #return tf.clip_by_value(o, 0., 1.)
            model_input = tf.squeeze(features[-1], axis=0)
            return tf.clip_by_value(tf.where(model_input == 0, o, model_input), 0., 1.)

        out = features[0]
        for layer, residual in zip(self._seq, features[1:]):
            out = tf.concat((layer(out), residual), axis=-1)

        return squasher(tf.squeeze(self.squash_layer(out), axis=0))


class UNet(tf.keras.Model, tf.Module):
    NAME = 'UNET_GENERATOR'

    def __init__(self, options):
        tf.keras.Model.__init__(self, name=self.NAME)

        self.optimizer = tf.optimizers.RMSprop(options.base_lr, momentum=options.momentum)
        self._seq = []
        self.global_step = tf.Variable(name='global_step', shape=[], dtype=tf.int64, trainable=False, initial_value=0)
        self.uuid = tf.Variable(name='uuid', shape=[], dtype=tf.string, trainable=False,
                                initial_value=uuid.uuid4().hex)
        self.created_at = tf.Variable(name='created_at', shape=[], dtype=tf.string, trainable=False,
                                      initial_value=str(datetime.datetime.utcnow()))

        self.num_channels = tf.constant(name='num_channels', shape=[], dtype=tf.uint32, value=len(options.out_channels))
        self._encoders = [_Encoder(channels=[channel],
                                   kernels=options.kernels,
                                   kernel_size=options.kernel_size,
                                   strides=options.strides,
                                   l2_reg=options.l2_reg) for channel in options.in_channels]
        self._decoder = _Decoder(num_channels=len(options.out_channels),
                                 kernels=options.kernels[::-1],
                                 kernel_size=options.kernel_size[::-1],
                                 strides=options.strides[::-1],
                                 l2_reg=options.l2_reg)
        self._seq.extend(self._encoders + [self._decoder])
        self._set_inputs(tf.keras.Input(shape=(None, None, len(options.in_channels)),
                                        batch_size=options.batch_size,
                                        dtype=tf.float32))
        self.build(input_shape=(None, None, None, len(options.in_channels)))
        self.summary()

    def call(self, features):
        features = tf.cast(features, tf.float32)

        # subsequent channels are added to one another before proceeding --
        # this diverges from some papers where some channels are added and some are concatenated
        encoder_out = [encoder(features) for encoder in self._encoders.layers]
        out = encoder_out[0]
        for x2_out in encoder_out[1:]:
            out = [tf.concat((x1, x2), axis=-1) for x1, x2 in zip(out, x2_out)]
        return self._decoder(out)
