import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import tensorflow as tf

import command_line
import pipeline
import train
import unet_2nd


def build_discriminator():
    tkl = tf.keras.layers
    l2_reg = 5e-5
    args = {'padding': 'same',
            'kernel_initializer': 'he_uniform',
            'kernel_regularizer': tf.keras.regularizers.l2(l2_reg)}
    model = tf.keras.models.Sequential([
        tkl.Conv2D(64, 3, 2, **args),
        tkl.BatchNormalization(),
        tkl.LeakyReLU(),
        tkl.Conv2D(128, 3, 2, **args),
        tkl.BatchNormalization(),
        tkl.LeakyReLU(),
        tkl.Conv2D(128, 3, 2, **args),
        tkl.GlobalAveragePooling2D(),
        tkl.Dense(1, 'sigmoid', kernel_initializer='uniform', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM))
    return model


if __name__ == '__main__':
    options = command_line.generator_options()
    options.kernel_size = [(k, 3) for k in options.kernel_size]

    if not os.path.exists(os.path.join(options.model_dir, '1', 'saved_model.pb')):
        raise Exception(f'Generator 1 does not exist in the model directory')
    with tf.device('/device:gpu:1'):
        generator_1 = tf.keras.models.load_model(os.path.join(options.model_dir, '1'))
        pipeline = pipeline.StageTwoTemporalInfillPipeline(generator=generator_1, strides=options.strides, unoccluded=10,
                                                           target_size=(None, 96), noise=options.noise)

    # create the model
    with tf.device('/device:gpu:0'):
        tf.summary.trace_on()
        options.model_dir = os.path.join(options.model_dir, '2')
        if os.path.exists(os.path.join(options.model_dir, 'saved_model.pb')):
            print('Loading previously saved model.')
            generator_2 = tf.keras.models.load_model(options.model_dir)
        else:
            print('Creating a new model.')
            generator_2 = unet_2nd.UNet(options=options)
            generator_2.compile(optimizer=generator_2.optimizer, loss='mean_absolute_error')

        discriminator = None
        if options.is_gan:
            model_dir = os.path.join(options.model_dir, 'discriminator')
            discriminator = tf.keras.models.load_model(model_dir) if os.path.exists(model_dir) else build_discriminator()

        # choose the appropriate training routine
        train.train_model(generator_2, discriminator, pipeline, options, tf.losses.Huber(delta=10))
