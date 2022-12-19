import argparse
import os
import tensorflow as tf

import unet_2Df_2Df
import unet_2Df_3Df
import unet_2Df_3Dp
import unet_3Df_2Df
import unet_3Df_3Df
import unet_3Df_3Dp

TRAIN = 'train'             # intended for training a model
VALIDATION = 'validation'   # intended for validating the quality of a model
TEST = 'test'               # (optional) intended for testing model parameters to avoid overfitting to training
MASKS = 'masks'             # (optional) Intended for (ED) generator networks to apply to the input


def generator_options(parser=None, args=None):
    """Setup and parse the command line arguments needed to create/train/resume a generator

    Parameters
    ----------
    parser : argparse.ArgumentParser (optional)
        Add additional parameters to an existing parser.
    args : list of str (optional)
        The argument list to be parsed. If None, sys.argv will be used.

    Returns
    -------
    options : argparse.Options
        The parsed options into a dictionary
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--noise', type=float, nargs='+', default=[30., 5.],
                        help='The noise supplied per channel to the second network. Values in [0, 255]')
    parser.add_argument('--in-channels', type=int, nargs='+', default=[0,1],
                        help='Channel selection and ordering of the input channels.')
    parser.add_argument('--out-channels', type=int, nargs='+', default=[0,1],
                        help='Channel selection and ordering of the output channels.')
    parser.add_argument('--encode-type', type=str, default='3Df', choices=['2Df', '3Df'],
                        help='Specify the type of encoder convolutions.')
    parser.add_argument('--decode-type', type=str, default='2Df', choices=['2Df', '3Df', '3Dp'],
                        help='Specify the type of decoder convolutions.')
    parser.add_argument('--kernels', type=int, nargs='+', help='Number in each layer')
    parser.add_argument('--kernel-size', type=int, nargs='+', help='Size of the kernels in each layer')
    parser.add_argument('--strides', type=int, nargs='+', help='Stride factor in each layer. Works similar to subsampling.')
    parser.add_argument('--depth-multiplier', dest='depth_multiplier', type=int,
                        help='If specified, separable convolutions are used with a per-channel upsample equivalent to '
                             'this value. Separable convolutions make the network computation more efficient at the '
                             'risk of reducing the network capacity. If None, regular 2D convolution is used.')
    parser.add_argument('--l2-regularization', dest='l2_reg', type=float, default=1e-5,
                        help='Specify the L2 regularization strength for the weights.')
    parser.add_argument('--random-mask', dest='random_mask', action='store_true', default=False,
                        help='Turn on binomial mask generation (Not based on sensor pattern).')
    parser.add_argument('--batch', dest='batch_size', type=int, default=128,
                        help='Specify the batch size to use during training.')
    parser.add_argument('--model-dir', dest='model_dir', type=str, default='./model-output',
                        help='Specify the directory where we output the model for this run.')
    parser.add_argument('--num-checkpoints', dest='checkpoints', type=int, default=2,
                        help='Specify the number of checkpoints to keep on disk.')
    parser.add_argument('--epochs', dest='epochs', type=float, default=1,
                        help='Specify the number of epochs to complete between checkpoints. '
                             'Fractional numbers provide multiple evaluations per epoch.')
    parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=60,
                        help='Specify the total of epochs to complete before stopping training. '
                             'The --patience option can early-stop this value.')
    parser.add_argument('--patience', dest='patience', type=int, default=20,
                        help='Specify the number of iterations to perform poorly before early stoppage.')
    parser.add_argument('--learning-rate', dest='base_lr', type=float, default=0.1,
                        help=f'Specify the base learning rate for training.')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help=f'Specify the amount of the last gradient update to add to this round. 0.0 is no momentum '
                             f'and 1.0 adds the gradient from last round.')
    parser.add_argument('--train-gan', dest='is_gan', action='store_true', default=False,
                        help='Turn on Generative Adversarial Training.')
    parser.add_argument('directory', type=str, help="The directory with the tfrecords for train/validation.")

    # parse the options
    options = parser.parse_args(args)

    # tensorflow operates on batches instead of epochs, so convert our options to batches
    options.num_train_examples = len(os.listdir(os.path.join(options.directory, TRAIN)))
    options.num_val_examples = len(os.listdir(os.path.join(options.directory, VALIDATION)))
    options.batch_size = options.batch_size if options.batch_size < options.num_train_examples else \
                         options.num_train_examples
    options.train_steps = int(options.num_train_examples / float(options.batch_size) * options.epochs)

    return options

def unet_factory(options):
    unet_class = None
    if options.encode_type == '3Df':
        if options.decode_type == '2Df':
            unet_class = unet_3Df_2Df.UNet
        elif options.decode_type == '3Df':
            unet_class = unet_3Df_3Df.UNet
        elif options.decode_type == '3Dp':
            unet_class = unet_3Df_3Dp.UNet
    elif options.encode_type == '2Df':
        if options.decode_type == '2Df':
            unet_class = unet_2Df_2Df.UNet
        elif options.decode_type == '3Df':
            unet_class = unet_2Df_3Df.UNet
        elif options.decode_type == '3Dp':
            unet_class = unet_2Df_3Dp.UNet
    return unet_class