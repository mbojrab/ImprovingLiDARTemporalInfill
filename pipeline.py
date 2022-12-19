import bz2
import cv2 as cv
import numpy as np
import pathlib
import pickle
import re
import tensorflow as tf

import command_line


class _TemporalInfillPipeline:
    """Class to randomly apply vehicle-like occlusions to a sequence of images"""

    def __init__(self, strides, unoccluded=35, target_shape=None):
        self._unoccluded = unoccluded
        self._max_occlude_length = 30
        self._strides = strides
        self._target_shape = target_shape

    def _intersect(self, a, b):
        return np.intersect1d(a, b, assume_unique=True)

    def _intersects(self, a, b):
        return len(self._intersect(a, b))

    def _sort_npzs(self, directory):
        return sorted([(f, int(f.name.split('_')[0])) for f in directory.glob('*.npz')], key=lambda x: x[1])

    def _get_uuid(self, file):
        return re.search(r'[0-9a-f]{32}', file).group(0)

    def _read_pklgz(self, pklgz):
        with bz2.BZ2File(pklgz.as_posix(), 'r') as zip:
            return pickle.load(zip)

    def _adjust_dims(self, x):
        for stride in self._strides:
            x = int(np.ceil(float(x) / stride))
        for stride in self._strides[::-1]:
            x = int(x * stride)
        return x

    def _subsample_polyline(self, polyline, factor=4):
        subsampled = []
        for ii in range(len(polyline[:-1])):
            pt = polyline[ii]
            pt_ = polyline[ii+1]

            v = pt_[1:] - pt[1:]
            for jj in range(factor):
                subsampled.append(np.concatenate(([pt[0]], pt[1:] + (float(jj) / factor) * v)))
        subsampled.append(polyline[-1])
        return np.vstack(subsampled)

    def _randomize_mask(self, npzs, mode):

        dir = npzs[0][0].parent

        # training mode randomizes the variables each time
        if mode == 'train':
            # read parameters of the directory
            start_uuid = self._get_uuid(npzs[0][0].name)
            end_uuid = self._get_uuid(npzs[-1][0].name)
            start_delins = list(dir.glob(f'*{start_uuid}.pkl.gz'))
            end_delins = list(dir.glob(f'*{end_uuid}.pkl.gz'))

            start_trail_kk = self._read_pklgz(start_delins[0])[0][0]
            end_trail_kk = self._read_pklgz(end_delins[0])[-1][0]

            # randomize the read
            shadow_possible = end_trail_kk - (start_trail_kk + (self._unoccluded * 2))
            random_length = np.random.randint(10, int(np.minimum(self._max_occlude_length, shadow_possible)),
                                              (1,)).astype(np.uint16)[0]
            random_start = np.random.randint(start_trail_kk + self._unoccluded,
                                             end_trail_kk - (self._unoccluded + random_length),
                                             (1,)).astype(np.uint16)[0]  # in meters
            random_radius = np.random.randint(3 * 5, 12 * 5, (1,)).astype(np.float16)[0]  # in pixels
            random_delin_kk = np.random.randint(0, len(start_delins), (1,)).astype(np.uint16)[0]

            occluded_trail_indx = np.arange(random_start, random_start + random_length, dtype=np.uint16)
            full_trail_indx = np.arange(random_start - self._unoccluded,
                                        random_start + random_length + self._unoccluded, dtype=np.uint16)

        else:
            random_radius, random_delin_kk, occluded_trail_indx, full_trail_indx = \
                self._read_pklgz(npzs[0][0].parent.joinpath('random_mask.pkl.gz'))
        return random_radius, random_delin_kk, occluded_trail_indx, full_trail_indx

    def _create(self, data_dir, mode):
        """Create the data pipeline with the provided augmentation pipeline.

        Parameters
        ----------
        data_dir : str
            Parent directory where the .tfrecords are sitting.
        mode : str
            This is which of the files to select.

        Returns
        -------
        dataset : yield object
            The dataset representing the fully configured data pipeline.
        """
        data_dir = pathlib.Path(data_dir, mode)
        subdirs = list(data_dir.iterdir())
        if mode == 'train':
            np.random.shuffle(subdirs)

        for ii, dir in enumerate(subdirs):
            npzs = self._sort_npzs(dir)

            random_radius, random_delin_kk, occluded_trail_indx, full_trail_indx = self._randomize_mask(npzs, mode)

            target_shape = [0, 0]
            images, masks = [], []
            for npz, jj in npzs:

                # test if the shadow is in this image
                uuid = self._get_uuid(npz.name)
                delin = self._read_pklgz(pathlib.Path(dir, f'{jj}_d{random_delin_kk}_{uuid}.pkl.gz'))
                delin_indx = delin[:, 0].astype(np.uint32)
                if self._intersects(delin_indx, full_trail_indx):

                    # read the image and stack it
                    image = np.load(npz.as_posix())
                    image = np.dstack((image['image'], image['depth'])).astype(np.uint8)
                    target_shape = [max(target_shape[0], image.shape[0]), max(target_shape[1], image.shape[1])]
                    images.append(image)

                    # make a custom mask
                    mask = np.ones((image.shape[0], image.shape[1]))
                    if self._intersects(delin_indx, occluded_trail_indx):
                        # NOTE: downsample by 4x to eliminate artifacts from circular masks -- determined empirically
                        for d in self._subsample_polyline(delin, factor=4):

                            # determine if the 'vehicle' occluded this area
                            if self._intersects(d[0].astype(np.uint32), occluded_trail_indx):

                                # apply the mask around each delineator node
                                grid = np.stack(np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])),
                                                axis=-1).astype(np.uint16)
                                dists = np.linalg.norm(d[1:-1] - grid, axis=-1)
                                mask *= np.where(dists < random_radius, 0, 1).astype(np.uint8)
                    masks.append(np.expand_dims(mask, axis=-1))

            # adjust the final target shape based on the model downsamples --
            # this ensures there are not size mismatches due to model operation on this data
            target_shape = [self._adjust_dims(target_shape[0]), self._adjust_dims(target_shape[1])]

            # stack the images together with a common size
            padded_images = np.zeros((len(images), target_shape[0], target_shape[1], 2), dtype=np.uint8)
            padded_masks = np.ones((len(images), target_shape[0], target_shape[1], 1), dtype=bool)
            for jj, (image, mask) in enumerate(zip(images, masks)):
                # center justify the data into the tensor
                start_height = int((target_shape[0] - image.shape[0]) / 2)
                end_height = start_height + image.shape[0]
                start_width = int((target_shape[1] - image.shape[1]) / 2)
                end_width = start_width + image.shape[1]
                slice = np.s_[jj, start_height:end_height, start_width:end_width, :]

                # slice it into the pre-made tensor
                padded_images[slice] = image
                padded_masks[slice]  = mask

            # create a generator to work with tensorflow
            yield padded_images, padded_masks


class StageOneTemporalInfillPipeline(_TemporalInfillPipeline):
    """Class to randomly apply vehicle-like occlusions to a sequence of images"""

    def __init__(self, strides, unoccluded=35, target_size=None):
        _TemporalInfillPipeline.__init__(self, strides, unoccluded, target_size)

    def _process(self, images, masks, mode):
        x = images * masks
        y = images
        resized_masks = masks

        # add data augmentation in training mode
        if mode == command_line.TRAIN:

            # this is equivalent to driving the road backwards
            rand_ud = np.random.randint(0, 2, 1)[0]
            if rand_ud:
                x = np.flipud(x[::-1,:,:,:])
                y = np.flipud(y[::-1,:,:,:])

            # randomly flip left and right
            rand_lr = np.random.randint(0, 2, 1)[0]
            if rand_lr:
                x = np.fliplr(x)
                y = np.fliplr(y)

        # randomly crop the requested support
        if self._target_shape is not None:
            self._target_shape = list(self._target_shape)
            _shape = x.shape[1:3]
            if self._target_shape[0] is None:
                self._target_shape[0] = _shape[0]
            if self._target_shape[1] is None:
                self._target_shape[1] = _shape[1]

            start_row = np.random.randint(0, int(_shape[0] - (self._target_shape[0]-1)), 1)[0]
            mask_center = int(np.mean(np.where(masks == 0)[2]))
            half_width = int(self._target_shape[1] // 2.)
            start_col = int(np.maximum(mask_center - half_width, 0.))
            slice = np.s_[:, start_row : start_row + self._target_shape[0],
                             start_col : start_col + self._target_shape[1], :]
            x = x[slice]
            y = y[slice]
            resized_masks = resized_masks[slice]

        return x, y, resized_masks

    def create(self, data_dir, mode, batch_size=128, num_epochs=1):
        """Create the data pipeline with the provided augmentation pipeline.

        Parameters
        ----------
        data_dir : str
            Parent directory where the .tfrecords are sitting.
        mode : str
            This is which of the files to select.
        batch_size : int
            This is an unused variable.
        num_epochs : int
            This is an unused variable.

        Returns
        -------
        dataset : yield object
            The dataset representing the fully configured data pipeline.
        """
        for images, masks in self._create(data_dir, mode):

            x, y, masks = self._process(images, masks, mode)
            x = tf.cast(x / 255., tf.float32)
            y = tf.cast(y / 255., tf.float32)

            yield tf.convert_to_tensor(x), tf.convert_to_tensor(y)

class StageTwoTemporalInfillPipeline(StageOneTemporalInfillPipeline):
    """Class to randomly apply vehicle-like occlusions to a sequence of images"""

    def __init__(self, generator, strides, unoccluded=35, target_size=None, noise=None):
        StageOneTemporalInfillPipeline.__init__(self, strides, unoccluded, target_size)
        self._generator = generator
        self._noise = noise

        self._target_size = target_size
        self._noise = noise
        self._num_noises = 50
        self._noise_i, self._noise_d = None, None
        self._noise_i_rand = tf.random.shuffle(tf.range(0, self._num_noises, dtype=tf.int32))
        self._noise_d_rand = tf.random.shuffle(tf.range(0, self._num_noises, dtype=tf.int32))
        self._indx = 0

    def create_noise(self, target_shape, noise):
        return tf.random.normal(shape=target_shape, mean=0.0, stddev=noise / 255., dtype=tf.float32)

    def _process(self, images, masks, mode):
        x, y, masks = StageOneTemporalInfillPipeline._process(self, images, masks, mode)

        if self._noise_i is None:
            shape = [self._num_noises] + list(tf.shape(x)[1:-1])
            self._noise_i = self.create_noise(target_shape=shape, noise=self._noise[0])
            self._noise_d = self.create_noise(target_shape=shape, noise=self._noise[1])

        x_ = self._generator(tf.cast(x / 255., tf.float32))
        noise = []
        for x_i in range(tf.shape(x_)[0]):
            noise.append(tf.stack([self._noise_i[self._noise_i_rand[self._indx % self._num_noises]],
                                   self._noise_d[self._noise_d_rand[self._indx % self._num_noises]]], axis=-1))
            self._indx += 1
        return x_ + tf.stack(noise, axis=0), y

    def create(self, data_dir, mode, batch_size=128, num_epochs=1):
        """Create the data pipeline with the provided augmentation pipeline.

        Parameters
        ----------
        data_dir : str
            Parent directory where the .tfrecords are sitting.
        mode : str
            This is which of the files to select.
        batch_size : int
            This is an unused variable.
        num_epochs : int
            This is an unused variable.

        Returns
        -------
        dataset : yield object
            The dataset representing the fully configured data pipeline.
        """
        for images, masks in self._create(data_dir, mode):

            x, y = self._process(images, masks, mode)
            y = tf.cast(y / 255., tf.float32)

            yield tf.convert_to_tensor(x), tf.convert_to_tensor(y)
