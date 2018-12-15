# ---------------------------------------------------------
# Tensorflow WGAN-GP Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import numpy as np
import scipy.misc
import tensorflow as tf

import utils as utils

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


def _init_logger(flags, log_path):
    if flags.is_train:
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(log_path, 'dataset.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

class Lear(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (self.flags.img_height, self.flags.img_width, 3)
        self.num_trains = 0

        self.lear_path = os.path.join('../Data', self.dataset_name, 'train')
        self._load_lear()

    def _load_lear(self):
        logger.info('Load {} dataset...'.format(self.dataset_name))
        self.train_data = utils.all_files_under(self.lear_path, extension='.jpg')
        self.num_trains = len(self.train_data)

        logger.info('Load {} dataset SUCCESS!'.format(self.dataset_name))
        logger.info('Img size: {}'.format(self.image_size))
        logger.info('Num. of training data: {}'.format(self.num_trains))

    def train_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.train_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, self.flags.img_height, self.flags.img_width, is_gray_scale=False) for batch_path in batch_paths]
        return np.asarray(batch_imgs)


# noinspection PyPep8Naming
def Dataset(sess, flags, dataset_name, log_path=None):
    if flags.is_train:
        _init_logger(flags, log_path)  # init logger

    if dataset_name == 'lear':
        return Lear(flags, dataset_name)
    else:
        raise NotImplementedError
