"""Code for training SIFA."""
from datetime import datetime
import json
import numpy as np
import random
import os
import cv2
import time

import tensorflow as tf
from PIL import Image


import data_loader_organ_eval as data_loader
import model
from stats_func import *
import math
import medpy.metric.binary as mmb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

save_interval = 300
evaluation_interval = 10
random_seed = 1234

class SIFA:
    """The SIFA module."""
    def __init__(self, config, ratio):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._source_train_pth = config['source_train_pth']
        self._target_train_pth = config['target_train_pth']
        self._source_val_pth = config['source_val_pth']
        self._target_val_pth = config['target_val_pth']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, 'cross-modality-multi-organ-training-1.0-0')
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._pool_size = int(config['pool_size'])
        self._lambda_a = float(config['_LAMBDA_A'])
        self._lambda_b = float(config['_LAMBDA_B'])
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])
        self._base_lr = float(config['base_lr'])
        self._max_step = int(config['max_step'])
        self._keep_rate_value = float(config['keep_rate_value'])
        self._is_training_value = bool(config['is_training_value'])
        self._batch_size = int(config['batch_size'])
        self._lr_gan_decay = bool(config['lr_gan_decay'])
        self._to_restore = bool(config['to_restore'])
        self._checkpoint_dir = config['checkpoint_dir']
        self.ratio = ratio

    def model_setup(self):

        self.input_a = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="input_B")
        self.gt_a = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                self._num_cls
            ], name="gt_A")
        self.gt_b = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                self._num_cls
            ], name="gt_B") # for validation only, not used during training


        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.input_a,
            'fake_pool_b': self.input_b,
        }

        outputs = model.get_outputs(inputs, skip=self._skip, is_training=False, keep_rate=1.0)

        self.pred_mask_fake_a = outputs['pred_mask_fake_a']
        self.pred_mask_b = outputs['pred_mask_b']

        self.predicter_b = tf.nn.softmax(self.pred_mask_b)
        self.predicter_fake_a = tf.nn.softmax(self.pred_mask_fake_a)

        self.final_predict = self.ratio*self.predicter_b + (1-self.ratio)*self.predicter_fake_a
        self.compact_pred_b = tf.argmax(self.final_predict, 3)

        self.compact_y_b = tf.argmax(self.gt_b, 3)


    def test(self):
        # Load Dataset
        self.inputs_val = data_loader.load_data(self._source_val_pth, self._target_val_pth, self._batch_size, False)
        with open(self._target_val_pth, 'r') as f:
            filenames = f.readlines()

        # Build the network
        self.model_setup()
        init = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            saver = tf.train.Saver()

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                # chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, self._checkpoint_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            cnt = -1
            dice_list = []
            assd_list = []

            iterations = [25, 23, 20, 30]
            
            for pid in iterations:
                tmp_pred = np.zeros((pid,256,256))
                tmp_gt = np.zeros((pid,256,256))
                for i in range(pid):

                    images_i_val, images_j_val, gts_i_val, gts_j_val = sess.run(self.inputs_val)
                    
                    compact_pred_a, compact_y_a = sess.run([self.compact_pred_b, self.compact_y_b], feed_dict={self.input_b: images_j_val, self.gt_b: gts_j_val})
                    
                    tmp_pred[i] = compact_pred_a.copy()
                    tmp_gt[i] = compact_y_a.copy()

                    cnt += 1

                for c in range(1, self._num_cls):
                    pred_test_data_tr = tmp_pred.copy()
                    pred_test_data_tr[pred_test_data_tr != c] = 0

                    pred_gt_data_tr = tmp_gt.copy()
                    pred_gt_data_tr[pred_gt_data_tr != c] = 0

                    dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                    assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))

            coord.request_stop()
            coord.join(threads)

            dice_arr = 100 * np.reshape(dice_list, [4, -1]).transpose()

            dice_mean = np.mean(dice_arr, axis=1)
            dice_std = np.std(dice_arr, axis=1)

            print('Dice:')
            print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
            print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
            print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
            print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
            print('Mean:%.1f' % np.mean(dice_mean))

            assd_arr = np.reshape(assd_list, [4, -1]).transpose()

            assd_mean = np.mean(assd_arr, axis=1)
            assd_std = np.std(assd_arr, axis=1)

            print('ASSD:')
            print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
            print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
            print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
            print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
            print('Mean:%.1f' % np.mean(assd_mean))

            print('%.1f' % dice_mean[3]+'\n'+'%.1f' % dice_mean[1]+'\n'+'%.1f' % dice_mean[2]+'\n'+'%.1f' % dice_mean[0]+'\n'+'%.1f' % np.mean(dice_mean)+'\n'+'%.1f' % assd_mean[3]+'\n'+'%.1f' % assd_mean[1]+'\n'+'%.1f' % assd_mean[2]+'\n'+'%.1f' % assd_mean[0]+'\n'+'%.1f' % np.mean(assd_mean)+'\n')


def main(config_filename):

    # tf.set_random_seed(random_seed)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    for i in [5]:
        sifa_model = SIFA(config, i/10.)
        sifa_model.test()

if __name__ == '__main__':
    main(config_filename='./config_param_organ_ct_to_mri.json')

