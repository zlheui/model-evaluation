import tensorflow as tf
import json


def _decode_samples(image_list, shuffle=False):
    decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 1]
        'label_vol': tf.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 1]
    volume_size = [256, 256, 1]
    label_size = [256, 256, 1]

    data_queue = tf.train.string_input_producer(image_list, shuffle=shuffle)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    # data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    label_vol = tf.decode_raw(parser['label_vol'], tf.uint8)
    label_vol = tf.reshape(label_vol, raw_size)
    # label_vol = tf.slice(label_vol, [0, 0, 0], label_size)

    batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)

    return data_vol, batch_y


def _load_samples(source_pth, target_pth):

    with open(source_pth, 'r') as fp:
        rows = fp.readlines()
    imagea_list = [row[:-1] for row in rows]

    with open(target_pth, 'r') as fp:
        rows = fp.readlines()
    imageb_list = [row[:-1] for row in rows]

    data_vola, label_vola = _decode_samples(imagea_list, shuffle=False)
    data_volb, label_volb = _decode_samples(imageb_list, shuffle=False)

    return data_vola, data_volb, label_vola, label_volb


def load_data(source_pth, target_pth, batch_size, do_shuffle=True):

    image_i, image_j, gt_i, gt_j = _load_samples(source_pth, target_pth)

    #default, update
    if 'mr' in source_pth:
        image_i = tf.subtract(tf.multiply(tf.div(tf.subtract(image_i, -2.6), tf.subtract(4.8, -2.6)), 2.0), 1)
    elif 'ct' in source_pth:
        image_i = tf.subtract(tf.multiply(tf.div(tf.subtract(image_i, -2.3), tf.subtract(4.4, -2.3)), 2.0), 1)
        # image_i = tf.reverse(image_i, [0])
        # gt_i = tf.reverse(gt_i, [0])

    if 'ct' in target_pth:
        image_j = tf.subtract(tf.multiply(tf.div(tf.subtract(image_j, -2.3), tf.subtract(4.4, -2.3)), 2.0), 1)
        # image_j = tf.reverse(image_j, [0])
        # gt_j = tf.reverse(gt_j, [0])
    elif 'mr' in target_pth:
        image_j = tf.subtract(tf.multiply(tf.div(tf.subtract(image_j, -2.6), tf.subtract(4.8, -2.6)), 2.0), 1)

    # Batch
    if do_shuffle is True:
        images_i, images_j, gt_i, gt_j = tf.train.shuffle_batch([image_i, image_j, gt_i, gt_j], batch_size, 500, 100)
    else:
        images_i, images_j, gt_i, gt_j = tf.train.batch([image_i, image_j, gt_i, gt_j], batch_size=batch_size, num_threads=1, capacity=500)

    return images_i, images_j, gt_i, gt_j
