import tensorflow as tf
import layers
import json


BATCH_SIZE = 6

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

ngf = 32
ndf = 64


def get_outputs(inputs, skip=False, is_training=True, keep_rate=0.75):
    images_a = inputs['images_a']
    images_b = inputs['images_b']
    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']
    

    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:

        current_discriminator = discriminator
        current_encoder = build_encoder_update
        current_segmenter = build_segmenter_update

        prob_real_a_is_real = current_discriminator(images_a, "d_A")
        prob_real_b_is_real = current_discriminator(images_b, "d_B")

        fake_images_b = build_generator_resnet_9blocks(images_a, images_a, name='g_A', skip=skip)
        fake_images_a = build_generator_resnet_9blocks(images_b, images_b, name='g_B', skip=skip)

        latent_a = current_encoder(images_a, name='e_A', skip=skip, bn_scope='source_batch_norm', is_training=is_training, keep_rate=keep_rate)
        pred_mask_a = current_segmenter(latent_a, name='s_A', is_training=is_training, keep_rate=keep_rate)

        latent_b = current_encoder(images_b, name='e_B', skip=skip, bn_scope='target_batch_norm', is_training=is_training, keep_rate=keep_rate)
        pred_mask_b = current_segmenter(latent_b, name='s_B', is_training=is_training, keep_rate=keep_rate)

        prob_fake_a_is_real = current_discriminator(fake_images_a, "d_A")
        prob_fake_b_is_real = current_discriminator(fake_images_b, "d_B")

        cycle_images_b = build_generator_resnet_9blocks(fake_images_a, fake_images_a, 'g_A', skip=skip)
        cycle_images_a = build_generator_resnet_9blocks(fake_images_b, fake_images_b, 'g_B', skip=skip)

        latent_fake_a = current_encoder(fake_images_a, 'e_A', skip=skip, bn_scope='target_batch_norm', is_training=is_training, keep_rate=keep_rate)
        pred_mask_fake_a = current_segmenter(latent_fake_a, 's_A', is_training=is_training, keep_rate=keep_rate)

        latent_fake_b = current_encoder(fake_images_b, 'e_B', skip=skip, bn_scope='source_batch_norm', is_training=is_training, keep_rate=keep_rate)
        pred_mask_fake_b = current_segmenter(latent_fake_b, 's_B', is_training=is_training, keep_rate=keep_rate)
        
        prob_fake_pool_a_is_real = current_discriminator(fake_pool_a, "d_A")
        prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, "d_B")
        
    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
        'pred_mask_a': pred_mask_a,
        'pred_mask_b': pred_mask_b,
        'pred_mask_fake_a': pred_mask_fake_a,
        'pred_mask_fake_b': pred_mask_fake_b,
    }


def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT", norm_type=None, bn_scope='batch_norm', is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim, 3, 3, 1, 1, 0.01, "VALID", "c1", norm_type=norm_type, bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim, 3, 3, 1, 1, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        return tf.nn.relu(out_res + inputres)


def build_resnet_block_ins(inputres, dim, name="resnet", padding="REFLECT"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d_ga(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1", norm_type='Ins')
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d_ga(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False, norm_type='Ins')

        return tf.nn.relu(out_res + inputres)


def build_resnet_block_ds(inputres, dim_in, dim_out, name="resnet", padding="REFLECT", norm_type=None, bn_scope='batch_norm', is_training=True, keep_rate=0.75):

    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim_out, 3, 3, 1, 1, 0.01, "VALID", "c1", norm_type=norm_type, bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim_out, 3, 3, 1, 1, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        inputres = tf.pad(inputres, [[0, 0], [0, 0], [0, 0], [(dim_out - dim_in) // 2, (dim_out - dim_in) // 2]], padding)

        return tf.nn.relu(out_res + inputres)


def build_drn_block(inputdrn, dim, name="drn", padding="REFLECT", norm_type=None, bn_scope='batch_norm', is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_drn = tf.pad(inputdrn, [[0, 0], [2, 2], [2, 2], [0, 0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim, dim, 3, 3, 2, 0.01, "VALID", "c1", norm_type=norm_type, bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out_drn = tf.pad(out_drn, [[0, 0], [2, 2], [2, 2], [0, 0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim, dim, 3, 3, 2, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        return tf.nn.relu(out_drn + inputdrn)


def build_drn_block_ds(inputdrn, dim_in, dim_out, name='drn_ds', padding="REFLECT", norm_type=None, bn_scope='batch_norm', is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_drn = tf.pad(inputdrn, [[0,0], [2,2], [2,2], [0,0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim_in, dim_out, 3, 3, 2, 0.01, 'VALID', "c1", norm_type=norm_type, bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out_drn = tf.pad(out_drn, [[0,0], [2,2], [2,2], [0,0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim_out, dim_out, 3, 3, 2, 0.01, 'VALID', "c2", do_relu=False, norm_type=norm_type, bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        inputdrn = tf.pad(inputdrn, [[0,0], [0,0], [0, 0], [(dim_out-dim_in)//2,(dim_out-dim_in)//2]], padding)

        return tf.nn.relu(out_drn + inputdrn)


def build_generator_resnet_9blocks(inputgen, inputimg, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d_ga(pad_input, ngf, f, f, 1, 1, 0.02, name="c1", norm_type='Ins')
        o_c2 = layers.general_conv2d_ga(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2", norm_type='Ins')
        o_c3 = layers.general_conv2d_ga(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3", norm_type='Ins')

        o_r1 = build_resnet_block_ins(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block_ins(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block_ins(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block_ins(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block_ins(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block_ins(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block_ins(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block_ins(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block_ins(o_r8, ngf * 4, "r9", padding)

        o_c4 = layers.general_deconv2d(o_r9, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4", norm_type='Ins')
        o_c5 = layers.general_deconv2d(o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5", norm_type='Ins')
        o_c6 = layers.general_conv2d_ga(o_c5, 1, f, f, 1, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputimg + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def _phase_shift(I, r, batch_size):
    # Helper function with main phase shift operation

    _, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (batch_size, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    if batch_size == 1:
        X = tf.expand_dims( X, 0 )
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    if batch_size == 1:
        X = tf.concat([x for x in X], 2 )
    else:
        X = tf.concat([tf.squeeze(x) for x in X], 2)  #
    out =  tf.reshape(X, (batch_size, a*r, b*r, 1))
    if batch_size == 1:
        out = tf.transpose( out, (0,2,1,3)  )
    return out


def PS(X, r, batch_size, n_channel = 8):
  # Main OP that you can arbitrarily use in you tensorflow code
    Xc = tf.split(X, n_channel, -1 )
    X = tf.concat([_phase_shift(x, r, batch_size) for x in Xc], 3)
    return X


def build_encoder_update(inputen, name='encoder', skip=False, bn_scope='batch_norm', is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        fb = 16
        k1 = 3
        padding = "CONSTANT"

        o_c1 = layers.general_conv2d(inputen, fb, 3, 3, 1, 1, 0.01, 'SAME', name="c1", norm_type="Batch", bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r1 = build_resnet_block(o_c1, fb, "r1", padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out1 = tf.nn.max_pool(o_r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r2 = build_resnet_block_ds(out1, fb, fb*2, "r2", padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out2 = tf.nn.max_pool(o_r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r3 = build_resnet_block_ds(out2, fb*2, fb*4, 'r3', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r4 = build_resnet_block(o_r3, fb*4, 'r4', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out3 = tf.nn.max_pool(o_r4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r5 = build_resnet_block_ds(out3, fb*4, fb*8, 'r5', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r6 = build_resnet_block(o_r5, fb*8, 'r6', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        
        o_r7 = build_resnet_block_ds(o_r6, fb*8, fb*16, 'r7', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r8 = build_resnet_block(o_r7, fb*16, 'r8', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        o_r9 = build_resnet_block(o_r8, fb*16, 'r9', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r10 = build_resnet_block(o_r9, fb * 16, 'r10', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        o_d1 = build_drn_block(o_r10, fb*16, 'd1', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_d2 = build_drn_block(o_d1, fb*16, 'd2', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        o_c2 = layers.general_conv2d(o_d2, fb*16, k1, k1, 1, 1, 0.01, 'SAME', 'c2', norm_type='Batch', bn_scope=bn_scope, is_training=is_training,keep_rate=keep_rate)
        o_c3 = layers.general_conv2d(o_c2, fb*16, k1, k1, 1, 1, 0.01, 'SAME', 'c3', norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        return o_c3


def build_encoder(inputen, name='encoder', skip=False, bn_scope='batch_norm', is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        fb = 16
        k1 = 3
        padding = "CONSTANT"

        o_c1 = layers.general_conv2d(inputen, fb, 7, 7, 1, 1, 0.01, 'SAME', name="c1", norm_type="Batch", bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r1 = build_resnet_block(o_c1, fb, "r1", padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out1 = tf.nn.max_pool(o_r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r2 = build_resnet_block_ds(out1, fb, fb*2, "r2", padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out2 = tf.nn.max_pool(o_r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r3 = build_resnet_block_ds(out2, fb*2, fb*4, 'r3', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r4 = build_resnet_block(o_r3, fb*4, 'r4', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        out3 = tf.nn.max_pool(o_r4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r5 = build_resnet_block_ds(out3, fb*4, fb*8, 'r5', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r6 = build_resnet_block(o_r5, fb*8, 'r6', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        o_r7 = build_resnet_block_ds(o_r6, fb*8, fb*16, 'r7', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r8 = build_resnet_block(o_r7, fb*16, 'r8', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        o_r9 = build_resnet_block(o_r8, fb*16, 'r9', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_r10 = build_resnet_block(o_r9, fb * 16, 'r10', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        o_d1 = build_drn_block(o_r10, fb*16, 'd1', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)
        o_d2 = build_drn_block(o_d1, fb*16, 'd2', padding, norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        o_c2 = layers.general_conv2d(o_d2, fb*16, k1, k1, 1, 1, 0.01, 'SAME', 'c2', norm_type='Batch', bn_scope=bn_scope, is_training=is_training,keep_rate=keep_rate)
        o_c3 = layers.general_conv2d(o_c2, fb*16, k1, k1, 1, 1, 0.01, 'SAME', 'c3', norm_type='Batch', bn_scope=bn_scope, is_training=is_training, keep_rate=keep_rate)

        return o_c3, o_r10


def build_decoder(inputde, inputimg, name='decoder', skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        o_c1 = layers.general_conv2d(inputde, ngf * 4, ks, ks, 1, 1, 0.02, "SAME", "c1", norm_type='Ins')
        o_r1 = build_resnet_block(o_c1, ngf * 4, "r1", padding, norm_type='Ins')
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding, norm_type='Ins')
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding, norm_type='Ins')
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding, norm_type='Ins')
        o_c3 = layers.general_deconv2d(o_r4, [BATCH_SIZE, 64, 64, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c3", norm_type='Ins')
        o_c4 = layers.general_deconv2d(o_c3, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4", norm_type='Ins')
        o_c5 = layers.general_deconv2d(o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5", norm_type='Ins')
        o_c6 = layers.general_conv2d(o_c5, 1, f, f, 1, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputimg + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def build_segmenter(inputse, name='segmenter', is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        fb = 16
        k1 = 1

        # o_c8 = layers.general_conv2d(inputse, fb*8, k1, k1, 1, 1, 0.01, 'SAME', 'c8', norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        
        o_c9 = layers.general_conv2d(inputse, 5, k1, k1, 1, 1, 0.01, 'SAME', 'c9', do_norm=False, do_relu=False, keep_rate=keep_rate)
        out_seg = tf.image.resize_images(o_c9, (256, 256))

        return out_seg

def build_segmenter_update(inputse, name='segmenter', is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        fb = 16
        k1 = 1

        # flat_conv10_1 = PS(inputse, r = 8, n_channel = 5 * 8, batch_size = BATCH_SIZE)
        # o_c9 = layers.general_conv2d(flat_conv10_1, 5, 5, 5, 1, 1, 0.01, 'SAME', 'c9', do_norm=False, do_relu=False, keep_rate=keep_rate)        
        # return o_c9

        o_c9 = layers.general_conv2d(inputse, 5, 1, 1, 1, 1, 0.01, 'SAME', 'c9', do_norm=False, do_relu=False, keep_rate=keep_rate)
        out_seg = tf.image.resize_images(o_c9, (256, 256))

        # feat_cls = tf.image.resize_images(inputse, (256, 256))
        # out_seg = layers.general_conv2d(feat_cls, 5, k1, k1, 1, 1, 0.01, 'SAME', 'c9', do_norm=False, do_relu=False, keep_rate=keep_rate)

        return out_seg


def discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        padw = 2

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2, 0.02, "VALID", "c1", do_norm=False, relufactor=0.2, norm_type='Ins')

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2, 0.02, "VALID", "c2", relufactor=0.2, norm_type='Ins')

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2, 0.02, "VALID", "c3", relufactor=0.2, norm_type='Ins')

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1, 0.02, "VALID", "c4", relufactor=0.2, norm_type='Ins')

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(pad_o_c4, 1, f, f, 1, 1, 0.02, "VALID", "c5", do_norm=False, do_relu=False)

        return o_c5


def discriminator_feature(inputdisc, name="discriminator_feature"):
    with tf.variable_scope(name):
        f = 4

        o_c1 = layers.general_conv2d(inputdisc, ndf * 8, f, f, 2, 2 , 0.02, "VALID", "c1", do_norm=False, relufactor=0.2, norm_type='Ins')
        o_c2 = layers.general_conv2d(o_c1, ndf * 8, f, f, 2, 2 , 0.02, "VALID", "c2", relufactor=0.2, norm_type='Ins')

        o_c3 = layers.general_conv2d(o_c2, 1, f, f, 1, 1, 0.02, "VALID", "c3", do_norm=False, do_relu=False)

        return o_c3


def discriminator_aux(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        padw = 2

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2, 0.02, "VALID", "c1", do_norm=False, relufactor=0.2, norm_type='Ins')

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2, 0.02, "VALID", "c2", relufactor=0.2, norm_type='Ins')

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2, 0.02, "VALID", "c3", relufactor=0.2, norm_type='Ins')

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1, 0.02, "VALID", "c4", relufactor=0.2, norm_type='Ins')

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(pad_o_c4, 2, f, f, 1, 1, 0.02, "VALID", "c5", do_norm=False, do_relu=False)

        return tf.expand_dims(o_c5[...,0], axis=3), tf.expand_dims(o_c5[...,1], axis=3)

