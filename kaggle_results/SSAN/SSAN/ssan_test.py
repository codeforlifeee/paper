# -*- coding: utf-8 -*-
"""
@author: Erting Pan
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import indices
data = indices.ReadDatasets()

num_classes = 16
num_input   = 1
timesteps   = 200
batch_size  = 128

prediction = np.zeros((0, num_classes), dtype=np.int32)
true_y     = np.zeros((0, num_classes), dtype=np.int32)

import glob
ckpt_meta_files = glob.glob('./model/ssan_step_*.ckpt.meta')
if not ckpt_meta_files:
    ckpt_base = './model/ssan030.ckpt'
else:
    ckpt_meta_files.sort(key=lambda x: int(x.split('_step_')[1].split('.ckpt')[0]))
    ckpt_base = ckpt_meta_files[-1].replace('.meta', '')
print(f"Loading checkpoint: {ckpt_base}")
saver = tf.train.import_meta_graph(ckpt_base + '.meta')
# Start testing
with tf.Session() as sess:
    saver.restore(sess, ckpt_base)
    spe_X = sess.graph.get_operation_by_name('spe_X').outputs[0]
    spa_X = sess.graph.get_operation_by_name('spa_X').outputs[0]
    y     = sess.graph.get_tensor_by_name('Softmax:0')
    keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]
    for index in range((data.test._num_examples // batch_size) + 1):
        batch_spe_x, batch_spa_x, Y = data.test.next_batch_test(batch_size)
        if index == (data.test._num_examples // batch_size):
            batch_spe_x = batch_spe_x.reshape(((data.test._num_examples % batch_size), timesteps, num_input))
        else:
            batch_spe_x = batch_spe_x.reshape((batch_size, timesteps, num_input))
        pre_pro     = sess.run(y, feed_dict={spe_X: batch_spe_x, spa_X: batch_spa_x, keep_prob: 1.0})
        prediction  = np.concatenate((prediction, pre_pro), axis=0)
        true_y      = np.concatenate((true_y, Y), axis=0)
predict_label = np.argmax(prediction, 1) + 1
true_y        = np.argmax(true_y, 1) + 1

every_class, confusion_mat =indices.CalAccuracy(true_y,predict_label, num_classes)
acc= every_class[:]
print("Accuracy for Testing sets:",acc)



import glob
ckpt_meta_files = glob.glob('./model/ssan_step_*.ckpt.meta')
if not ckpt_meta_files:
    ckpt_base = './model/ssan030.ckpt'
else:
    ckpt_meta_files.sort(key=lambda x: int(x.split('_step_')[1].split('.ckpt')[0]))
    ckpt_base = ckpt_meta_files[-1].replace('.meta', '')
saver = tf.train.import_meta_graph(ckpt_base + '.meta')
prediction = np.zeros((0, num_classes), dtype=np.int32)
with tf.Session() as sess:
    saver.restore(sess, ckpt_base)
    spe_X = sess.graph.get_operation_by_name('spe_X').outputs[0]
    spa_X = sess.graph.get_operation_by_name('spa_X').outputs[0]
    y     = sess.graph.get_tensor_by_name('Softmax:0')
    keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]
    for index in range((data.all._num_examples // batch_size) + 1):
        batch_spe_x, batch_spa_x, Y = data.all.next_batch_test(batch_size)
        if index == (data.all._num_examples // batch_size):
            batch_spe_x = batch_spe_x.reshape(((data.all._num_examples % batch_size), timesteps, num_input))
        else:
            batch_spe_x = batch_spe_x.reshape((batch_size, timesteps, num_input))
        pre_pro     = sess.run(y, feed_dict={spe_X: batch_spe_x, spa_X: batch_spa_x, keep_prob: 1.0})
        prediction  = np.concatenate((prediction, pre_pro), axis=0)
predict_label = np.argmax(prediction, 1) + 1

image = indices.ColorResult(predict_label)

print("ok")
