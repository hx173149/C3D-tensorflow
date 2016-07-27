"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import model_build 
import math
import numpy as np


# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 4 
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 16 , 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')

FLAGS = flags.FLAGS

MOVING_AVERAGE_DECAY = 0.999

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         model_build.FRAMES,
                                                         model_build.IMAGE_SIZE,
                                                         model_build.IMAGE_SIZE,
                                                         model_build.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def _variable_on_cpu(name, shape, initializer,cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd,cpu_id):
  var = _variable_on_cpu(name, shape,tf.truncated_normal_initializer(stddev=stddev),cpu_id)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def tower_loss(scope,images,labels,weights,biases,batch_size):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  logits = model_build.inference_c3d(images,0.5,batch_size,weights,biases) 
  cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
  cross_entropy_mean = tf.Print(cross_entropy_mean,[cross_entropy_mean],'cross_entropy_mean: ')
  tf.add_to_collection('losses', cross_entropy_mean)

  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')
  total_loss = tf.Print(total_loss,[total_loss],'loss: ')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='loss')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  correct_pred = tf.equal(tf.argmax(logits,1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss,accuracy

def run_training():
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size*gpu_num)
    lr = tf.placeholder(tf.float32, shape=[])
    tower_grads = []
    accuracys = []
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    for gpu_index in range(0,gpu_num):
      with tf.device('/gpu:%d' % (gpu_index)):
        with tf.name_scope('%s_%d' % ('c3d', gpu_index)) as scope:
          with tf.variable_scope('c3d_var') as var_scope:
            weights = {
              'wc1': _variable_with_weight_decay('wc1',[3, 3, 3, 3, 64],0.04,0.00,gpu_index),
              'wc2': _variable_with_weight_decay('wc2',[3, 3, 3, 64, 128],0.04,0.00,gpu_index),
              'wc3a': _variable_with_weight_decay('wc3a',[3, 3, 3, 128, 256],0.04,0.00,gpu_index),
              'wc3b': _variable_with_weight_decay('wc3b',[3, 3, 3, 256, 256],0.04,0.00,gpu_index),
              'wc4a': _variable_with_weight_decay('wc4a',[3, 3, 3, 256, 512],0.04,0.00,gpu_index),
              'wc4b': _variable_with_weight_decay('wc4b',[3, 3, 3, 512, 512],0.04,0.00,gpu_index),
              'wc5a': _variable_with_weight_decay('wc5a',[1, 3, 3, 512, 512],0.04,0.00,gpu_index),
              'wc5b': _variable_with_weight_decay('wc5b',[1, 3, 3, 512, 512],0.04,0.00,gpu_index),
              'wd1': _variable_with_weight_decay('wd1',[8192, 4096],0.04,0.005,gpu_index),
              'wd2': _variable_with_weight_decay('wd2',[4096, 4096],0.04,0.005,gpu_index),
              'out': _variable_with_weight_decay('wout',[4096, model_build.NUM_CLASSES],0.04,0.005,gpu_index)
            }
            biases = {
              'bc1': _variable_with_weight_decay('bc1',[64],0.04,0.0,gpu_index),
              'bc2': _variable_with_weight_decay('bc2',[128],0.04,0.0,gpu_index),
              'bc3a': _variable_with_weight_decay('bc3a',[256],0.04,0.0,gpu_index),
              'bc3b': _variable_with_weight_decay('bc3b',[256],0.04,0.0,gpu_index),
              'bc4a': _variable_with_weight_decay('bc4a',[512],0.04,0.0,gpu_index),
              'bc4b': _variable_with_weight_decay('bc4b',[512],0.04,0.0,gpu_index),
              'bc5a': _variable_with_weight_decay('bc5a',[512],0.04,0.0,gpu_index),
              'bc5b': _variable_with_weight_decay('bc5b',[512],0.04,0.0,gpu_index),
              'bd1': _variable_with_weight_decay('bd1',[4096],0.04,0.0,gpu_index),
              'bd2': _variable_with_weight_decay('bd2',[4096],0.04,0.0,gpu_index),
              'out': _variable_with_weight_decay('bout',[model_build.NUM_CLASSES],0.04,0.0,gpu_index),
            }
          tf.get_variable_scope().reuse_variables()
          loss,accuracy = tower_loss(scope,images_placeholder[gpu_index*FLAGS.batch_size:(gpu_index+1)*FLAGS.batch_size,:,:,:,:],
                  labels_placeholder[gpu_index*FLAGS.batch_size:(gpu_index+1)*FLAGS.batch_size],weights,biases,FLAGS.batch_size) 
          grads = opt.compute_gradients(loss)
          tower_grads.append(grads)
          accuracys.append(accuracy) 
    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op, variables_averages_op) 
    init_op = tf.initialize_all_variables()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # Create a session for running Ops on the Graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)
    #saver.restore(sess,"./model_c3d_ucf101_0713_pm")
    # And then after everything is built, start the training loop.
    lr_step = 2000
    for step in xrange(FLAGS.max_steps):
      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      start_time = time.time()
      print ('learning rate: ' + str(FLAGS.learning_rate*pow(0.1,(int)(step/lr_step))))
      train_images,train_labels = input_data.ReadDataLabelFromFile_16('list/train_ucf101.trainVideos',FLAGS.batch_size*gpu_num)
      sess.run(train_op, feed_dict={images_placeholder: train_images, 
          labels_placeholder: train_labels,
          lr: FLAGS.learning_rate*pow(0.1,(int)(step/lr_step))
          }
          )
      duration = time.time() - start_time
      print('Step %d: %.3f sec' % (step, duration))

      # Save a checkpoint and evaluate the model periodically.
      if (step) % 50 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, 'models_0727/my_modle', global_step=step)
        print('Training Data Eval:')
        acc = sess.run(accuracys, feed_dict={images_placeholder: train_images, labels_placeholder: train_labels})
        acc = np.array(acc).mean()
        print ("accuracy: " + "{:.5f}".format(acc))
        print('Validation Data Eval:')
        val_images,val_labels = input_data.ReadDataLabelFromFile_16('list/test_ucf101.trainVideos',FLAGS.batch_size*gpu_num)
        acc = sess.run(accuracys, feed_dict={images_placeholder: val_images, labels_placeholder: val_labels})
        acc = np.array(acc).mean()
        print ("accuracy: " + "{:.5f}".format(acc))
  print("done")

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
