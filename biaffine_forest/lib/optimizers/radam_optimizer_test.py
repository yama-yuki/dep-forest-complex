from absl.testing import absltest
import numpy as np
import tensorflow.compat.v1 as tf

from lib.optimizers import radam_optimizer

tf.disable_eager_execution()


class RadamOptimizerTest(tf.test.TestCase):

  def test_simple(self):
    batch_size = 3
    dimension = 7
    classes = 5

    with tf.Session() as sess:
      optimizer = radam_optimizer.RadamOptimizer(config_file='test_data/test.cfg')

      inputs = tf.random.normal([batch_size, dimension])
      labels = tf.nn.softmax(tf.random.normal([batch_size, classes]), axis=1)

      matrix = tf.get_variable('var', [dimension, classes])
      logits = tf.matmul(inputs, matrix)
      loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

      ops = optimizer.minimize(loss)

      sess.run(tf.global_variables_initializer())
      self.assertEqual(sess.run(optimizer.global_step), 0)
      matrix1_np = sess.run(matrix)

      sess.run(ops)

      self.assertEqual(sess.run(optimizer.global_step), 1)
      matrix2_np = sess.run(matrix)

      # Check any updates.
      self.assertTrue((np.abs(matrix2_np - matrix1_np) > 1e-7).any())

      sess.run(ops)

      self.assertEqual(sess.run(optimizer.global_step), 2)
      matrix3_np = sess.run(matrix)
      self.assertTrue((np.abs(matrix3_np - matrix2_np) > 1e-7).any())

if __name__ == '__main__':
  absltest.main()
