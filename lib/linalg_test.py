from absl.testing import absltest
import tensorflow.compat.v1 as tf

from lib import linalg

tf.disable_eager_execution()


class LinalgTest(tf.test.TestCase):

  def test_linear(self):
    dimension = 7
    batch_size = 3
    output_dimension = 11
    length = 5

    with tf.Session() as sess:
      x = tf.random.normal([batch_size, length, dimension])
      outputs = linalg.linear(x, output_dimension)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, output_dimension))

  def test_bilinear(self):
    dimension1 = 7
    dimension2 = 13
    batch_size = 3
    output_dimension = 11
    length = 5

    with tf.Session() as sess:
      x1 = tf.random.normal([batch_size, length, dimension1])
      x2 = tf.random.normal([batch_size, length, dimension2])
      outputs = linalg.bilinear(x1, x2, output_dimension)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape,
                       (batch_size, length, output_dimension, length))

  def test_diagonal_bilinear(self):
    dimension = 7
    batch_size = 3
    output_dimension = 11
    length = 5

    with tf.Session() as sess:
      x1 = tf.random.normal([batch_size, length, dimension])
      x2 = tf.random.normal([batch_size, length, dimension])
      outputs = linalg.diagonal_bilinear(x1, x2, output_dimension)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape,
                       (batch_size, length, output_dimension, length))


if __name__ == '__main__':
  absltest.main()
