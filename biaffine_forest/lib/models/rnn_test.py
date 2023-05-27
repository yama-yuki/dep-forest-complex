from absl.testing import absltest
import tensorflow.compat.v1 as tf

from lib.models import rnn
from lib.rnn_cells import rnn_cell

tf.disable_eager_execution()


class RNNTest(absltest.TestCase):

  def test_rnn(self):
    dimension = 5
    batch_size = 3
    length = 7

    with tf.Session() as sess:
      cell = rnn_cell.RNNCell(
          input_size=dimension,
          output_size=dimension,
          config_file='test_data/test.cfg')

      x = tf.random.uniform([batch_size, length, dimension])
      # TODO(taro): Check whether the code base is aware of the variable scopes.
      with tf.variable_scope("rnn"):
        outputs, state = rnn.rnn(cell, tf.unstack(x, axis=1), dtype=tf.float32)

      sess.run(tf.global_variables_initializer())
      outputs_np, state_np = sess.run((outputs, state))

      self.assertLen(outputs_np, 7)
      for o in outputs_np:
        self.assertEqual(o.shape, (batch_size, dimension))
      self.assertEqual(state_np.shape, (batch_size, dimension))

  def test_bidirectional_rnn(self):
    dimension = 5
    batch_size = 3
    length = 7

    with tf.Session() as sess:
      cell_fw = rnn_cell.RNNCell(
          input_size=dimension,
          output_size=dimension,
          config_file='test_data/test.cfg')
      cell_bw = rnn_cell.RNNCell(
          input_size=dimension,
          output_size=dimension,
          config_file='test_data/test.cfg')

      x = tf.random.uniform([batch_size, length, dimension])
      # TODO(taro): Check whether the code base is aware of the variable scopes.
      with tf.variable_scope("bidirectional_rnn"):
        outputs, state_fw, state_bw = rnn.bidirectional_rnn(
            cell_fw, cell_bw, tf.unstack(x, axis=1), dtype=tf.float32)

      sess.run(tf.global_variables_initializer())
      outputs_np, state_fw_np, state_bw_np = sess.run(
          (outputs, state_fw, state_bw))

      self.assertLen(outputs_np, 7)
      for o in outputs_np:
        self.assertEqual(o.shape, (batch_size, dimension * 2))
      self.assertEqual(state_fw_np.shape, (batch_size, dimension))
      self.assertEqual(state_bw_np.shape, (batch_size, dimension))

  def test_dynamic_rnn(self):
    dimension = 5
    batch_size = 3
    length = 7

    with tf.Session() as sess:
      cell = rnn_cell.RNNCell(
          input_size=dimension,
          output_size=dimension,
          config_file='test_data/test.cfg')
      x = tf.random.uniform([batch_size, length, dimension])
      # TODO(taro): Check whether the code base is aware of the variable scopes.
      with tf.variable_scope("dynamic_rnn"):
        outputs, state = rnn.dynamic_rnn(cell, x, dtype=tf.float32)

      sess.run(tf.global_variables_initializer())
      outputs_np, state_np = sess.run((outputs, state))

      self.assertEqual(outputs_np.shape, (batch_size, length, dimension))
      self.assertEqual(state_np.shape, (batch_size, dimension))

  def test_bidirectional_dynamic_rnn(self):
    dimension = 5
    batch_size = 3
    length = 7


    with tf.Session() as sess:
      cell_fw = rnn_cell.RNNCell(
          input_size=dimension,
          output_size=dimension,
          config_file='test_data/test.cfg')
      cell_bw = rnn_cell.RNNCell(
          input_size=dimension,
          output_size=dimension,
          config_file='test_data/test.cfg')

      x = tf.random.uniform([batch_size, length, dimension])
      # TODO(taro): Check whether the code base is aware of the variable scopes.
      with tf.variable_scope("dynamic_bidirectional_rnn"):
        outputs, state_fw, state_bw = rnn.dynamic_bidirectional_rnn(
            cell_fw, cell_bw, x, [length] * batch_size, dtype=tf.float32)

      sess.run(tf.global_variables_initializer())
      outputs_np, state_fw_np, state_bw_np = sess.run(
          (outputs, state_fw, state_bw))

      self.assertEqual(outputs_np.shape, (batch_size, length, dimension * 2))
      self.assertEqual(state_fw_np.shape, (batch_size, dimension))
      self.assertEqual(state_bw_np.shape, (batch_size, dimension))

if __name__ == '__main__':
  absltest.main()
