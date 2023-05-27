from absl.testing import absltest
import tensorflow.compat.v1 as tf

from lib.rnn_cells import cif_lstm_cell

tf.disable_eager_execution()


class CifLSTMCellTest(absltest.TestCase):

  def test_simple(self):
    dimension = 5
    batch_size = 3

    with tf.Session() as sess:
      cell = cif_lstm_cell.CifLSTMCell(
          input_size=dimension,
          output_size=dimension,
          config_file='test_data/test.cfg')
      init = cell.zero_state(batch_size, tf.float32)
      x = tf.random.normal([batch_size, dimension])
      hidden, state = cell(x, init)

      sess.run(tf.global_variables_initializer())
      hidden_np, state_np = sess.run((hidden, state))
      self.assertEqual(hidden_np.shape, (batch_size, dimension))
      self.assertEqual(state_np.shape, (batch_size, dimension * 2))


if __name__ == '__main__':
  absltest.main()
