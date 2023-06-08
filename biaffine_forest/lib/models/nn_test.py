from absl.testing import absltest
import numpy as np
import tensorflow.compat.v1 as tf

from lib.models import nn

tf.disable_eager_execution()


class NNTest(tf.test.TestCase):

  def test_embed_concat(self):
    word_dimension = 5
    tag_dimension = 1
    batch_size = 3
    length = 7

    with tf.Session() as sess:
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      words = tf.random.normal([batch_size, length, word_dimension])
      tags = tf.random.normal([batch_size, length, tag_dimension])
      outputs = nn_obj.embed_concat(words, tags)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape,
                       (batch_size, length, word_dimension + tag_dimension))

  def test_rnn(self):
    # test_data/test.cfg specifies 7 dimension with bidirectional-LSTM
    dimension = 7
    batch_size = 3
    length = 5

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      x = tf.random.normal([batch_size, length, dimension])
      nn_obj.sequence_lengths = [length] * batch_size
      outputs, state = nn_obj.RNN(x)

      sess.run(tf.global_variables_initializer())
      outputs_np, state_np = sess.run((outputs, state))

      self.assertEqual(state_np.shape, (batch_size, dimension * 2))
      self.assertEqual(outputs_np.shape, (batch_size, length, dimension * 2))

  def test_soft_attn(self):
    # test_data/test.cfg specifies 7 dimension with bidirectional-LSTM
    dimension = 7
    batch_size = 3
    length = 5

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      x = tf.random.normal([batch_size, length, dimension * 2])
      outputs = nn_obj.soft_attn(x)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, dimension * 4))

  def test_linear(self):
    dimension = 7
    batch_size = 3
    length = 5

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      x = tf.random.normal([batch_size, length, dimension])
      outputs = nn_obj.linear(x, dimension)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, dimension))

  def test_softmax(self):
    dimension = 7
    batch_size = 3
    length = 5

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      x = tf.random.normal([batch_size, length, dimension])
      outputs = nn_obj.softmax(x)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, dimension))
      self.assertAllClose(outputs_np.sum(axis=2), np.ones((batch_size, length)))

  def test_mlp(self):
    dimension = 7
    batch_size = 3
    length = 5

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      x = tf.random.normal([batch_size, length, dimension])
      outputs = nn_obj.MLP(x, dimension)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, dimension))

  def test_double_mlp(self):
    # test_data/test.cfg specifies 5 dimension with bidirectional-LSTM
    dimension = 5
    batch_size = 3
    length = 5

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      x = tf.random.normal([batch_size, length, dimension])
      # TODO(taro): Investigates what is doing here.
      outputs = nn_obj.double_MLP(x)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape,
                       (batch_size, length, length, dimension))

  def test_linear_classifier(self):
    classes = 5
    dimension = 7
    batch_size = 3
    length = 11

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      x = tf.random.normal([batch_size, length, dimension])
      outputs = nn_obj.linear_classifier(x, classes)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, classes))

  def test_bilinear_classifier(self):
    dimension = 7
    dimension2 = 5
    batch_size = 3
    length = 11

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      # TODO(taro): Verifies what is doing here.
      x1 = tf.random.normal([batch_size, length, dimension])
      x2 = tf.random.normal([batch_size, dimension2, dimension])
      outputs = nn_obj.bilinear_classifier(x1, x2)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, dimension2))

  def test_diagonal_bilinear_classifier(self):
    dimension = 7
    dimension2 = 5
    batch_size = 3
    length = 11

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      # TODO(taro): Verifies what is doing here.
      x1 = tf.random.normal([batch_size, length, dimension])
      x2 = tf.random.normal([batch_size, dimension2, dimension])
      outputs = nn_obj.diagonal_bilinear_classifier(x1, x2)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, dimension2))

  def test_conditional_linear_classifier(self):
    classes = 5
    dimension = 7
    batch_size = 3
    length = 11

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      # TODO(taro): Verifies what is doing here.
      x = tf.random.normal([batch_size, length, length, dimension])
      probs = tf.nn.softmax(
          tf.random.normal([batch_size, length, length]), axis=2)
      weighted, outputs = nn_obj.conditional_linear_classifier(
          x, classes, probs)

      sess.run(tf.global_variables_initializer())
      weighted_np, outputs_np = sess.run((weighted, outputs))

      self.assertEqual(outputs_np.shape, (batch_size, length, length, classes))
      self.assertEqual(weighted_np.shape, (batch_size, length, classes, 1))

  def test_conditional_diagonal_bilinear_classifier(self):
    classes = 5
    dimension = 7
    batch_size = 3
    length = 11

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      # TODO(taro): Verifies what is doing here.
      x1 = tf.random.normal([batch_size, length, dimension])
      x2 = tf.random.normal([batch_size, length, dimension])
      probs = tf.nn.softmax(
          tf.random.normal([batch_size, length, length]), axis=2)
      weighted, outputs = nn_obj.conditional_diagonal_bilinear_classifier(
          x1, x2, classes, probs)

      sess.run(tf.global_variables_initializer())
      weighted_np, outputs_np = sess.run((weighted, outputs))

      self.assertEqual(outputs_np.shape, (batch_size, length, classes, length))
      self.assertEqual(weighted_np.shape, (batch_size, length, classes, 1))

  def test_output(self):
    classes = 5
    batch_size = 3
    length = 7

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')
      nn_obj.tokens_to_keep3D = tf.ones([batch_size, length])
      nn_obj.n_tokens = tf.convert_to_tensor(length + 0.)

      # TODO(taro): Verifies what is doing here.
      logits = tf.random.normal([batch_size, length, classes])
      targets = tf.random.uniform([batch_size, length],
                                  minval=0,
                                  maxval=classes,
                                  dtype=tf.int32)

      outputs_dict = nn_obj.output(logits, targets)

      sess.run(tf.global_variables_initializer())
      outputs_dict_np = sess.run(outputs_dict)

  def test_conditional_probabilities(self):
    classes = 5
    batch_size = 3
    length = 7
    length2 = 11

    with tf.Session() as sess:
      # test_data/test.cfg states
      nn_obj = nn.NN(config_file='test_data/test.cfg')

      logits = tf.random.normal([batch_size, length, classes, length2])
      outputs = nn_obj.conditional_probabilities(logits)

      sess.run(tf.global_variables_initializer())
      outputs_np = sess.run(outputs)

      self.assertEqual(outputs_np.shape, (batch_size, length, length2, classes))
      self.assertAllClose(
          outputs_np.sum(axis=3), np.ones((batch_size, length, length2)))


if __name__ == '__main__':
  absltest.main()
