from absl.testing import absltest
import tensorflow.compat.v1 as tf

import configurable
import vocab

tf.disable_eager_execution()

class VocabTest(absltest.TestCase):

  def test_vocab(self):
    with tf.Session() as sess:
      config = configurable.Configurable(config_file='test_data/test.cfg')
      vocabulary = vocab.Vocab(
        config.word_file,
        1,
        use_pretrained=True,
        config_file='test_data/test.cfg')

      self.assertEqual(vocabulary['of'], (5, 6))
      self.assertEqual(vocabulary['UNKNOWN-WORD'], (2, 2))

      lookup, pretrained_lookup = vocabulary.embedding_lookup([5, 2], [6, 2])
      sess.run(tf.global_variables_initializer())
      lookup_np, pretrained_lookup_np = sess.run((lookup, pretrained_lookup))

      self.assertEqual(lookup_np.shape, (2, 3))
      self.assertEqual(pretrained_lookup_np.shape, (2, 3))


if __name__ == '__main__':
  absltest.main()
