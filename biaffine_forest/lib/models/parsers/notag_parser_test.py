from absl.testing import absltest
import tensorflow.compat.v1 as tf

import configurable
import dataset
from lib.models.parsers import notag_parser
import vocab

tf.disable_eager_execution()


class NoTagParserTest(absltest.TestCase):

  def test_parser(self):
    with tf.Session() as sess:
      config = configurable.Configurable(config_file='test_data/test.cfg')

      vocab_files = [(config.word_file, 1, 'Words'),
                     (config.tag_file, [3, 4], 'Tags'),
                     (config.rel_file, 7, 'Rels')]
      vocabs = []
      for i, (vocab_file, index, name) in enumerate(vocab_files):
        vocabs.append(
            vocab.Vocab(
                vocab_file,
                index,
                name=name,
                cased=config.cased if not i else True,
                use_pretrained=(not i),
                config_file='test_data/test.cfg'))

      ds = dataset.Dataset(
          config.train_file,
          vocabs,
          lambda: None,
          config_file='test_data/test.cfg')

      parser = notag_parser.NoTagParser(config_file='test_data/test.cfg')
      outputs = parser(ds)

      mini_batches = ds.get_minibatches(ds.train_batch_size, parser.input_idxs,
                                        parser.target_idxs, parser.forest_idxs)
      feed_dict, _ = next(mini_batches)
      sess.run(tf.global_variables_initializer())
      sess.run(outputs, feed_dict=feed_dict)


if __name__ == '__main__':
  absltest.main()
