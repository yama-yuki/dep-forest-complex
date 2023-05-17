# Notes on the implementation

* Training data reading is a bit tricky and buggy. Please check the code in
  `dataset.py` and `vocab.py`. In particular, you need to tweak `file_iterator`
  in `dataset.py` and `add_train_file` in `vocab.py`.

* `RadamOptimizer` is buggy and does not work as expected, probably because
  there's significant changes in handling model parameters in TF. Thus, Adam
  optimizer is directly borrwed from TF which works as expected.

* 1-best parsing algorithm is based on maximum (minimum) spanning tree
  alorithm. See `prob_argmax` of `lib/models/parser/parser.py` and
  `parse_argmax` of `lib/models/nn.py`.

* Eisner's algorithm simply uses the edge scores for parsing. See
  `lib/models/parsers/eisner_nbest.py` for details, in particular, the simple
  unit test. The code seems to be portable so that you have only to feed two
  NumPy array, one for relation probabilitis and the other for label
  probabilities.

  * It will return nbest parses, not hypergraphs. However, it is probably
    trivial to output a hypergraph by returning the intermediate data
    structures.

* The edge-wise method simply enumerates the probable edge with labels by
  multiplying the relation and label probabilites without considering the tree
  structures. I feel the code should be modified to compute the posterior
  probabilities. See `vallidate_cube` in `lib/models/parse/base_parser.py`.

  * For details on computing posteriors, check Kirchhoff's matrix tree
    theorem which allows computing all the non-projective tree
    structures. Basically, you need to compute the matrix inverse with
    approprivate transformation of logits. See
    [https://aclanthology.org/D07-1015/](https://aclanthology.org/D07-1015/),
    [https://aclanthology.org/D07-1014/](https://aclanthology.org/D07-1014/) and
    [https://aclanthology.org/I17-1007/](https://aclanthology.org/I17-1007/).
