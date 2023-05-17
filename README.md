# Notes on the implementation
Notes from [Song+,19]:
* 1-best parsing algorithm is based on maximum (minimum) spanning tree
  alorithm. See `prob_argmax` of `lib/models/parser/parser.py` and
  `parse_argmax` of `lib/models/nn.py`.

* Eisner's algorithm simply uses the edge scores for parsing. See
  `lib/models/parsers/eisner_nbest.py` for details, in particular, the simple
  unit test. The code seems to be portable so that you have only to feed two
  NumPy array, one for relation probabilitis and the other for label
  probabilities. 

Modifications made:
* In `eisner_nbest.py`, we added `eisner_dp_forest` that outputs a forest .json (a set of hypergraphs)
