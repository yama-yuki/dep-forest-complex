# Dependency Forest Rescoring
Create Binarized Dependency Forest [Hayashi+,13] with Forest Parser [Song+,19] to do rescoring through Cube Pruning [Huang,08].
Our system comprises 3 components including `Dependency Forest Parser`, `Rescoring Module`, and `Forest Decoder`.

## Dependency Forest Parser
Implementation based on https://github.com/freesunshine0316/dep-forest-re/tree/master/biaffine_forest [Song+,19]

Notes from [Song+,19]:
* 1-best parsing algorithm is based on maximum (minimum) spanning tree
  alorithm. See `prob_argmax` of `lib/models/parser/parser.py` and
  `parse_argmax` of `lib/models/nn.py`.
* Eisner's algorithm simply uses the edge scores for parsing. See
  `lib/models/parsers/eisner_nbest.py` for details, in particular, the simple
  unit test. The code seems to be portable so that you have only to feed two
  NumPy array, one for relation probabilitis and the other for label
  probabilities. 

Extensions:
* In `eisner_nbest.py`, we added `eisner_dp_forest` that outputs a forest (a set of hypergraphs) in .json format.
* `DepHeadBinarizer` converts intermediate eisner span into binarized structure `BinHyperedge` by Head-Binarization method to resolve spuriousness.
* 

## Rescoring Module
some experimental results and stuff ...

## Forest Decoder
Implementation of [Huang,08]
