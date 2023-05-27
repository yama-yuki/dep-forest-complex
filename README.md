# Dependency Forest Rescoring
Create Binarized Dependency Forest [Hayashi+,13] with Forest Parser [Song+,19] to do rescoring through Cube Pruning [Huang,08].
Our system comprises 3 components including `Dependency Forest Parser`, `Rescoring Module`, and `Forest Decoder`.

## Dependency Forest Parser
Implementation based on https://github.com/freesunshine0316/dep-forest-re/tree/master/biaffine_forest [Song+,19]

Notes from [Song+,19]:
* 1-best parsing algorithm is based on maximum (minimum) spanning tree
  alorithm. See `prob_argmax` of `biaffine_forest/lib/models/parser/parser.py` and
  `parse_argmax` of `biaffine_forest/lib/models/nn.py`.
* Eisner's algorithm simply uses the edge scores for parsing. See
  `biaffine_forest/lib/models/parsers/eisner_nbest.py` for details, in particular, the simple
  unit test. The code seems to be portable so that you have only to feed two
  NumPy array, one for relation probabilitis and the other for label
  probabilities. 

Extensions: `biaffine_forest/lib/models/parser/eisner_nbest.py` (also `biaffine_forest/lib/models/parser/base_parser.py` and `biaffine_forest/network.py`)

* In `eisner_nbest.py`, we added `eisner_dp_forest` that outputs a forest (a set of hypergraphs) in .json format.
* `DepHeadBinarizer` converts intermediate eisner spans into binarized structure `BinHyperedge` by Head-Binarization method to resolve spuriousness.
* todo: current implementation targets all spans (including incomplete) which may be causing some errors in later cube pruning. better to target only complete spans

## Rescoring Module
some experimental results and stuff: `rescore_module`

## Forest Decoder
CURRENTLY BEING FIXED: `forest_decoder.py`

Adopting Cube Pruning for Binarized Dependency Forest

`main_loop`: load hyperedges from forest .json, initialize derivation memory, select_k for each node (head, lgov, rgov), output final derivation

`select_k`: actual cube pruning function: find incoming edges, initialize cubes, manage priority queue

`cube_next`: search on cubes, rescore inside

`bert_rescore`: actual scoring function

The scores in each step do not seem to align with vanilla k-best even without rescoring
* Bugs (some duplication fixed, overevaluation: somehow keeps spans with the lower scores, tends to prune spans with broader range)
* todo?: Rescoring model sometimes overevaluates the case where the verb of matrix clause becomes the dependent of the verb of subordinate clause. Threshold for rescoring S_BERT (e.g., rescore only when S_BERT > S_biaffine)?
* todo?: However, it's better not adding  much complicated rules... considering improvements training-wise

## etc
`analyze_deprel.py`: head prediction accuracy based on deprel

`analyze_forest.py`: density: nodes&edges per sentence, check if gold tree is included (in progress)
