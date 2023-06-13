# Dependency Forest Rescoring
Create Binarized Dependency Forest [Hayashi+,13] with Forest Parser [Song+,19] to do rescoring through Cube Pruning [Huang,08].
Our system comprises 3 components including `Dependency Forest Parser`, `Rescoring Module`, and `Forest Decoder`.

## Dependency Forest Parser
We have extended the implementation of https://github.com/freesunshine0316/dep-forest-re/tree/master/biaffine_forest [Song+,19]

Notes from [Song+,19]:
* 1-best parsing algorithm is based on maximum (minimum) spanning tree
  algorithm. See `prob_argmax` of `biaffine_forest/lib/models/parser/parser.py` and
  `parse_argmax` of `biaffine_forest/lib/models/nn.py`.
* Eisner's algorithm simply uses the edge scores for parsing. See
  `biaffine_forest/lib/models/parsers/eisner_nbest.py` for details, in particular, the simple
  unit test. The code seems to be portable so that you have only to feed two
  NumPy array, one for relation probabilitis and the other for label
  probabilities. 

Extensions: `biaffine_forest/lib/models/parser/eisner_nbest.py` (also `biaffine_forest/lib/models/parser/base_parser.py` and `biaffine_forest/network.py`)

* In `eisner_nbest.py`, we have added `eisner_dp_forest` that outputs a forest (a set of dependency hyperedges) in .json format.
* `DepHeadBinarizer` converts intermediate eisner spans (only complete spans) into binarized structure `BinHyperedge` by Head-Binarization method to resolve spuriousness.

## Rescoring Module
Some experimental results and stuff in `rescore_module`

* BERT-based sentence-pair model that predicts the head span of an input word (snt1) in an input sentence (snt2).
* Our best performing model `bert-base-uncased_1_2_3e-5_32` is placed in `rescore_module/models/`.

Accuracy(%) of head prediction on UD-v2 by number of subordinate clauses in a sentence:

| # | biaffine | token | children | g-children |
|:---|:---:|:---:|:---:|---:|
| 1 | 88.15 | 89.30 | 90.45 | 89.57 |
| 2 | 84.63 | 88.42 | 90.07 | 89.83 |
| 3 | 78.95 | 84.21 | 80.70 | 82.46 |

Models:

`biaffine`: biaffine classifier

`token`: BERT trained on token-level input in snt1

`children`: BERT trained on children-level input in snt1

`grand-children`: BERT trained on grand-children-level input in snt1

## Forest Decoder
Bugs (duplicate computation / overevaluation) fixed in Third Commit of `forest_decoder.py`

(A) Forest Reader: take out Xspans from forest and sort them for cube pruning

(B) Cube Pruning Algorithm: the part where combinatory operation of subderivations (Aspan & Bspan) and rescoring happens

* This algorithm searches Kbest derivations of a **Xspan** which is a triplet of **X** (head node), **a** (leftmost governing span boundary of X), and **b** (rightmost governing span boundary of X)
* Then returns the resulting 1-best dependency tree with a root node (0) governing a (-1) to b (snt_len-1)

(C) Rescoring Function: applied inside `cube_next()` function when combining Aspan & Bsapan

Cube Pruning Algorthm overview:

`main_loop()`: load hyperedges from forest .json, initialize derivation chart with Xspans(X,a,b), `select_k()` for each Xspan, output final derivation Xspan(0,-1,snt_len-1)

`select_k()`: actual cube pruning function to find incoming edges, initialize cubes, manage priority queue, check validity of new Xspan derivation, insert new derivations to `best_K_buffer`, and move on to next cube grid

`cube_next()`: look for next best combination of Aspan & Bspan and rescore inside cubes

`bert_rescore()`: actual rescoring function, applied when conditions are met

## etc
`analyze_deprel.py`: evaluate head prediction accuracy based on deprel

`analyze_forest.py`: analyze density (nodes&edges per sentence) of a forest and check if the correct tree is included (in progress)




