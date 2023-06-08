#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ossaudiodev import SNDCTL_TMR_CONTINUE

import numpy as np
import tensorflow.compat.v1 as tf

from vocab import Vocab
from lib.models import NN
#from lib.models.parsers import eisner_dp_nbest
from lib.models.parsers.eisner_nbest import eisner_dp_nbest, eisner_dp_forest

import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/rescore/scripts'))
from main_test import bert_head_scores
from pprint import pprint
from collections import defaultdict

from tqdm import tqdm

tf.disable_eager_execution()

#***************************************************************
class BaseParser(NN):
  """"""

  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""

    raise NotImplementedError

  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""

    raise NotImplementedError

  #=============================================================
  def sanity_check(self, inputs, targets, predictions, vocabs, fileobject, feed_dict={}):
    """"""

    for tokens, golds, parse_preds, rel_preds in zip(inputs, targets, predictions[0], predictions[1]):
      for l, (token, gold, parse, rel) in enumerate(zip(tokens, golds, parse_preds, rel_preds)):
        if token[0] > 0:
          word = vocabs[0][token[0]]
          glove = vocabs[0].get_embed(token[1])
          tag = vocabs[1][token[2]]
          gold_tag = vocabs[1][gold[0]]
          pred_parse = parse
          pred_rel = vocabs[2][rel]
          gold_parse = gold[1]
          gold_rel = vocabs[2][gold[2]]
          fileobject.write('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t%d\t%s\n' % (l, word, glove, tag, gold_tag, pred_parse, pred_rel, gold_parse, gold_rel))
      fileobject.write('\n')
    return

  #=============================================================
  def validate_cube(self, mb_inputs, mb_targets, mb_probs, is_sparse, rel_vocab):
    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs # mb_parse_probs: [batch, words, words], mb_rel_probs: [batch, words, words, rel]
    for inputs, targets, parse_probs, rel_probs in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      assert parse_probs.shape[0] >= length + 1
      assert parse_probs.shape == rel_probs.shape[:2]
      if parse_probs.shape[0] > length + 1:
        parse_probs = parse_probs[:length+1,:length+1]
        rel_probs = rel_probs[:length+1,:length+1,:]
      final_probs = np.expand_dims(parse_probs, axis=2) * rel_probs # [words, words, rel]
      if is_sparse:
        triples = []
        for mi in range(1, final_probs.shape[0]):
          for hi in range(final_probs.shape[1]):
            for lb in range(final_probs.shape[2]):
              if final_probs[mi,hi,lb] >= 0.01:
                triples.append((float(final_probs[mi,hi,lb]),mi,hi,rel_vocab[lb]))
        assert len(triples) > 0
        sents.append(triples) # [batch, triples, prb-mi-hi-lb]
      else:
        assert False, 'Under construction. We should also output real relation strings!'
        sents.append(final_probs.tolist()) # [batch, words, words, rel]
    return sents

  def validate_nbest(self, mb_inputs, mb_targets, mb_probs, rel_vocab, snts, voc):
    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs, snt in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs, snts):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT) #Vocab.ROOT=1
      length = np.sum(tokens_to_keep)
      '''
      print(snt)
      print(inputs)
      print(tokens_to_keep)
      for w in snt:
        print(voc._str2idx[w.lower()])
      print(length)
      '''
      nbest = [[[float(x[0]), int(x[1]), int(x[2]), rel_vocab[x[-1]],] for x in cur_best] \
              for cur_best in eisner_dp_nbest(length, parse_probs, rel_probs)]
      sents.append(nbest) # [batch, nbest, edges]
    return sents

  def rescoring(self, inputs, words, tags):
    inp_snt = [words._idx2str[inp[0]] for inp in inputs if inp[0]>Vocab.ROOT]
    inp_tags = [tags._idx2str[inp[2]] for inp in inputs if inp[2]>Vocab.ROOT]

    ##root考慮して学習すべき
    print(inp_snt)
    print(inp_tags)
    print('len_inp_snt: '+str(len(inp_snt)))
    rescores = bert_head_scores(inp_snt, inp_tags)
    print(rescores)

    inp_snt.insert(0,'root')
    #rescores = [None]*(len(inp_snt))
    verb_nodes = [bool(tag[0] == 'V') for tag in inp_tags]
    verb_nodes.insert(0,False)
    
    #print(inp_snt)
    #print(verb_nodes)
    verb_idx = [i for i in range(len(verb_nodes)) if verb_nodes[i]]
    #print(verb_idx)
    #print([inp_snt[i] for i in verb_idx])

    rescore_dict = defaultdict(list)
    for i,tok in enumerate(inp_snt):
      if i == 0:
        rescore_dict[i] = [tok, None]
      else:
        rescore_dict[i] = [tok, rescores[i-1]]

    print(rescore_dict)

    return rescore_dict

  def validate_nbest_rescore(self, mb_inputs, mb_targets, mb_probs, rel_vocab, snts, words, tags, RESCORE, NBEST, ALPHA):
    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs, snt in tqdm(zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs, snts)):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT) #Vocab.ROOT=1
      length = np.sum(tokens_to_keep)
      '''
      ['Several', 'traders', 'could', 'be', 'seen', 'shaking', 'their', 'heads', 'when', 'the', 'news', 'flashed', '.']
      13
      #inputs
      [[    1     1     1]
        [  256   204     7]
        [  285  3185     8]
        [   90    97    22]
        [   30    33    14]
        [  932   544    18]
        [ 6763  9177    20]
        [   57    47    24]
        [ 2754  1990     8]
        [   72    64    34]
        [    4     3     6]
        [  331   175     3]
        [18287 18538    13]
        [    5     5    10]
        [    0     0     0]
        [    0     0     0]
        [    0     0     0]]
      #tokens_to_keep
      [False  True  True  True  True  True  True  True  True  True  True  True
      True  True False False False]
      '''
      #print(snt)
      #print(inputs)
      #print(tokens_to_keep)
      #print([words._str2idx[w.lower()] for w in snt])
      #print([words._idx2str[inp[0]] for inp in inputs])
      #print([tags._idx2str[inp[2]] for inp in inputs])
      #print(length)
      #print(rel_probs.shape)

      if RESCORE:

        rescores = self.rescoring(inputs, words, tags)
        #print(rescores)
        '''
        [0.999235987663269, 4, 5, 'nsubj']
        edge_prob, child, head, rel
        '''

        nbest = [[[float(x[0]), int(x[1]), int(x[2]), rel_vocab[x[-1]],] for x in cur_best] \
                for cur_best in eisner_dp_nbest(length, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA)]
        
        sents.append(nbest) # [batch, nbest, edges]

        #pprint(nbest)

        '''
        for tree in nbest:
            print('---')
            for t in tree:
                new_t = list(t)
                new_t[1] = rescores[t[1]][0]
                new_t[2] = rescores[t[2]][0]
                print(new_t)
            print('---')
        '''

      else:
        rescores = None
        nbest = [[[float(x[0]), int(x[1]), int(x[2]), rel_vocab[x[-1]],] for x in cur_best] \
                for cur_best in eisner_dp_nbest(length, parse_probs, rel_probs, rescores, RESCORE, NBEST, ALPHA)]
        
        sents.append(nbest) # [batch, nbest, edges]

    return sents

  def validate_nbest_forest(self, mb_inputs, mb_targets, mb_probs, rel_vocab, snts, words, tags, NBEST):
    forests = []
    parse_probs_list = []
    rel_probs_list = []
    input_tags_list = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs, snt in tqdm(zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs, snts)):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT) #Vocab.ROOT=1
      length = np.sum(tokens_to_keep)

      forest = eisner_dp_forest(length, parse_probs, rel_probs, rel_vocab, NBEST)

      forests.append(forest)
      parse_probs_list.append(parse_probs)
      rel_probs_list.append(rel_probs)

      inp_tags = [tags._idx2str[inp[2]] for inp in inputs if inp[2]>Vocab.ROOT]
      input_tags_list.append(inp_tags)

    return forests, parse_probs_list, rel_probs_list, input_tags_list


  def validate(self, mb_inputs, mb_targets, mb_probs):
    """"""

    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      parse_preds, rel_preds = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep)

      sent = -np.ones( (length, 9), dtype=int)
      tokens = np.arange(1, length+1)
      sent[:,0] = tokens
      sent[:,1:4] = inputs[tokens]
      sent[:,4] = targets[tokens,0]
      sent[:,5] = parse_preds[tokens]
      sent[:,6] = rel_preds[tokens]
      sent[:,7:] = targets[tokens, 1:]
      sents.append(sent)
    return sents

  #=============================================================
  @staticmethod
  def evaluate(filename, punct=NN.PUNCT):
    """"""

    correct = {'UAS': [], 'LAS': []}
    with open(filename) as f:
      for line in f:
        line = line.strip().split('\t')
        if len(line) == 10 and line[4] not in punct:
          correct['UAS'].append(0)
          correct['LAS'].append(0)
          if line[6] == line[8]:
            correct['UAS'][-1] = 1
            if line[7] == line[9]:
              correct['LAS'][-1] = 1
    correct = {k:np.array(v) for k, v in correct.items()}
    return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct

  #=============================================================
  @property
  def input_idxs(self):
    return (0, 1, 2)
  @property
  def target_idxs(self):
    return (3, 4, 5)

  @property
  def forest_idxs(self):
    return (6, 7, 8, 9)
