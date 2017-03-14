#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition

logger = logging.getLogger(__name__)

class Timer(object):

    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

class BeamSearch(object):

    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples, ignore_unk=False, minlen=1, getRep = False):
        cdata = self.comp_repr(seq)
        #print len(cdata)
        c = cdata[0]

        forward_rester = cdata[1]
        forward_updater = cdata[2]
        backward_rester = cdata[3]
        backward_updater = cdata[4]

        max_forward_rester = numpy.amax(forward_rester, axis = 1)
        max_backward_rester = numpy.amax(backward_rester, axis = 1)

        max_forward_updater = numpy.amax(forward_updater, axis = 1)
        max_backward_updater = numpy.amax(backward_updater, axis = 1)

        for_retend = []
        back_retend = []

        for_uptend = []
        back_uptend = []


        for i in range(0, max_forward_rester.shape[0]-1):
            for_retend.append(max_forward_rester[i+1]-max_forward_rester[i])

        for i in range(0, max_backward_rester.shape[0]-1):
            back_retend.append(max_backward_rester[i]-max_backward_rester[i+1])
        
        for_retend = numpy.array(for_retend)
        back_retend = numpy.array(back_retend)

        for i in range(0, max_forward_updater.shape[0]-1):
            for_uptend.append(max_forward_updater[i+1]-max_forward_updater[i])

        for i in range(0, max_backward_updater.shape[0]-1):
            back_uptend.append(max_backward_updater[i]-max_backward_updater[i+1])
        
        for_uptend = numpy.array(for_uptend)
        back_uptend = numpy.array(back_uptend)

        print (for_retend+back_retend) / 2.0
        print (for_uptend+back_uptend) / 2.0

        print("----------------------------------------------")
        
        avg_forward_rester = numpy.sum(forward_rester, axis = 1) / forward_rester.shape[1]
        avg_backward_rester = numpy.sum(backward_rester, axis = 1) / backward_rester.shape[1]

        avg_forward_updater = numpy.sum(forward_updater, axis = 1) / forward_updater.shape[1]
        avg_backward_updater = numpy.sum(backward_updater, axis = 1) / backward_updater.shape[1]

        for_retend = []
        back_retend = []

        for_uptend = []
        back_uptend = []


        for i in range(0, avg_forward_rester.shape[0]-1):
            for_retend.append(avg_forward_rester[i+1]-avg_forward_rester[i])

        for i in range(0, avg_backward_rester.shape[0]-1):
            back_retend.append(avg_backward_rester[i]-avg_backward_rester[i+1])
        
        for_retend = numpy.array(for_retend)
        back_retend = numpy.array(back_retend)

        for i in range(0, avg_forward_updater.shape[0]-1):
            for_uptend.append(avg_forward_updater[i+1]-avg_forward_updater[i])

        for i in range(0, avg_backward_updater.shape[0]-1):
            back_uptend.append(avg_backward_updater[i]-avg_backward_updater[i+1])
        
        for_uptend = numpy.array(for_uptend)
        back_uptend = numpy.array(back_uptend)

        print (for_retend+back_retend) / 2.0
        print (for_uptend+back_uptend) / 2.0



        if getRep:
            return c




        # print "c shape is %s " % (str(c.shape))
        states = map(lambda x : x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]

        num_levels = len(states)

        fin_trans = []
        fin_costs = []
        fin_align = []

        trans = [[]]
        costs = [0.0]

        dec_rester = [[]]*n_samples
        dec_updater = [[]]*n_samples

        fin_dec_rester = []
        fin_dec_updater = []

        align = []
        for i in range(n_samples):
            align.append(numpy.array([numpy.zeros(len(seq))]))

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))

            ans = self.comp_next_probs(c, k, last_words, *states)

            probs = ans[0]
            alignments = ans[1]
            log_probs = numpy.log(probs)

            trester = ans[2]
            tupdater = ans[3]

            trester = numpy.sum(trester, axis = 1) / trester.shape[1]
            tupdater = numpy.sum(tupdater, axis = 1) / tupdater.shape[1]

            #print "___________________"
            #print tupdater



            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,self.unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:,self.eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]
            #print best_costs_indices

            # Form a beam for the next iteration

            new_rester = [[]] * n_samples
            new_updater = [[]] * n_samples
            new_align = [[]]*n_samples

            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost

                #dec_rester[i] = dec_rester[i] + [trester[orig_idx]]
                #print orig_idx
                #align[i] = numpy.concatenate((align[i] , [alignments[:,orig_idx]]), axis=0)
                new_align[i] = numpy.concatenate((align[orig_idx] , [alignments[:,orig_idx]]), axis=0)
                new_rester[i] = dec_rester[orig_idx] + [trester[orig_idx]]
                new_updater[i] = dec_updater[orig_idx] + [tupdater[orig_idx]]

                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, k, inputs, *new_states)

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            align = []
            dec_rester = []
            dec_updater = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    align.append(new_align[i])
                    dec_rester.append(new_rester[i])
                    dec_updater.append(new_updater[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
                    fin_align.append(new_align[i])
                    fin_dec_rester.append(new_rester[i])
                    fin_dec_updater.append(new_updater[i])
            states = map(lambda x : x[indices], new_states)


        

        for i in range(len(fin_align)): 
            talign = fin_align[i]
            fin_align[i] = talign[1:,:]

        #print fin_align

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, False, minlen)
            elif n_samples < 500:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, False, minlen)
            else:
                logger.error("Translation failed")

        tfin_align = []
        index = numpy.argsort(fin_costs)

        for i in range(0, len(index)):
            tfin_align.append(fin_align[index[i]])

        fin_dec_rester = numpy.array(fin_dec_rester)[numpy.argsort(fin_costs)]
        fin_dec_updater = numpy.array(fin_dec_updater)[numpy.argsort(fin_costs)]

        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        return fin_trans, fin_costs, tfin_align, fin_dec_rester, fin_dec_updater

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen


class SampleBlock(object):
    '''
    class for sampling
    '''

    def __init__(self):
        # para setting
        self.arg_state = 'search_state.pkl'
        self.arg_changes = ""
        self.arg_model_path = 'search_model.npz'
        self.arg_beam_search = True
        self.arg_ignore_unk = False
        self.arg_normalize = False


        self.state = prototype_state() 
        with open(self.arg_state) as src:
            self.state.update(cPickle.load(src))
        self.state.update(eval("dict({})".format(self.arg_changes)))

        logging.basicConfig(level=getattr(logging, self.state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
        
        rng = numpy.random.RandomState(self.state['seed'])
        self.enc_dec = RNNEncoderDecoder(self.state, rng, skip_init=True)
        self.enc_dec.build()
        self.lm_model = self.enc_dec.create_lm_model()
        self.lm_model.load(self.arg_model_path)
        self.indx_word = cPickle.load(open(self.state['word_indx'],'rb'))

        self.beam_search = None
        self.beam_search = BeamSearch(self.enc_dec)
        self.beam_search.compile()


        self.idict_src = cPickle.load(open(self.state['indx_word'],'r'))

    '''
    seqin: input sentence, k sample number
    return a list of tuple(sentence, score) 
    '''
    def getSamples(self, seqori, k):
        # split the sentence
        seqin = ""
        for i in range(0, len(seqori), 3):
            w = seqori[i:i+3]
            seqin = seqin + w + " "

        print "split seq:#%s#" % (seqin)
        #return

        seq,parsed_in = parse_input(self.state, self.indx_word, seqin, idx2word=self.idict_src)
        
        ans, align, rester, updater = self.sample(seq, k)

        return ans, align, rester, updater

    def sample(self, seq, n_samples):

        ans = []
        trans, costs, align, rester, updater  = self.beam_search.search(seq, n_samples, ignore_unk=self.arg_ignore_unk, minlen=len(seq) / 2)
        if self.arg_normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(self.lm_model.word_indxs, trans[i])
            ans.append((" ".join(sen),  costs[i]))
        return ans, align, rester, updater

    def getRep(self, seqori):
        seqin = ""
        for i in range(0, len(seqori), 3):
            w = seqori[i:i+3]
            seqin = seqin + w + " "

        print "split seq:#%s#" % (seqin)

        seq,parsed_in = parse_input(self.state, self.indx_word, seqin, idx2word=self.idict_src)
        rep = self.beam_search.search(seq, 20, ignore_unk=self.arg_ignore_unk, minlen=len(seq) / 2, getRep = True)

        return rep


