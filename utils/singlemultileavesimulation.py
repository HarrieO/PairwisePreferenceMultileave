# -*- coding: utf-8 -*-

import sys
import os
import argparse
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils.clicks as clicks
from utils.folddata import get_fold_data
from utils.rankings import get_score_rankings, get_candidate_score_ranking, rank_candidate_queries
from utils.evaluate import get_idcg_list, evaluate, evaluate_ranking


class SingleMultileaveSimulation(object):

    def __init__(self, sim_args, output_queue, click_model, datafold):

        self.click_model = click_model
        self.bias_experiment = self.click_model.get_name()[:6] == "random"

        self.datafold = datafold

        self.k = sim_args.k
        self.n_runs = sim_args.n_runs
        self.n_impressions = sim_args.n_impressions

        self.feature_count = self.datafold.num_features
        self.n_train_queries = self.datafold.train_doclist_ranges.shape[0] - 1
        self.n_test_queries = self.datafold.test_doclist_ranges.shape[0] - 1
        self.output_list = ['DATA FOLDER ' + str(self.datafold.data_path)]
        self.output_list += ['HELDOUT DATA ' + str(self.datafold.heldout_tag)]
        self.output_list += ['CLICK MODEL ' + self.click_model.get_name()]
        self.output_queue = output_queue

        self.rankers = []
        self.weights = None
        self.n_rankers = sim_args.n_rankers

        self.run_index = 0

        self.start_lines = sim_args.print_start
        self.print_frequency = sim_args.print_freq

    def prepare_rankers(self, multileave_method):
        self.train_idcg_vector = get_idcg_list(self.datafold.train_label_vector,
                                               self.datafold.train_doclist_ranges, self.k)
        self.test_idcg_vector = get_idcg_list(self.datafold.test_label_vector,
                                              self.datafold.test_doclist_ranges, self.k)
        
        mul_feat = self.datafold.get_multileave_feat()
        n_mul_feat = len(mul_feat)
        assert self.n_rankers <= len(mul_feat)
        selection = mul_feat[np.random.permutation(n_mul_feat)[:self.n_rankers]]
        self.weights = np.zeros((self.n_rankers, self.feature_count))
        self.weights[np.arange(self.n_rankers), selection] = 1.
        self.train_descending, self.train_inverted = rank_candidate_queries(self.weights,
                self.datafold.train_feature_matrix, self.datafold.train_doclist_ranges,
                inverted=multileave_method.needs_inverted)

    def shuffle_rankers(self, shuffling):
        self.weights = self.weights[shuffling, :]

    def get_test_rankings(self, ranker_model, feat_matrix, qptr):
        return self.get_rankings(ranker_model, feat_matrix, qptr)

    def get_rankings(self, ranker_model, feat_matrix, qptr):
        return get_score_rankings(ranker_model, feat_matrix, qptr, inverted=True)

    def compute_error(self, cur_pref, real_pref):
        return np.sum(np.abs(real_pref - cur_pref)) / self.n_rankers / (self.n_rankers - 1) / 2.0

    def compute_error_bin(self, cur_pref, real_pref_bin, epsilon):
        round_error = np.sign(cur_pref)
        round_error[np.abs(cur_pref) < epsilon] = 0
        return np.sum(real_pref_bin != round_error) / float(self.n_rankers) / (self.n_rankers - 1)

    def print_error(self, infer_pref, real_pref, real_pref_bin):
        error = self.compute_error(infer_pref, real_pref)
        error_str = 'ABS: %.03f' % (error)
        error_bin = self.compute_error_bin(infer_pref, real_pref_bin, 0)
        error_str += ' BIN: %.04f' % error_bin
        return error_str

    def make_multileaving(self, multileave_method, qid):
        rankings = {}
        start_i = self.datafold.train_doclist_ranges[qid]
        end_i = self.datafold.train_doclist_ranges[qid + 1]
        if multileave_method.needs_inverted:
            rankings['inverted_rankings'] = self.train_inverted[:,start_i:end_i]
        if multileave_method.needs_descending:
            rankings['descending_rankings'] = self.train_descending[:,start_i:end_i]
        if multileave_method.needs_oracle:
            rankings['document_labels'] = self.datafold.train_label_vector[start_i:end_i]
        return multileave_method.make_multileaving(**rankings)

    def infer_preferences(self, multileave_method, clicks):
        raise NotImplementedError('Multileaving method not implemented in experiment.')

    def run(self, multileave_method, direct_print=False, output_key=None):
        starttime = time.time()
        if direct_print:
            for line in self.output_list:
                print line

        starting_prints = self.start_lines
        print_counter = 1

        self.prepare_rankers(multileave_method)
        if self.bias_experiment:
            real_pref = np.zeros((self.n_rankers, self.n_rankers))
            real_pref_bin = np.zeros((self.n_rankers, self.n_rankers))
        else:
            self.test_ndcgs = np.zeros(self.n_rankers)
            for i in range(self.n_rankers):
                test_rankings = self.get_test_rankings(self.weights[i, :],
                        self.datafold.test_feature_matrix, self.datafold.test_doclist_ranges)
                self.test_ndcgs[i] = evaluate(test_rankings, self.datafold.test_label_vector,
                                              self.test_idcg_vector, self.n_test_queries, self.k)
            real_pref = self.test_ndcgs[:, None] - self.test_ndcgs[None, :]
            real_pref_bin = np.sign(real_pref)

        total_pref = np.zeros((self.n_rankers, self.n_rankers))
        if multileave_method.vector_aggregation:
            vector_pref = np.zeros(self.n_rankers)

        error_str = "0 TRAIN: 0 "
        error_str += self.print_error(total_pref, real_pref, real_pref_bin)
        self.output_list.append(error_str)
        if direct_print:
            print error_str

        impressions = 0
        pref_update = 0
        for step_i in range(self.n_impressions):
            qid = np.random.randint(0, self.n_train_queries)

            start_i = self.datafold.train_doclist_ranges[qid]
            end_i = self.datafold.train_doclist_ranges[qid + 1]
            n_query_docs = end_i - start_i
            query_labels = self.datafold.train_label_vector[start_i:end_i]

            multileaving = self.make_multileaving(multileave_method, qid)

            cur_clicks = self.click_model.generate_clicks(multileaving, query_labels)

            if np.any(cur_clicks):
                # pref_w, pref =
                pref = multileave_method.infer_preferences(multileaving, cur_clicks)
                if multileave_method.vector_aggregation:
                    vector_pref += pref
                    total_pref[:,:] = (vector_pref[:,None] - vector_pref[None,:])
                else:
                    total_pref += pref
                pref_update += 1

            if print_counter == self.print_frequency:
                error_str = "%d TRAIN: %.02f " % (step_i+1, np.mean(evaluate_ranking(multileaving,
                            self.datafold.train_label_vector[start_i:end_i],
                            self.train_idcg_vector[start_i], self.k)))

                error_str += self.print_error(total_pref / (step_i + 1), real_pref, real_pref_bin)
                self.output_list.append(error_str)
                if direct_print:
                    print error_str

            if print_counter >= self.print_frequency:
                print_counter = 0
            print_counter += 1

        total_time = time.time() - starttime
        seconds = total_time % 60
        minutes = total_time / 60 % 60
        hours = total_time / 3600
        end_string = 'END TIME %d SECONDS %02d:%02d:%02d' % (total_time, hours, minutes, seconds)
        self.output_list.append(end_string)
        if direct_print:
            print end_string

        if output_key is None:
            self.output_queue.put(self.output_list)
        else:
            self.output_queue.put((output_key, self.output_list))


        del self.train_idcg_vector
        del self.test_idcg_vector
        del self.train_descending
        del self.train_inverted
        multileave_method.clean()
