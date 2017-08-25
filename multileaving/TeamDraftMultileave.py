# -*- coding: utf-8 -*-

import numpy as np


class TeamDraftMultileave(object):

    def __init__(self, num_data_features, k=10):
        self._name = 'Probabilistic Multileave'
        self._k = k
        self.uses_inverted_rankings = False
        self.needs_inverted = False
        self.needs_descending = True
        self.needs_oracle = False
        self.vector_aggregation = False

    def clean(self):
        del self.teams

    def next_index_to_add(self, inter_result, inter_n, ranking, index):
        while index < ranking.shape[0] and np.any(ranking[index] == inter_result[:inter_n]):
            index += 1
        return index

    def make_multileaving(self, descending_rankings):

        rankings = descending_rankings

        n_rankings = rankings.shape[0]
        k = min(self._k,rankings.shape[1])
        teams = np.zeros(k,dtype=np.int32)
        multileaved = np.zeros(k,dtype=np.int32)

        multi_i = 0
        while multi_i < k and np.all(rankings[1:,multi_i]==rankings[0,multi_i]):
            multileaved[multi_i] = rankings[0][multi_i]
            teams[multi_i] = -1
            multi_i += 1

        indices  = np.zeros(n_rankings, dtype=np.int32) + multi_i
        assign_i = n_rankings
        while multi_i < k:
            if assign_i == n_rankings:
                assignment = np.arange(n_rankings)
                np.random.shuffle(assignment)
                assign_i = 0

            rank_i = assignment[assign_i]
            indices[rank_i] = self.next_index_to_add(multileaved, multi_i, rankings[rank_i,:], indices[rank_i])
            multileaved[multi_i] = rankings[rank_i,indices[rank_i]]
            teams[multi_i] = rank_i
            indices[rank_i] += 1
            multi_i += 1
            assign_i += 1

        self.teams = teams
        self.n_rankers = n_rankings
        return multileaved

    def infer_preferences(self, result_list, clicked_docs):
        clicked_docs = clicked_docs.astype(bool)
        assigned_clicks = np.sum(np.arange(self.n_rankers)[:,None]==self.teams[clicked_docs][None,:],axis=1)
        return np.sign(assigned_clicks[:,None] - assigned_clicks[None,:])
