# -*- coding: utf-8 -*-

import numpy as np


class ProbabilisticMultileave(object):

    def __init__(self, num_data_features, n_samples=10000, k=10, tau=3.0):
        self._name = 'Probabilistic Multileave'
        self._k = k
        self._tau = tau
        self._n_samples = n_samples
        self.uses_inverted_rankings = True
        self.needs_inverted = True
        self.needs_descending = False
        self.needs_oracle = False
        self.vector_aggregation = False

    def clean(self):
        del self._last_inverted_rankings

    def make_multileaving(self, inverted_rankings):
        '''
        ARGS: (all np.array of docids)
        - inverted_rankings: matrix (rankers x documents) where [x,y] corresponds to the rank of doc y in ranker x

        RETURNS
        - ranking of indices corresponding to inverted_rankings
        '''

        self._last_inverted_rankings = inverted_rankings
        self._last_n_rankers = inverted_rankings.shape[0]
        n = inverted_rankings.shape[1]
        k = min(n, self._k)

        assignments = np.random.randint(0, inverted_rankings.shape[0], k)

        denominator = np.zeros(k) + np.sum(float(1) / (np.arange(n) + 1) ** self._tau)
        probs = 1. / (inverted_rankings[assignments, :] + 1) ** self._tau

        ranking = np.zeros(k, dtype=np.int32)

        docids = np.arange(n)

        for i in range(k):
            upper = np.cumsum(probs[i, :])
            lower = np.zeros(upper.shape)
            lower[1:] += upper[:-1]

            coinflip = np.random.rand()

            logic = np.logical_and(lower / denominator[i] < coinflip, upper / denominator[i]
                                   >= coinflip)

            raw_i = np.where(logic)[0][0]

            ranking[i] = docids[raw_i]

            docids[raw_i:-1] = docids[raw_i + 1:]

            denominator -= probs[:, raw_i]
            if raw_i < n - 1:
                probs[:, raw_i:-1] = probs[:, raw_i + 1:]

        return ranking

    def infer_preferences(self, result_list, clicked_docs):
        if np.any(clicked_docs):
            return self.preferences_of_list(self.probability_of_list(result_list,
                                            self._last_inverted_rankings,
                                            clicked_docs.astype(bool), self._tau), self._n_samples)
        else:
            return np.zeros((self._last_n_rankers, self._last_n_rankers))

    def probability_of_list(self, result_list, inverted_rankings, clicked_docs, tau):
        '''
        ARGS: (all np.array of docids)
        - result_list: the multileaved list
        - inverted_rankings: matrix (rankers x documents) where [x,y] corresponds to the rank of doc y in ranker x
        - clicked_docs: boolean array of result_list length indicating clicks

        RETURNS
        -sigmas: matrix (rankers x clicked_docs) with probabilty ranker added clicked doc
        '''
        n_docs = inverted_rankings.shape[1]
        n_rankers = inverted_rankings.shape[0]

        click_doc_ind = result_list[clicked_docs]

        # normalization denominator for the complete ranking
        sigmoid_total = np.sum(float(1) / (np.arange(n_docs) + 1) ** self._tau)

        
        # cumsum is used to renormalize the probs, it contains the part
        # the denominator that has to be removed due to previously added docs
        cumsum = np.zeros((n_rankers, result_list.shape[0]))
        cumsum[:, 1:] = np.cumsum(float(1) / (inverted_rankings[:, result_list[:-1]] + 1.)
                                  ** self._tau, axis=1)

        # make sure inverted rankings is of dtype float
        sigmas = 1 / (inverted_rankings[:, click_doc_ind].T + 1.) ** self._tau
        sigmas /= sigmoid_total - cumsum[:, clicked_docs].T

        return sigmas / np.sum(sigmas, axis=1)[:, None]

    def preferences_of_list(self, probs, n_samples):
        '''
        ARGS:
        -probs: clicked docs x rankers matrix with probabilities ranker added clicked doc  (use probability_of_list)
        -n_samples: number of samples to base preference matrix on

        RETURNS:
        - preference matrix: matrix (rankers x rankers) in this matrix [x,y] > 0 means x won over y and [x,y] < 0 means x lost from y
          the value is analogous to the (average) degree of preference
        '''

        n_clicks = probs.shape[0]
        n_rankers = probs.shape[1]
        # determine upper bounds for each ranker (to see prob distribution as set of ranges)
        upper = np.cumsum(probs, axis=1)

        # determine lower bounds
        lower = np.zeros(probs.shape)
        # lower[:,0] = 0
        lower[:, 1:] += upper[:, :-1]

        # flip coins, coins fall between lower and upper
        coinflips = np.random.rand(n_clicks, self._n_samples)
        # make copies for each sample and each ranker
        comps = coinflips[:, :, None]
        # determine where each coin landed
        log_assign = np.logical_and(comps > lower[:, None, :], comps <= upper[:, None, :])
        # click count per ranker (samples x rankers)
        click_count = np.sum(log_assign, axis=0)
        # the preference matrix for each sample
        prefs = np.sign(click_count[:, :, None] - click_count[:, None, :])

        # the preferences are averaged for each pair
        # in this matrix [x,y] > 0 means x won over y and [x,y] < 0 means x lost from y
        # the value is analogous to the (average) degree of preference
        return np.sum(prefs, axis=0) / float(self._n_samples)
