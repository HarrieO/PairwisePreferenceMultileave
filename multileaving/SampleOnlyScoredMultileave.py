import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.rankings as rnk
from  multileaving.ProbabilisticMultileave import ProbabilisticMultileave

class SampleOnlyScoredMultileave(ProbabilisticMultileave):

    def __init__(self, *args, **kwargs):
        ProbabilisticMultileave.__init__(self, *args, **kwargs)
        self._name = 'Sample Only Scored Multileave'
        self.needs_descending = True

    def next_index_to_add(self, inter_result, inter_n, ranking, index):
        while index < ranking.shape[0] and np.any(ranking[index] == inter_result[:inter_n]):
            index += 1
        return index

    def make_multileaving(self, descending_rankings, inverted_rankings):
        self._last_inverted_rankings = inverted_rankings
        self._last_n_rankers = inverted_rankings.shape[0]

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

        return multileaved

    def infer_preferences(self, result_list, clicked_docs):
        if np.any(clicked_docs):
            return self.preferences_of_list(result_list, self._last_inverted_rankings, clicked_docs.astype(bool))
        else:
            return np.zeros((self._last_n_rankers, self._last_n_rankers))

    def preferences_of_list(self, result_list, doc_scores, clicks):
        '''
        ARGS: (all np.array of docids)
        - result_list: the multileaved list
        - doc_scores: matrix (rankers x documents) where [x,y] corresponds to the score of doc y in ranker x
                      ranking is done descendingly in score
        - clicks: boolean array with clicked documents (lenght: multileaved.length)
        - tau: tau used in softmax-functions for rankers

        RETURNS:
        - preference matrix: matrix (rankers x rankers) in this matrix [x,y] > 0 means x won over y and [x,y] < 0 means x lost from y
          the value is analogous to the (average) degree of preference
        '''
        # n = result_list.shape[0]
        # # normalization denominator for the complete ranking
        # sigmoid_total = np.sum(float(1) / (np.arange(n) + 1) ** tau)

        sample_ranking = rnk.rank_query(doc_scores[:,result_list],inverted=True)

        sigmas = 1./(sample_ranking[:,clicks]+1.)**self._tau

        scores = np.sum(sigmas,axis=1)

        return np.sign(scores[:,None] - scores[None,:])

