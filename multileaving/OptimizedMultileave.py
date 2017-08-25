# -*- coding: utf-8 -*-

import numpy as np
import time
import gurobipy

class OptimizedMultileave(object):

    def __init__(self, num_data_features, k=10):
        self._name = 'Pairwise Preferences Multileave'
        self._k = k
        self.needs_inverted = True
        self.needs_descending = True
        self.needs_oracle = False
        self.vector_aggregation = False

    def clean(self):
        pass

    def create_possible_lists(self, descending_rankings, n_samples=10):
        n_rankers = descending_rankings.shape[0]
        n_docs = descending_rankings.shape[1]
        top_docs = descending_rankings[:,:10]
        allowed_leavings = {}
        length = min(self._k, n_docs)
        for _ in range(n_samples):
            available = np.ones((n_rankers, length), dtype=bool)
            sampled_list = np.empty(length, dtype=np.int32)
            assign = np.random.randint(0, n_rankers, self._k)
            sampled_list[0] = descending_rankings[assign[0],0]
            available[top_docs == sampled_list[0]] = False
            for i in range(1,length):
                to_add = np.where(available[assign[i],:])[0][0]
                sampled_list[i] = top_docs[assign[i], to_add]
                available[top_docs == sampled_list[i]] = False
            hashable = tuple(sampled_list)
            allowed_leavings[hashable] = sampled_list
        self._allowed_leavings = allowed_leavings.values()

    def make_multileaving(self, descending_rankings, inverted_rankings):
        self.create_possible_lists(descending_rankings)

        # L x k x r
        C = []
        for ml in self._allowed_leavings:
            C.append(1./(inverted_rankings[:,ml].T+1))

        n_docs = descending_rankings.shape[1]
        n_rankers = descending_rankings.shape[0]

        length = min(self._k,n_docs)

        m = gurobipy.Model("system")
        m.params.outputFlag = 0
        P = []
        # Add a parameter Pi for each list that adheres to equation (6)
        for i in range(len(self._allowed_leavings)):
            P.append(m.addVar(lb=0.0, ub=1.0, name='p%d' % i))
        m.update()
        m.addConstr(gurobipy.quicksum(P) == 1, 'sum')
        biasconstrs = []

        V = []
        for k in range(length):
            V.append(m.addVar(name='var%d' % k))
        m.update()
        # Constraint for equation (7)
        # Constraints for equation(8) for each k
        for k in range(length):
            for x in range(n_rankers):
                s = []
                for i in range(len(self._allowed_leavings)):
                    s.append(P[i] * gurobipy.quicksum(
                                        [C[i][j,x] for j in range(k)]))
                biasconstrs.append(
                    m.addConstr(gurobipy.quicksum(s) == V[k], "c%d" % k))
        
        # Add sensitivity as an objective to the optimization, equation (13)
        S = []
        for i in range(len(self._allowed_leavings)):
            # Replacing Equation (9, 10, 11)
            s = []

            mu = 0.0
            for x in range(n_rankers):
                for j in range(length):
                    mu += self.f(j + 1) * C[i][j,x]
            mu /= len(self._allowed_leavings)
            for x in range(n_rankers):
                s.append((sum([
                          self.f(j + 1) * C[i][j,x]
                          for j in range(length)]) - mu) ** 2
                         )
            

            S.append(P[i] * sum(s))

        m.setObjective(gurobipy.quicksum(S), gurobipy.GRB.MINIMIZE)

        # Optimize the system and if it is infeasible, relax the constraints
        self.relaxed = False
        m.optimize()
        if m.status != gurobipy.GRB.GENCONSTR_ABS:
            self.relaxed = True
            m.feasRelaxS(1, False, True, True)
            m.optimize()

        if m.status != gurobipy.GRB.GENCONSTR_ABS:
            choice = np.random.choice(np.arange(len(self._allowed_leavings)))
            self._credits = np.zeros(C[0].shape)
            return self._allowed_leavings[choice]

        probs = np.array([P[i].x for i in range(len(self._allowed_leavings))])
        probs[probs < 0] = 0

        if np.all(probs==0):
            choice = np.random.choice(np.arange(len(self._allowed_leavings)))
            self._credits = np.zeros(C[0].shape)
        else:
            choice = np.random.choice(np.arange(len(self._allowed_leavings)), p=probs/np.sum(probs))
        
        self._credits = C[choice]

        return self._allowed_leavings[choice]

    def infer_preferences(self, result_list, clicked_docs):
        if np.any(clicked_docs):
            creds = np.sum(self._credits[clicked_docs.astype(bool),:],axis=0)
            return creds[:,None] - creds[None,:]
        else:
            return np.zeros((self._last_n_rankers, self._last_n_rankers))

    def f(self, i):
        # Implemented as footnote 4 suggests
        return 1. / i