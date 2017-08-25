import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.multileavesimulation import MultileaveSimulation
from utils.argparsers.multileaveargparser import MultileaveArgumentParser
from multileaving.SampleOnlyScoredMultileave import SampleOnlyScoredMultileave
from multileaving.ProbabilisticMultileave import ProbabilisticMultileave
from multileaving.TeamDraftMultileave import TeamDraftMultileave
from multileaving.PairwisePreferenceMultileave import PairwisePreferenceMultileave



import utils.rankings as rnk

parser = MultileaveArgumentParser()

rankers = []
output_line, args, sim_args = parser.parse_all_args()
ranker_params = {}

ranker_params = {'n_samples': 10000, 'tau':3.0}
rankers.append((output_line, 'SOS_Multileave_%drankers' % args.n_rankers, SampleOnlyScoredMultileave, [],
                       ranker_params))
rankers.append((output_line, 'Probabilistic_Multileave_%drankers' % args.n_rankers, ProbabilisticMultileave, [],
                       ranker_params))

ranker_params = {}
rankers.append((output_line, 'TeamDraft_Multileave_%drankers' % args.n_rankers, TeamDraftMultileave, [],
                       ranker_params))

rankers.append((output_line, 'PairwisePreference_Multileave%drankers' % args.n_rankers, PairwisePreferenceMultileave, [],
                       ranker_params))

experiment = MultileaveSimulation(sim_args)

experiment.run(rankers)
