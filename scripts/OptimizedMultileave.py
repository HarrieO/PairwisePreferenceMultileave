import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.multileavesimulation import MultileaveSimulation
from utils.argparsers.multileaveargparser import MultileaveArgumentParser
from multileaving.OptimizedMultileave import OptimizedMultileave

parser = MultileaveArgumentParser()


rankers = []
output_line, args, sim_args = parser.parse_all_args()
ranker_params = {}

rankers.append((output_line,
 'Optimized_Multileave_%drankers' % args.n_rankers, OptimizedMultileave, [],
                       ranker_params))



experiment = MultileaveSimulation(sim_args)

experiment.run(rankers)
