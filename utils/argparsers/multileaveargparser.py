# -*- coding: utf-8 -*-

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.argparsers.simulationargparser import SimulationArgumentParser


class MultileaveArgumentParser(SimulationArgumentParser):

    def __init__(self, description=None, set_arguments={}):
        set_arguments['print_feature_count'] = False
        super(MultileaveArgumentParser, self).__init__(description=description,
                                                set_arguments=set_arguments)
        # self.set_argument_namespace('MultileaveArgumentParser')

        # self.add_argument('--bias', dest='bias_experiment', action='store_true', required=False,
        #                   default=False, help='Flag for bias experiment.')

        # self.add_argument('--k --n_results', dest='k', default=10, type=int,
        #                   help='Number of results shown after each query.')

        self.add_argument('--n_rankers', dest='n_rankers', required=True, type=int,
                          help='Number of rankers to use in simulation.')

    # def get_multileave_args(self, args):
    #     return self.get_args(args, 'MultileaveArgumentParser')

    # def parse_args_rec(self):
    #     output_str, args, sim_args = super(MultileaveArgumentParser, self).parse_args_rec()
    #     multileave_args = self.get_multileave_args(args)
    #     if not sim_args.no_run_details:
    #         output_str += '\nMultileave Arguments'
    #         output_str += '\n---------------------'
    #         for name, value in vars(multileave_args).items():
    #             output_str += '\n%s %s' % (name, value)
    #         output_str += '\n---------------------'
    #     return output_str, args, sim_args, multileave_args
