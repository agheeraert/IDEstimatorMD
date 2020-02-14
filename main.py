import argparse
import os
from IDEstimatorMD import IDEstimatorMD

parser = argparse.ArgumentParser(description='Plot the Estimated ID for MD trajectories')
parser.add_argument('tr',  type=str, nargs='+',
                    help='Input trajectories')
parser.add_argument('top',  type=str,
                    help='Input topology')
parser.add_argument('o',  type=str,
                    help='Output folder')  
parser.add_argument('-m',  type=str, nargs='+', default=["RMSD"],
                    help='List of methods used to compute distances')
parser.add_argument('-s',  type=str, default="",
                    help='Output string to characterize the plot')
parser.add_argument('-d',  type=bool, default=True,
                    help='Toggle if tail should be discarded or not')
parser.add_argument('-csplit',  type=int, default=None, nargs='+',
                    help='Precise index of end of chains to compute the distances only on specific chains')
parser.add_argument('-clabel',  type=str, default=None, nargs='+',
                    help='To label each chain')
# parser.add_argument('-second_order',  type=bool, default=False,
#                     help='Toggle if second order fit')                


args = parser.parse_args()

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
mkdir(args.o)

IDEstimatorMD(args.tr, args.top, args.m, args.o, args.s, args.d, args.csplit, args.clabel).plot_ID()
