import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import csv


def main(args):
    sns.set()
    for i, path in enumerate(args.path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            steps = []
            values = []
            for row in reader:
                steps.append(row[0])
                values.append(row[1])
        if args.label is None:
            label = path
        else:
            label = args.label[i][0]
        plt.plot(np.array(steps), np.array(values), label=label)

    if not args.hide_legend:
        plt.legend()

    if args.save is None:
        plt.show()
    else:
        plt.savefig(args.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, help='file name to save figure')
    parser.add_argument('--hide-legend', action='store_true')
    parser.add_argument('--label', nargs='*', action='append',
                        help='labels of plots')
    parser.add_argument('path', nargs='+', help='path to csv files')
    args = parser.parse_args()
    main(args)
