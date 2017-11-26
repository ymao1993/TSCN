from collections import Counter
from data_manager import DataManager

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_caption_stats(caption_file):
    data_manager = DataManager()
    data_manager.load_captions(caption_file)
    captions = data_manager.raw_captions
    lengths = [len(caption) for caption in captions]
    counter = Counter()
    counter.update(lengths)
    num_captions = len(captions)
    Xs = list([key for key in counter])
    Ys = list([float(counter[key])/num_captions for key in counter])
    cum_val = 0
    cumulative_Ys = []
    for y in Ys:
        cum_val += y
        cumulative_Ys.append(cum_val)
    plt.fill_between(Xs, cumulative_Ys)
    plt.ylabel('Cumulative Frequency')
    plt.xlabel('Caption Length')
    plt.xticks(np.arange(0, 376, 25))
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.xlim([0, 376])
    plt.ylim([0, 1])
    plt.savefig('train_caption_stats.png')


def main():
    generate_caption_stats('/home/mscvproject/yu_code/captions/train.json')

if __name__ == '__main__':
    main()
