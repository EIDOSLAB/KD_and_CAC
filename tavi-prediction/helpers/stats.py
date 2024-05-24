import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def get_config(file_name):
    with open(file_name) as log_file:
        for line in log_file:
            return line # first line.

def get_line_stats(stats, match, fold_type, stats_to_include):
    if match:
        fold = match.group(1)
        epoch = match.group(2)
        stat_name = match.group(3)
        value = match.group(4)

        if stat_name in stats_to_include or len(stats_to_include) == 0:
            if fold not in stats:
                stats[fold] = dict()

            if stat_name not in stats[fold]:
                stats[fold][stat_name] = dict()

            if fold_type not in stats[fold][stat_name]:
                stats[fold][stat_name][fold_type] = []

            stats[fold][stat_name][fold_type].append(float(value))
        
def get_train_stats(file_name, stats_to_include=[]):
    with open(file_name) as log_file:
        stats = dict()
        for line in log_file:
            train_match = re.search(r'Fold ([0-9]+) Epoch ([0-9]+).*Train stats ([a-zA-Z_]+): ([0-9\.e-]+)', line)
            get_line_stats(stats, train_match, 'Train', stats_to_include)

            valid_match = re.search(r'Fold ([0-9]+) Epoch ([0-9]+).*Valid stats ([a-zA-Z_]+): ([0-9\.e-]+)', line)
            get_line_stats(stats, valid_match, 'Valid', stats_to_include)

            all_match = re.search(r'(ALL) Epoch ([0-9]+).*Train stats ([a-zA-Z_]+): ([0-9\.e-]+)', line)
            get_line_stats(stats, all_match, 'Train', stats_to_include)
    return stats
        
def draw_stats(stats, figsize=(12,7)):
    for fold, fold_stats in stats.items():
        fig = plt.figure(figsize=figsize)
        items = len(fold_stats.keys())
        cols = 3
        rows = int(math.ceil(items / cols))
        i = 1
        for stat_name, fold_types in fold_stats.items():
            subplot = fig.add_subplot(rows, cols, i)
            i += 1
            max_y = 1.02
            min_y = 0
            first = True
            for fold_type, values in fold_types.items():
                x = range(1, len(values)+1)
                y = [float(v) for v in values]
                
#                 colors = ['tab:blue', 'tab:orange']
#                 poly1d_fn = np.poly1d(np.polyfit(x,y,1))
#                 plt.plot(x, y, '.', x, poly1d_fn(x), '--', label=fold_type, c=colors[0 if first else 1])
#                 first = False
                
                plt.plot(x, y, marker='.', label=fold_type)
                max_y = max(y + [max_y])
                min_y = min(y + [min_y])
            plt.ylim(min_y, max_y)
            plt.xlim(0, len(values)+1)
            plt.legend()
            plt.grid(visible=True)
            plt.title(f'Fold {fold} {stat_name}')
        plt.show()

def get_test_stats(file_name, stats_to_include=[]):
    with open(file_name) as log_file:
        stats = dict()
        for line in log_file:
            match = re.search(r'\[TEST (.*)\] Stats ([a-zA-Z_]+): ([0-9\.e-]+)', line)
            if match:
                model = match.group(1)
                stat_name = match.group(2)
                value = match.group(3)
                
                if stat_name in stats_to_include or len(stats_to_include) == 0:
                    if model not in stats:
                        stats[model] = dict()

                    stats[model][stat_name] = float(value)
    return stats

def get_network(file_name):
    net_lines = []
    with open(file_name) as log_file:
        reading_net = False
        first_using = False
        for line in log_file:
            match = re.search(' Using ', line)
            if match:
                if not first_using:
                    first_using = True
                else:
                    reading_net = True
            else:
                match = re.search('Starting Epoch', line)
                if match:
                    reading_net = False
                    break
                
            if reading_net:
                net_lines.append(line)
    return net_lines
                