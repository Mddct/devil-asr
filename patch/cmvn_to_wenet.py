import json
import sys
from typing import Tuple, List


def load_paraformer_cmvn(cmvn_file) -> Tuple[List, List]:
    with open(cmvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                means_list = list(map(float, list(add_shift_line)))
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                vars_list = list(map(float, list(rescale_line)))
                continue
    return means_list, vars_list


def to_wenet_cmvn(cmvn_file):
    means, istd = load_paraformer_cmvn(cmvn_file)

    for i in range(len(means)):
        # paraformer mean is negative
        means[i] = means[i] * (-1)

    d = {}
    d['mean_stat'] = means
    d['istd_stat'] = istd
    d['frame_num'] = 1
    d['norm'] = True

    return json.dumps(d)


print(to_wenet_cmvn(sys.argv[1]))
