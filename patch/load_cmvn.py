import json
import numpy as np

def _load_json_cmvn(json_cmvn_file):
    """ Load the json format cmvn stats file and calculate cmvn
    Args:
        json_cmvn_file: cmvn stats file in json format
    Returns:
        a numpy array of [means, vars]
    """
    with open(json_cmvn_file) as f:
        cmvn_stats = json.load(f)

    if 'norm' in cmvn_stats:
        assert 'istd_stat' in cmvn_stats
        return np.array([cmvn_stats['mean_stat'], cmvn_stats['istd_stat']])
    means = cmvn_stats['mean_stat']
    variance = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn = np.array([means, variance])
    return cmvn

_load_json_cmvn('../assets/global_cmvn')
