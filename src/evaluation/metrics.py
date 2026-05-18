"""Statistical metrics: MCC, Cohen d, bootstrap CI, paired t-test."""
import numpy as np
from scipy import stats
from sklearn.metrics import matthews_corrcoef

def compute_mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def paired_t_test(scores_a, scores_b):
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    se = np.std(np.array(scores_a) - np.array(scores_b), ddof=1) / np.sqrt(len(scores_a))
    return {"t": t_stat, "p": p_value, "se": se}

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi
