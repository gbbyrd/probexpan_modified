import numpy as np
from tqdm import tqdm
import json
import mmap
import scipy.stats
import copy

# adding JAX functionality
import jax
import jax.numpy as jnp

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def load_vocab(filename):
    eid2name = {}
    keywords = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[1])
            eid2name[eid] = temp[0]
            keywords.append(eid)
    eid2idx = {w: i for i, w in enumerate(keywords)}
    print(f'Vocabulary: {len(keywords)} keywords loaded')
    return eid2name, keywords, eid2idx


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    num_hits = 0.0
    score = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def geometric_mean(data):
    total = 1
    for i in data:
        total *= i
    return pow(total, 1/len(data))

def KL_divergence(p, q):
    return scipy.stats.entropy(p, q)

# modified function to add jax functionality to utilize GPU
def standardization(data):
    mu = jnp.mean(data, axis=0)
    sigma = jnp.std(data, axis=0)
    return (data - mu) / sigma

# modified function to add jax functionality to utilize GPU
def normalization(data):
    _range = jnp.max(data) - jnp.min(data)
    return (data - jnp.min(data)) / _range
