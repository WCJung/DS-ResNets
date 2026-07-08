import random

import numpy as np
import torch


def init_random(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def softmax(d, axis=-1):
    """수치 안정 softmax (max-shift). 관측 공간 변환에는 torch.softmax를 쓰고,
    numpy 배열이 필요한 곳에서만 사용."""
    d = np.asarray(d, dtype=np.float64)
    shifted = d - np.max(d, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)
