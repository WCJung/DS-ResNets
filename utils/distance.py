import numpy as np
import torch
import numba


def minkovski(input1, input2, p):
    if not isinstance(input1, torch.Tensor):
        input1 = torch.from_numpy(np.array(input1))
    if not isinstance(input2, torch.Tensor):
        input2 = torch.from_numpy(np.array(input2))
    tmp = torch.pow(torch.abs(input1 - input2), p)
    tmp = torch.sum(tmp, dim=-1)

    return torch.pow(tmp, (1 / p))


def euclidean(input1, input2):
    tmp = np.power(np.abs(input1 - input2), 2)
    tmp = np.sum(tmp, axis=-1)

    return np.power(tmp, (1 / 2))


def manhattan(input1, input2):
    res = np.sum(np.abs(input1 - input2), axis=-1)

    return res


def chebychev(input1, input2):
    res = np.max(np.abs(input1 - input2), axis=-1)

    return res


def hausdorff_distance(input1, input2):
    raise NotImplementedError("hausdorff_distance is not yet implemented")

