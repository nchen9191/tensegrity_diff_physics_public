import torch


def zeros(shape, dtype=None, device=None, ref_tensor=None):
    if ref_tensor is None and (dtype is None or device is None):
        raise Exception("Need to specify either ref tensor or (dtype and device)")

    if ref_tensor is not None:
        dtype = ref_tensor.dtype
        device = ref_tensor.device

    return torch.zeros(shape, dtype=dtype, device=device)


def ones(shape, dtype=None, device=None, ref_tensor=None):
    if ref_tensor is None and (dtype is None or device is None):
        raise Exception("Need to specify either ref tensor or (dtype and device)")

    if ref_tensor is not None:
        dtype = ref_tensor.dtype
        device = ref_tensor.device

    return torch.ones(shape, dtype=dtype, device=device)


def interleave_tensors(*tensors):
    height = sum(t.shape[0] for t in tensors)
    width = tensors[0].shape[1]

    return torch.hstack(tensors).view(height, width)


def norm_tensor(tensor, dim=1):
    return tensor / torch.clamp_min(tensor.norm(dim=dim, keepdim=True), 1e-8)


def tensor_3d_to_2d(tensor_3d):
    """
    (batch_size, num_comps, num_vecs_per_batch)
    -> (batch_size * num_vecs_per_batch, num_comps)
    """

    return tensor_3d.transpose(1, 2).reshape(-1, tensor_3d.shape[1])