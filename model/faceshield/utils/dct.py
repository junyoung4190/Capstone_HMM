"""
utils/dct.py

DROP-IN REPLACEMENT for faceshield's utils/dct.py.

CHANGE: dct_pass_filter() now supports a 'soft' raised-cosine taper in
addition to the original paper's binary (hard) mask.

  - mode='soft' (default): smooth cutoff -> reduces ringing artifacts
                           in flat regions (e.g., skin) while preserving
                           low-frequency protection signal.
  - mode='hard'          : original FaceShield paper binary mask.

You can also override the mode at runtime via env var without editing code:
    DCT_MODE=hard sh run.sh
    DCT_MODE=soft sh run.sh   (default)

ALL OTHER FUNCTIONS in this file are UNCHANGED from the original
faceshield repo (make_dct_basis, encode, decode, padding, blockfy, deblockfy).
"""

import os
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Original helpers (unchanged)
# ----------------------------------------------------------------------
def make_dct_basis(N, device):
    x, y = torch.meshgrid(torch.arange(N, device=device),
                          torch.arange(N, device=device), indexing='ij')
    u, v = torch.meshgrid(torch.arange(N, device=device),
                          torch.arange(N, device=device), indexing='ij')
    basis = torch.cos((2 * x + 1) * u.unsqueeze(-1).unsqueeze(-1) * torch.pi / (2 * N)) * \
            torch.cos((2 * y + 1) * v.unsqueeze(-1).unsqueeze(-1) * torch.pi / (2 * N))

    N_tensor = torch.arange(N, device=device)
    alpha_u = torch.where(N_tensor == 0,
                          torch.sqrt(torch.tensor(1.0 / N, device=device)),
                          torch.sqrt(torch.tensor(2.0 / N, device=device)))
    alpha_v = torch.where(N_tensor == 0,
                          torch.sqrt(torch.tensor(1.0 / N, device=device)),
                          torch.sqrt(torch.tensor(2.0 / N, device=device)))
    basis = torch.einsum('u,v,uvhw->uvhw', alpha_u, alpha_v, basis)
    return basis


def encode(dct_blocks, DCT_basis):
    R_dct_block = torch.einsum('abcd,cdef->abef', dct_blocks[0], DCT_basis)
    G_dct_block = torch.einsum('abcd,cdef->abef', dct_blocks[1], DCT_basis)
    B_dct_block = torch.einsum('abcd,cdef->abef', dct_blocks[2], DCT_basis)
    return torch.stack([R_dct_block, G_dct_block, B_dct_block])


def decode(dct_blocks, IDCT_basis):
    R_idct_block = torch.einsum('abef,cdef->abcd', dct_blocks[0], IDCT_basis)
    G_idct_block = torch.einsum('abef,cdef->abcd', dct_blocks[1], IDCT_basis)
    B_idct_block = torch.einsum('abef,cdef->abcd', dct_blocks[2], IDCT_basis)
    return torch.stack([R_idct_block, G_idct_block, B_idct_block])


def padding(tensor, N):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    b, c, height, width = tensor.shape
    padding_length = 0
    padding_width = 0
    if height % N != 0:
        padding_length = N - height % N
    if width % N != 0:
        padding_width = N - width % N
    padded_data = F.pad(tensor, (0, padding_width, 0, padding_length))
    return padded_data, (padding_length, padding_width)


def blockfy(tensor, N):
    padded_data, pad_size = padding(tensor, N)
    b, channel, height, width = padded_data.shape
    num_blocks_height = height // N
    num_blocks_width = width // N
    unfolded = padded_data.unfold(2, N, N).unfold(3, N, N)
    blocks = unfolded.contiguous().view(channel, num_blocks_height, num_blocks_width, N, N)
    return blocks, pad_size


def deblockfy(blocks, pad_size):
    channel, num_blocks_height, num_blocks_width, N, N = blocks.shape
    height = num_blocks_height * N
    width = num_blocks_width * N
    blocks_reshaped = blocks.permute(0, 1, 3, 2, 4).reshape(1, channel, height, width)
    tensor = blocks_reshaped[:, :, :height - pad_size[0], :width - pad_size[1]]
    return tensor


# ----------------------------------------------------------------------
# DCT pass filter (MODIFIED for capstone improvement)
# ----------------------------------------------------------------------
def _hard_low_pass_8x8(device):
    """Original FaceShield paper's binary lowpass mask."""
    return torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], device=device, dtype=torch.float32)


def _soft_low_pass_8x8(device, d_low=4, d_high=7):
    """
    Raised-cosine taper on diagonal frequency index d = i + j.

      d <= d_low : weight = 1.0    (low-freq, fully preserved)
      d_low < d <= d_high : weight = 0.5 * (1 + cos(pi * (d - d_low) / (d_high - d_low)))
      d > d_high : weight = 0.0

    Default (d_low=4, d_high=7) preserves ~85% of original mask energy
    (vs hard mask), keeps DC + low frequencies fully intact, and replaces
    the abrupt cutoff with a smooth taper at d=5,6,7.
    """
    N = 8
    i = torch.arange(N, device=device).view(N, 1).expand(N, N).float()
    j = torch.arange(N, device=device).view(1, N).expand(N, N).float()
    d = i + j

    mask = torch.ones_like(d)
    in_transition = (d > d_low) & (d <= d_high)
    t = (d - d_low) / max(d_high - d_low, 1)
    taper = 0.5 * (1.0 + torch.cos(torch.pi * t))
    mask = torch.where(in_transition, taper, mask)
    mask = torch.where(d > d_high, torch.zeros_like(d), mask)
    return mask


def dct_pass_filter(device, mode=None, d_low=4, d_high=7):
    """
    8x8 DCT lowpass / highpass filter masks.

    Args:
        device: torch device.
        mode: 'soft' (default, capstone improvement) or 'hard' (paper original).
              If None, read from env var DCT_MODE (default 'soft').
        d_low, d_high: soft taper boundaries (only used in 'soft' mode).

    Returns:
        Low_pass_filter, High_pass_filter, each shape [1, 1, 1, 8, 8].
    """
    if mode is None:
        mode = os.environ.get('DCT_MODE', 'soft').lower()

    if mode == 'hard':
        low_pass_filter = _hard_low_pass_8x8(device)
    elif mode == 'soft':
        low_pass_filter = _soft_low_pass_8x8(device, d_low=d_low, d_high=d_high)
    else:
        raise ValueError(f'Unknown DCT_MODE: {mode!r} (expected "soft" or "hard")')

    high_pass_filter = 1.0 - low_pass_filter

    Low_pass_filter = low_pass_filter[None, None, None, :, :]
    High_pass_filter = high_pass_filter[None, None, None, :, :]
    return Low_pass_filter, High_pass_filter


# ----------------------------------------------------------------------
# Sanity-check helper: print current mask
#   python -c "from utils.dct import _print_mask; _print_mask()"
# ----------------------------------------------------------------------
def _print_mask(mode=None):
    import sys
    device = torch.device('cpu')
    lp, hp = dct_pass_filter(device, mode=mode)
    print(f'low_pass_filter (mode={mode or os.environ.get("DCT_MODE", "soft")}):')
    for row in lp[0, 0, 0].tolist():
        print('  [' + '  '.join(f'{v:.3f}' for v in row) + ']')
    print(f'\nsum  = {lp.sum().item():.3f}   (hard original = 25.000)')
    print(f'energy (sum of squares) = {(lp ** 2).sum().item():.3f}   (hard original = 25.000)')