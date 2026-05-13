"""
compare.py - Enhanced sanity check for FaceShield protection results.

Compares source.png vs protected.png across all per-image folders under
results/total_iter30/step_size1.0/noise_clamp12/ and reports:
  - ArcFace cosine similarity   (lower = embedding attack stronger)
  - CLIP   cosine similarity    (lower = embedding attack stronger)
  - PSNR, SSIM                  (higher = better visual quality preservation)
  - LPIPS                       (lower = perceptually closer to source; primary visual quality metric)
  - pixel_diff                  (mean abs pixel difference)

LPIPS is the perceptual quality metric used in the FaceShield paper. PSNR/SSIM
can be misleading for adversarial perturbations; LPIPS correlates much better
with human visual perception. By default LPIPS runs on CPU to save GPU memory.

Outputs:
  - prints per-image table + AVG/STD + delta vs previous run
  - appends one row to compare_log.csv (cumulative)

Usage:
  cd faceshield_original   # or _copy / _modified
  python compare.py
  python compare.py --tag baseline_v0
  python compare.py --results-dir results/total_iter30/step_size1.0/noise_clamp12 --tag softlowpass_v1
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.landmark.arcface_attack import AttackArcFace
from transformers import CLIPVisionModelWithProjection

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    print('[warn] lpips not installed. Run: pip install lpips')


# ----------------------------------------------------------------------
# Model loading (matches your existing compare code)
# ----------------------------------------------------------------------
def load_models(device='cuda', lpips_device='cpu', lpips_net='alex'):
    arc_model = torch.load(
        'models/arcface100_checkpoint.tar',
        map_location=device,
        weights_only=False,
    )
    arc_model.eval()
    preprocess_arc = AttackArcFace()

    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        'h94/IP-Adapter', subfolder='models/image_encoder'
    ).eval().to(device)

    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    # LPIPS (perceptual quality). CPU by default → no GPU pressure.
    lpips_fn = None
    if _LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False).to(lpips_device)
        lpips_fn.eval()
        print(f'  LPIPS loaded on {lpips_device} (net={lpips_net})')

    return arc_model, preprocess_arc, clip_model, clip_transform, lpips_fn, lpips_device


# ----------------------------------------------------------------------
# Per-pair measurement
# ----------------------------------------------------------------------
def measure_pair(src_path, prt_path, arc_model, preprocess_arc,
                 clip_model, clip_transform, lpips_fn=None, lpips_device='cpu',
                 device='cuda'):
    src = cv2.imread(str(src_path))
    prt = cv2.imread(str(prt_path))
    if src is None or prt is None:
        return None
    if src.shape != prt.shape:
        prt = cv2.resize(prt, (src.shape[1], src.shape[0]))

    pixel_diff = float(np.mean(np.abs(src.astype(float) - prt.astype(float))))

    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    prt_rgb = cv2.cvtColor(prt, cv2.COLOR_BGR2RGB)

    # ArcFace cosine sim
    src_t = torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    prt_t = torch.from_numpy(prt_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    _, src_arc = preprocess_arc.preprocess(src_t)
    _, prt_arc = preprocess_arc.preprocess(prt_t)
    with torch.no_grad():
        src_emb = arc_model(src_arc)
        prt_emb = arc_model(prt_arc)
    arc_sim = torch.nn.functional.cosine_similarity(src_emb, prt_emb).item()

    # CLIP cosine sim (IP-Adapter image encoder)
    src_pil = Image.fromarray(src_rgb)
    prt_pil = Image.fromarray(prt_rgb)
    src_clip = clip_transform(src_pil).unsqueeze(0).to(device)
    prt_clip = clip_transform(prt_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        src_feat = clip_model(src_clip).image_embeds
        prt_feat = clip_model(prt_clip).image_embeds
    clip_sim = torch.nn.functional.cosine_similarity(src_feat, prt_feat).item()

    # Visual quality: PSNR / SSIM (RGB domain)
    psnr = float(peak_signal_noise_ratio(src_rgb, prt_rgb, data_range=255))
    ssim = float(structural_similarity(src_rgb, prt_rgb, channel_axis=2, data_range=255))

    # LPIPS (perceptual). Expects [-1, 1] range, NCHW.
    lp = float('nan')
    if lpips_fn is not None:
        src_lp = torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device) / 127.5 - 1.0
        prt_lp = torch.from_numpy(prt_rgb).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device) / 127.5 - 1.0
        with torch.no_grad():
            lp = lpips_fn(src_lp, prt_lp).item()

    return {
        'pixel_diff': pixel_diff,
        'arc_sim': arc_sim,
        'clip_sim': clip_sim,
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lp,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
KEYS = ['pixel_diff', 'arc_sim', 'clip_sim', 'psnr', 'ssim', 'lpips']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir',
                        default='results/total_iter30/step_size1.0/noise_clamp12',
                        help='Parent folder containing per-image subfolders.')
    parser.add_argument('--tag', default='exp',
                        help='Experiment tag, e.g. baseline_v0, softlowpass_v1')
    parser.add_argument('--log', default='compare_log.csv',
                        help='Cumulative log CSV path.')
    parser.add_argument('--device', default='cuda',
                        help='Device for ArcFace/CLIP (cuda or cpu).')
    parser.add_argument('--lpips-device', default='cpu',
                        help='Device for LPIPS (cpu recommended to save VRAM).')
    parser.add_argument('--lpips-net', default='alex',
                        help='LPIPS backbone: alex (faster) or vgg (used in FaceShield paper).')
    parser.add_argument('--no-lpips', action='store_true', help='Skip LPIPS computation.')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f'[ERROR] results dir not found: {results_dir}')
        return

    image_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    if not image_dirs:
        print(f'[ERROR] no subfolders under {results_dir}')
        return

    print(f'Loading models on {args.device} ...')
    if args.no_lpips:
        # Load without LPIPS
        arc_model, preprocess_arc, clip_model, clip_transform, _, _ = \
            load_models(args.device, lpips_device='cpu', lpips_net=args.lpips_net)
        lpips_fn = None
        lpips_device = 'cpu'
    else:
        arc_model, preprocess_arc, clip_model, clip_transform, lpips_fn, lpips_device = \
            load_models(args.device, lpips_device=args.lpips_device, lpips_net=args.lpips_net)

    rows = []
    header = (f'{"image":<20} {"px_diff":>8} {"arc_sim":>8} {"clip_sim":>9} '
              f'{"psnr":>7} {"ssim":>7} {"lpips":>7}')
    print()
    print(header)
    print('-' * len(header))
    for d in image_dirs:
        src = d / 'source.png'
        prt = d / 'protected.png'
        m = measure_pair(src, prt, arc_model, preprocess_arc,
                         clip_model, clip_transform,
                         lpips_fn=lpips_fn, lpips_device=lpips_device,
                         device=args.device)
        if m is None:
            print(f'{d.name:<20} [skipped: missing source.png or protected.png]')
            continue
        rows.append({'image': d.name, **m})
        print(f'{d.name:<20} {m["pixel_diff"]:>8.2f} {m["arc_sim"]:>8.4f} '
              f'{m["clip_sim"]:>9.4f} {m["psnr"]:>7.2f} {m["ssim"]:>7.4f} '
              f'{m["lpips"]:>7.4f}')

    if not rows:
        print('No data to report.')
        return

    avg = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    std = {k: float(np.std([r[k] for r in rows])) for k in KEYS}

    print('-' * len(header))
    print(f'{"AVG":<20} {avg["pixel_diff"]:>8.2f} {avg["arc_sim"]:>8.4f} '
          f'{avg["clip_sim"]:>9.4f} {avg["psnr"]:>7.2f} {avg["ssim"]:>7.4f} '
          f'{avg["lpips"]:>7.4f}')
    print(f'{"STD":<20} {std["pixel_diff"]:>8.2f} {std["arc_sim"]:>8.4f} '
          f'{std["clip_sim"]:>9.4f} {std["psnr"]:>7.2f} {std["ssim"]:>7.4f} '
          f'{std["lpips"]:>7.4f}')

    # Delta vs previous run (last row in log). Tolerant to missing keys (older logs).
    log_path = Path(args.log)
    prev_avg = None
    prev_tag = '?'
    if log_path.exists():
        try:
            with open(log_path, 'r', newline='') as f:
                reader = list(csv.DictReader(f))
                if reader:
                    prev = reader[-1]
                    prev_tag = prev.get('tag', '?')
                    prev_avg = {}
                    for k in KEYS:
                        v = prev.get(f'avg_{k}', '')
                        prev_avg[k] = float(v) if v not in ('', None) else float('nan')
        except Exception as e:
            print(f'[warn] could not parse {log_path}: {e}')

    if prev_avg is not None:
        delta = {k: avg[k] - prev_avg[k] for k in KEYS}
        print('-' * len(header))
        print(f'{"Δ vs " + prev_tag:<20} {delta["pixel_diff"]:>+8.2f} '
              f'{delta["arc_sim"]:>+8.4f} {delta["clip_sim"]:>+9.4f} '
              f'{delta["psnr"]:>+7.2f} {delta["ssim"]:>+7.4f} '
              f'{delta["lpips"]:>+7.4f}')

    # Append to cumulative log
    log_row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'tag': args.tag,
        'n_images': len(rows),
        **{f'avg_{k}': avg[k] for k in KEYS},
        **{f'std_{k}': std[k] for k in KEYS},
    }
    write_header = not log_path.exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(log_row)

    print(f'\n[OK] tag="{args.tag}" with {len(rows)} images appended to {log_path}')


if __name__ == '__main__':
    main()