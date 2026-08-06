"""P1: Long-horizon consistency metrics (video + PhysMem logs).

Custom (framework-level aggregation, built on standard SSIM/CLIP):
  - Revisit Consistency (RC): SSIM between the current revisit frame and the
    retrieved first-visit frame (best_candidate_frame_id).
  - Long-term Temporal Consistency (TC_k): mean SSIM(F_t, F_{t+k}) curve.

Standard proxy (built on OpenAI CLIP):
  - CLIP-Frame similarity: cosine similarity between CLIP features of frame t
    and the first frame, plus adjacent-frame mean (enabled with --use_clip).
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np


RETRIEVE_MODES = {"retrieve_window", "soft_reuse_window"}


def read_frame_log(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ssim(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    """Structural similarity on grayscale (cv2 implementation, no skimage dep)."""
    a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + 1e-6
    )
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def latent_to_frame(latent_id: int, sub: int = 2) -> int:
    """Approx mapping latent frame -> video frame (4x upsampling, sub position).
    Calibrate against the debug PNGs before trusting RC values."""
    return int(latent_id) * 4 + int(sub)


def extract_frame(cap, frame_idx: int, total: int) -> np.ndarray | None:
    frame_idx = max(0, min(int(frame_idx), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def compute_rc(rows, cap, total, sub) -> dict:
    values = []
    pairs = []
    for r in rows:
        if r.get("view_state") != "Revisit":
            continue
        if r.get("memory_selection_mode") not in RETRIEVE_MODES:
            continue
        cand = r.get("best_candidate_frame_id")
        if not cand:
            continue
        cur = extract_frame(cap, latent_to_frame(r.get("frame_index", 0), sub), total)
        ref = extract_frame(cap, latent_to_frame(cand, sub), total)
        if cur is not None and ref is not None:
            s = ssim(cur, ref)
            values.append(s)
            pairs.append({"frame_index": r.get("frame_index"), "candidate_frame_id": int(float(cand)), "ssim": s})
    return {
        "rc_mean_ssim": float(np.mean(values)) if values else 0.0,
        "rc_count": len(values),
        "pairs": pairs,
    }


def compute_tc(cap, total, ks=(12, 24, 48, 96)) -> dict:
    tc = {}
    for k in ks:
        step = k * 4
        values = []
        t = 0
        while t + step < total:
            a = extract_frame(cap, t, total)
            b = extract_frame(cap, t + step, total)
            if a is not None and b is not None:
                values.append(ssim(a, b))
            t += step
        tc[f"tc_{k}"] = float(np.mean(values)) if values else 0.0
    return tc


def compute_clip_frame(cap, total, sub, device="cuda") -> dict:
    try:
        import torch
        import clip
    except ImportError as exc:
        return {"error": f"clip not available: {exc}"}
    model, preprocess = clip.load("ViT-B/32", device=device)
    feats = []
    idx = 0
    while idx < total:
        frame = extract_frame(cap, idx, total)
        if frame is None:
            break
        img = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.cpu().numpy().squeeze())
        idx += sub * 4  # sample every `sub*4` video frames (one latent step)

    if len(feats) < 2:
        return {"error": "too few frames"}
    feats = np.stack(feats)
    first = feats[0]
    to_first = [float(np.dot(feats[i], first)) for i in range(len(feats))]
    adjacent = [float(np.dot(feats[i], feats[i + 1])) for i in range(len(feats) - 1)]
    return {
        "clip_frame_vs_first_mean": float(np.mean(to_first)),
        "clip_frame_vs_first_min": float(np.min(to_first)),
        "clip_adjacent_mean": float(np.mean(adjacent)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 Long-horizon consistency metrics (RC / TC_k custom, CLIP-Frame standard)")
    parser.add_argument("--video", required=True, help="Generated video (mp4)")
    parser.add_argument("--log", required=True, help="Corresponding stableworld_frame_log.csv")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--sub", type=int, default=2, help="latent->video frame offset (calibrate against debug PNGs)")
    parser.add_argument("--use_clip", action="store_true", help="Also compute CLIP-Frame similarity")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = read_frame_log(args.log)
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit(f"Cannot read video {args.video}")

    result = {
        "video": args.video,
        "total_frames": total,
        "rc": compute_rc(rows, cap, total, args.sub),
        "tc_curve": compute_tc(cap, total),
    }
    if args.use_clip:
        result["clip_frame"] = compute_clip_frame(cap, total, args.sub, args.device)
    cap.release()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
