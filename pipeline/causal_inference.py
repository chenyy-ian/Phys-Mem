from typing import List, Optional
import csv
import json
import os
import numpy as np
import torch
import time
import copy

from einops import rearrange
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
from utils.visualize import process_video
import torch.nn.functional as F
from demo_utils.constant import ZERO_VAE_CACHE
from .experiment_tracking import ExperimentRecorder
from .stableworld_memory import MemoryBuffer, MemoryScheduler
from .stableworld_similarity import build_similarity_estimator, orb_ransac_score_chw
from tqdm import tqdm

def get_current_action(mode="universal"):

    CAM_VALUE = 0.1
    if mode == 'universal':
        print()
        print('-'*30)
        print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM\n (I: up, K: down, J: left, L: right, U: no move)")
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "i":  [CAM_VALUE, 0],
            "k":  [-CAM_VALUE, 0],
            "j":  [0, -CAM_VALUE],
            "l":  [0, CAM_VALUE],
            "u":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
            "q": [0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_mouse = input('Please input the mouse action (e.g. `U`):\n').strip().lower()
                idx_keyboard = input('Please input the keyboard action (e.g. `W`):\n').strip().lower()
                if idx_mouse in CAMERA_VALUE_MAP.keys() and idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    elif mode == 'gta_drive':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "a":  [0, -CAM_VALUE],
            "d":  [0, CAM_VALUE],
            "q":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0], "s": [0, 1],
            "q": [0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                indexes = input('Please input the actions (split with ` `):\n(e.g. `W` for forward, `W A` for forward and left)\n').strip().lower().split(' ')
                idx_mouse = []
                idx_keyboard = []
                for i in indexes:
                    if i in CAMERA_VALUE_MAP.keys():
                        idx_mouse += [i]
                    elif i in KEYBOARD_IDX.keys():
                        idx_keyboard += [i]
                if len(idx_mouse) == 0:
                    idx_mouse += ['q']
                if len(idx_keyboard) == 0:
                    idx_keyboard += ['q']
                assert idx_mouse in [['a'], ['d'], ['q']] and idx_keyboard in [['q'], ['w'], ['s']]
                flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse[0]]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard[0]]).cuda()
    elif mode == 'templerun':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Z, C, Q] FOR ACTIONS\n (W: jump, S: slide, A: left side, D: right side, Z: turn left, C: turn right, Q: no move)")
        print('-'*30)
        KEYBOARD_IDX = { 
            "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
            "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
            "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
            "q": [1, 0, 0, 0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_keyboard = input('Please input the action: \n(e.g. `W` for forward, `Z` for turning left)\n').strip().lower()
                if idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    
    if mode != 'templerun':
        return {
            "mouse": mouse_cond,
            "keyboard": keyboard_cond
        }
    return {
        "keyboard": keyboard_cond
    }

def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode='universal'):
    
    new_cond = {}
    
    new_cond["cond_concat"] = conditional_dict["cond_concat"][:, :, current_start_frame: current_start_frame + num_frame_per_block]
    new_cond["visual_context"] = conditional_dict["visual_context"]
    if replace != None:
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block
        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block-1)
        if mode != 'templerun':
            conditional_dict["mouse_cond"][:, -last_frame_num + final_frame: final_frame] = replace['mouse'][None, None, :].repeat(1, last_frame_num, 1)
        conditional_dict["keyboard_cond"][:, -last_frame_num + final_frame: final_frame] = replace['keyboard'][None, None, :].repeat(1, last_frame_num, 1)
    if mode != 'templerun':
        new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]
    new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]

    if replace != None:
        return new_cond, conditional_dict
    else:
        return new_cond

import cv2


class StableWorldDebugLogger:
    def __init__(self, output_dir: str, bucket_size: int = 100):
        self.output_dir = output_dir
        self.bucket_size = bucket_size
        self.frames_dir = os.path.join(output_dir, "frames")
        self.matches_dir = os.path.join(output_dir, "orb_matches")
        self.heatmaps_dir = os.path.join(output_dir, "matching_heatmaps")
        self.confidence_dir = os.path.join(output_dir, "confidence_distributions")
        self.displacement_dir = os.path.join(output_dir, "displacement_fields")
        self.motion_vector_dir = os.path.join(output_dir, "motion_vectors")
        self.depth_dir = os.path.join(output_dir, "depth_maps")
        self.depth_diff_dir = os.path.join(output_dir, "depth_differences")
        self.depth_heatmap_dir = os.path.join(output_dir, "depth_heatmaps")
        self.depth_histogram_dir = os.path.join(output_dir, "depth_histograms")
        self.records = []
        self.events = []
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.matches_dir, exist_ok=True)
        os.makedirs(self.heatmaps_dir, exist_ok=True)
        os.makedirs(self.confidence_dir, exist_ok=True)
        os.makedirs(self.displacement_dir, exist_ok=True)
        os.makedirs(self.motion_vector_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.depth_diff_dir, exist_ok=True)
        os.makedirs(self.depth_heatmap_dir, exist_ok=True)
        os.makedirs(self.depth_histogram_dir, exist_ok=True)

    def log_event(self, frame_index: int, stage: str, detail: str):
        item = {
            "frame_index": int(frame_index),
            "stage": stage,
            "detail": detail,
        }
        self.events.append(item)
        print(f"[StableWorldDebug][frame={frame_index:04d}][{stage}] {detail}")

    def log_decision(self, record: dict):
        self.records.append(record)
        print(
            "[StableWorldDebug]"
            f" frame={record['frame_index']:04d}"
            f" ref={record['reference_frame']}"
            f" middle={record['middle_frame']}"
            f" sim={record['similarity']:.4f}"
            f" delete={record['delete_frame']}"
            f" memory={record['memory_size']}"
            f" matches={record['matching_points']}"
            f" orb_ms={record['orb_runtime_ms']:.2f}"
        )

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        records_path = os.path.join(self.output_dir, "stableworld_frame_log.csv")
        with open(records_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_index",
                    "reference_frame",
                    "middle_frame",
                    "current_frame",
                    "similarity",
                    "delete_frame",
                    "delete_range",
                    "memory_size",
                    "matching_points",
                    "orb_runtime_ms",
                    "lightglue_runtime_ms",
                    "runtime_ms",
                    "confidence",
                    "semantic_similarity",
                    "final_similarity",
                    "spatial_alpha",
                    "average_displacement",
                    "median_displacement",
                    "maximum_displacement",
                    "depth_runtime_ms",
                    "geometry_similarity",
                    "depth_difference",
                    "depth_metric",
                    "depth_cache_hit_reference",
                    "depth_cache_hit_current",
                    "estimator",
                    "decision",
                    "window_before",
                    "window_after",
                ],
            )
            writer.writeheader()
            writer.writerows(self.records)

        summary_path = os.path.join(self.output_dir, "stableworld_summary_100f.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_bucket_start",
                    "frame_bucket_end",
                    "memory_replacements",
                    "average_similarity",
                    "average_matching_points",
                    "average_orb_runtime_ms",
                    "average_lightglue_runtime_ms",
                    "average_runtime_ms",
                    "average_confidence",
                    "average_semantic_similarity",
                    "average_final_similarity",
                    "average_displacement",
                    "median_displacement",
                    "maximum_displacement",
                    "average_depth_difference",
                    "average_depth_runtime_ms",
                    "average_geometry_similarity",
                    "depth_cache_hit_rate",
                ],
            )
            writer.writeheader()
            for row in self._bucket_summary():
                writer.writerow(row)

        events_path = os.path.join(self.output_dir, "stableworld_execution_trace.json")
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump({"events": self.events, "records": self.records}, f, indent=2)

        self._save_timeline()

    def _bucket_summary(self):
        if not self.records:
            return []
        buckets = {}
        for record in self.records:
            bucket_start = (record["frame_index"] // self.bucket_size) * self.bucket_size
            buckets.setdefault(bucket_start, []).append(record)

        rows = []
        for bucket_start in sorted(buckets):
            items = buckets[bucket_start]
            rows.append({
                "frame_bucket_start": bucket_start,
                "frame_bucket_end": bucket_start + self.bucket_size - 1,
                "memory_replacements": len(items),
                "average_similarity": float(np.mean([x["similarity"] for x in items])),
                "average_matching_points": float(np.mean([x["matching_points"] for x in items])),
                "average_orb_runtime_ms": float(np.mean([x["orb_runtime_ms"] for x in items])),
                "average_lightglue_runtime_ms": float(np.mean([x["lightglue_runtime_ms"] for x in items])),
                "average_runtime_ms": float(np.mean([x["runtime_ms"] for x in items])),
                "average_confidence": float(np.mean([x["confidence"] for x in items])),
                "average_semantic_similarity": float(np.mean([x["semantic_similarity"] for x in items])),
                "average_final_similarity": float(np.mean([x["final_similarity"] for x in items])),
                "average_displacement": float(np.mean([x["average_displacement"] for x in items])),
                "median_displacement": float(np.median([x["median_displacement"] for x in items])),
                "maximum_displacement": float(np.max([x["maximum_displacement"] for x in items])),
                "average_depth_difference": float(np.mean([x["depth_difference"] for x in items])),
                "average_depth_runtime_ms": float(np.mean([x["depth_runtime_ms"] for x in items])),
                "average_geometry_similarity": float(np.mean([x["geometry_similarity"] for x in items])),
                "depth_cache_hit_rate": float(np.mean([
                    (int(x["depth_cache_hit_reference"]) + int(x["depth_cache_hit_current"])) / 2.0
                    for x in items
                ])),
            })
        return rows

    def _save_timeline(self):
        if not self.records:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        xs = [r["frame_index"] for r in self.records]
        sims = [r["similarity"] for r in self.records]
        mem_sizes = [r["memory_size"] for r in self.records]
        delete_flags = [1 if r["decision"] == "delete_middle" else 0 for r in self.records]

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(xs, sims, marker="o", linewidth=1)
        axes[0].set_ylabel("Similarity")
        axes[0].grid(True, alpha=0.3)

        axes[1].stem(xs, delete_flags, basefmt=" ")
        axes[1].set_ylabel("Delete Event")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["oldest", "middle"])
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(xs, mem_sizes, marker="o", linewidth=1, color="tab:green")
        axes[2].set_ylabel("Memory Size")
        axes[2].set_xlabel("Frame Index")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "stableworld_memory_timeline.png"), dpi=200)
        plt.close(fig)


def chw_to_rgb_u8(chw: torch.Tensor) -> np.ndarray:
    x = chw.detach().cpu().float()
    if x.min() < 0:
        x = ((x + 1.0) * 127.5).clamp(0, 255.0)
    else:
        x = (x * 255.0).clamp(0, 255.0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.byte().permute(1, 2, 0).numpy()


def save_debug_frame(chw: torch.Tensor, path: str):
    rgb = chw_to_rgb_u8(chw)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def get_decoded_frame_by_latent(videos: list, latent_id: int, sub: int = 2):
    """
    From `videos`, retrieve the corresponding decoded single frame (CHW tensor)
    according to the latent frame index.

    - videos[blk] has shape [1, T_blk, 3, H, W],
      where T_blk is 9 for the first block and 12 for all later blocks
    - Each latent block contains 3 latent frames: pos = 0, 1, 2
    - `sub` selects which sub-frame to use among the 4x upsampled frames,
      where sub ∈ {0,1,2,3}
    - For the last latent in the first block, only one frame is available
    """
    blk = latent_id // 3           # which block the latent belongs to
    pos = latent_id % 3            # position within the block (0, 1, or 2)

    v = videos[blk]                # [1, T_blk, 3, H, W]
    T_blk = v.shape[1]

    if blk == 0:
        # First block: 4 + 4 + 1 = 9
        if pos == 0:          # first latent -> off: 0..3
            off = min(sub, 3)
        elif pos == 1:        # second latent -> off: 4..7
            off = min(4 + sub, 7)
        else:                 # third latent -> only off: 8
            off = 8
    else:
        # Other blocks: 4 + 4 + 4 = 12
        off = pos * 4 + min(sub, 3)   # 0..3, 4..7, 8..11

    # Safety fallback to avoid out-of-range indexing
    if off >= T_blk:
        off = T_blk - 1

    # Extract a single frame [3, H, W]
    chw = v[0, off]   # note: video layout is [1, T_blk, 3, H, W]
    return chw


def decide_and_update_window_ids_tri_9(
    window_ids: list,
    videos: list,
    sim_threshold: float = 0.75,
    sub: int = 2,
    debug_logger: StableWorldDebugLogger | None = None,
    experiment_recorder: ExperimentRecorder | None = None,
    frame_index: int | None = None,
    similarity_estimator_name: str = "orb",
    lightglue_spatial_alpha: float = 0.0,
    depth_metric: str = "l1",
    depth_model: str = "vits",
    depth_checkpoint: str | None = None,
    depth_cache_size: int = 256,
) -> tuple[int, list, float]:
    return schedule_stableworld_window_tri_9(
        window_ids=window_ids,
        videos=videos,
        sim_threshold=sim_threshold,
        sub=sub,
        debug_logger=debug_logger,
        experiment_recorder=experiment_recorder,
        frame_index=frame_index,
        similarity_estimator_name=similarity_estimator_name,
        lightglue_spatial_alpha=lightglue_spatial_alpha,
        depth_metric=depth_metric,
        depth_model=depth_model,
        depth_checkpoint=depth_checkpoint,
        depth_cache_size=depth_cache_size,
    )


def schedule_stableworld_window_tri_9(
    window_ids: list,
    videos: list,
    sim_threshold: float = 0.75,
    sub: int = 2,
    debug_logger: StableWorldDebugLogger | None = None,
    experiment_recorder: ExperimentRecorder | None = None,
    frame_index: int | None = None,
    similarity_estimator_name: str = "orb",
    lightglue_spatial_alpha: float = 0.0,
    depth_metric: str = "l1",
    depth_model: str = "vits",
    depth_checkpoint: str | None = None,
    depth_cache_size: int = 256,
) -> tuple[int, list, float]:
    """
    StableWorld tri-9 scheduling through the refactored stack:
    SimilarityEstimator -> MemoryScheduler -> MemoryBuffer.
    """
    memory_buffer = MemoryBuffer(window_ids)
    L = len(memory_buffer)
    assert L >= 6, "Window must contain at least 6 frames"

    if L < 7:
        evict_middle = 0
        new_ids = window_ids[3:] + [memory_buffer.last_id + 1, memory_buffer.last_id + 2, memory_buffer.last_id + 3]
        return evict_middle, new_ids, 0.0

    id2 = memory_buffer.reference_frame_id
    id5 = memory_buffer.middle_frame_id
    current_id = memory_buffer.current_frame_id
    window_before = memory_buffer.snapshot()

    img2 = get_decoded_frame_by_latent(videos, id2, sub=sub)
    img5 = get_decoded_frame_by_latent(videos, id5, sub=sub)

    similarity_estimator = build_similarity_estimator(
        similarity_estimator_name,
        lightglue_spatial_alpha=lightglue_spatial_alpha,
        depth_metric=depth_metric,
        depth_model=depth_model,
        depth_checkpoint=depth_checkpoint,
        depth_cache_size=depth_cache_size,
    )
    similarity_result = similarity_estimator.compute_similarity(
        img2,
        img5,
        return_debug=debug_logger is not None,
    )

    scheduler = MemoryScheduler(sim_threshold=sim_threshold)
    decision = scheduler.schedule(memory_buffer, similarity_result)
    new_ids = memory_buffer.apply_decision(decision)
    min_sim = similarity_result.similarity
    frame_index = current_id if frame_index is None else frame_index
    orb_runtime_ms = float(similarity_result.debug.get("orb_runtime_ms", 0.0))
    lightglue_runtime_ms = float(similarity_result.debug.get("lightglue_runtime_ms", 0.0))
    runtime_ms = lightglue_runtime_ms or orb_runtime_ms
    semantic_similarity = float(similarity_result.debug.get("semantic_similarity", similarity_result.similarity))
    final_similarity = float(similarity_result.debug.get("final_similarity", similarity_result.similarity))
    spatial_alpha = float(similarity_result.debug.get("spatial_alpha", lightglue_spatial_alpha))
    average_displacement = float(similarity_result.debug.get("average_displacement", 0.0))
    median_displacement = float(similarity_result.debug.get("median_displacement", 0.0))
    maximum_displacement = float(similarity_result.debug.get("maximum_displacement", 0.0))
    depth_runtime_ms = float(similarity_result.debug.get("depth_runtime_ms", 0.0))
    if depth_runtime_ms:
        runtime_ms = depth_runtime_ms
    geometry_similarity = float(similarity_result.debug.get("geometry_similarity", similarity_result.similarity))
    depth_difference = float(similarity_result.debug.get("depth_difference", 0.0))
    depth_metric_used = str(similarity_result.debug.get("depth_metric", ""))
    depth_cache_hit_reference = bool(similarity_result.debug.get("depth_cache_hit_reference", False))
    depth_cache_hit_current = bool(similarity_result.debug.get("depth_cache_hit_current", False))

    if experiment_recorder is not None:
        experiment_recorder.record(
            frame_id=int(frame_index),
            current_frame_id=int(current_id),
            ReferenceFrameID=int(id2),
            MiddleFrameID=int(id5),
            similarity=float(similarity_result.similarity),
            similarity_type=similarity_estimator_name,
            matching_points=int(similarity_result.matching_points),
            confidence=float(similarity_result.confidence),
            decision=decision.decision,
            memory_state=new_ids,
            memory_id="stableworld_tri_9",
            memory_size=int(len(new_ids)),
            replace_count=1,
            keep_count=int(len(new_ids) - len(decision.delete_range)),
            runtime=runtime_ms,
            EvidenceType=similarity_estimator_name,
            PolicyConfidence=float(decision.confidence),
            DecisionConfidence=float(similarity_result.confidence),
        )

    if debug_logger is not None:
        base_name = f"frame_{frame_index:04d}_ref_{id2:04d}_mid_{id5:04d}"
        ref_path = os.path.join(debug_logger.frames_dir, f"{base_name}_reference.png")
        middle_path = os.path.join(debug_logger.frames_dir, f"{base_name}_middle.png")
        match_path = os.path.join(debug_logger.matches_dir, f"{base_name}_{similarity_estimator_name}_match.png")
        heatmap_path = os.path.join(debug_logger.heatmaps_dir, f"{base_name}_heatmap.png")
        confidence_path = os.path.join(debug_logger.confidence_dir, f"{base_name}_confidence.png")
        displacement_path = os.path.join(debug_logger.displacement_dir, f"{base_name}_displacement.png")
        motion_vector_path = os.path.join(debug_logger.motion_vector_dir, f"{base_name}_motion_vector.png")
        depth_ref_path = os.path.join(debug_logger.depth_dir, f"{base_name}_depth_reference.png")
        depth_cur_path = os.path.join(debug_logger.depth_dir, f"{base_name}_depth_current.png")
        depth_diff_path = os.path.join(debug_logger.depth_diff_dir, f"{base_name}_depth_difference.png")
        depth_heatmap_path = os.path.join(debug_logger.depth_heatmap_dir, f"{base_name}_depth_heatmap.png")
        depth_histogram_path = os.path.join(debug_logger.depth_histogram_dir, f"{base_name}_depth_histogram.png")

        save_debug_frame(img2, ref_path)
        save_debug_frame(img5, middle_path)
        if similarity_result.debug.get("match_image") is not None:
            cv2.imwrite(match_path, similarity_result.debug["match_image"])
        if similarity_result.debug.get("matching_heatmap") is not None:
            cv2.imwrite(heatmap_path, similarity_result.debug["matching_heatmap"])
        if similarity_result.debug.get("confidence_distribution") is not None:
            cv2.imwrite(confidence_path, similarity_result.debug["confidence_distribution"])
        if similarity_result.debug.get("displacement_field") is not None:
            cv2.imwrite(displacement_path, similarity_result.debug["displacement_field"])
        if similarity_result.debug.get("motion_vector") is not None:
            cv2.imwrite(motion_vector_path, similarity_result.debug["motion_vector"])
        if similarity_result.debug.get("depth_reference") is not None:
            cv2.imwrite(depth_ref_path, similarity_result.debug["depth_reference"])
        if similarity_result.debug.get("depth_current") is not None:
            cv2.imwrite(depth_cur_path, similarity_result.debug["depth_current"])
        if similarity_result.debug.get("depth_difference_map") is not None:
            cv2.imwrite(depth_diff_path, similarity_result.debug["depth_difference_map"])
        if similarity_result.debug.get("depth_heatmap") is not None:
            cv2.imwrite(depth_heatmap_path, similarity_result.debug["depth_heatmap"])
        if similarity_result.debug.get("depth_histogram") is not None:
            cv2.imwrite(depth_histogram_path, similarity_result.debug["depth_histogram"])

        debug_logger.log_decision({
            "frame_index": int(frame_index),
            "reference_frame": int(id2),
            "middle_frame": int(id5),
            "current_frame": int(current_id),
            "similarity": float(similarity_result.similarity),
            "delete_frame": int(decision.delete_frame),
            "delete_range": "-".join(str(x) for x in decision.delete_range),
            "memory_size": int(len(new_ids)),
            "matching_points": int(similarity_result.matching_points),
            "orb_runtime_ms": orb_runtime_ms,
            "lightglue_runtime_ms": lightglue_runtime_ms,
            "runtime_ms": runtime_ms,
            "confidence": float(similarity_result.confidence),
            "semantic_similarity": semantic_similarity,
            "final_similarity": final_similarity,
            "spatial_alpha": spatial_alpha,
            "average_displacement": average_displacement,
            "median_displacement": median_displacement,
            "maximum_displacement": maximum_displacement,
            "depth_runtime_ms": depth_runtime_ms,
            "geometry_similarity": geometry_similarity,
            "depth_difference": depth_difference,
            "depth_metric": depth_metric_used,
            "depth_cache_hit_reference": depth_cache_hit_reference,
            "depth_cache_hit_current": depth_cache_hit_current,
            "estimator": similarity_estimator_name,
            "decision": decision.decision,
            "window_before": " ".join(str(x) for x in window_before),
            "window_after": " ".join(str(x) for x in new_ids),
        })

    return decision.evict_middle, new_ids, min_sim


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device="cuda",
            generator=None,
            vae_decoder=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
            
        self.vae_decoder = vae_decoder
        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent = None,
        return_latents = False,
        mode = 'universal',
        profile = False,
        evict_mode = False,
        Threshold=0.78,
        debug_stableworld: bool = False,
        debug_output_dir: str | None = None,
        similarity_estimator_name: str = "orb",
        lightglue_spatial_alpha: float = 0.0,
        depth_metric: str = "l1",
        depth_model: str = "vits",
        depth_checkpoint: str | None = None,
        depth_cache_size: int = 256,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        
        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        debug_logger = None
        experiment_recorder = None
        if debug_stableworld:
            debug_output_dir = debug_output_dir or "outputs/stableworld_debug"
            debug_logger = StableWorldDebugLogger(debug_output_dir)
            debug_logger.log_event(0, "Player", f"mode={mode}, evict_mode={evict_mode}, threshold={Threshold}, similarity_estimator={similarity_estimator_name}, lightglue_spatial_alpha={lightglue_spatial_alpha}, depth_metric={depth_metric}, depth_model={depth_model}")
            experiment_recorder = ExperimentRecorder(
                enabled=True,
                database_root=os.path.join(debug_output_dir, "experiment_tracking"),
                metadata={
                    "mode": mode,
                    "evict_mode": evict_mode,
                    "threshold": Threshold,
                    "similarity_estimator": similarity_estimator_name,
                    "lightglue_spatial_alpha": lightglue_spatial_alpha,
                    "depth_metric": depth_metric,
                    "depth_model": depth_model,
                    "depth_checkpoint": depth_checkpoint,
                    "depth_cache_size": depth_cache_size,
                    "num_output_frames": num_output_frames,
                    "num_frame_per_block": self.num_frame_per_block,
                },
            )

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache=None
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            if debug_logger is not None:
                debug_logger.log_event(0, "Memory", f"created visual/action/cross-attn KV caches, local_attn_size={self.local_attn_size}")
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block


        evict_middle=0
        all_num_frames = [self.num_frame_per_block] * num_blocks

        window_ids=[0,1,2,3,4,5,6,7,8]

        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            
        # Step 3
        for current_num_frames in tqdm(all_num_frames):
            generated_start_frame = current_start_frame
            if debug_logger is not None:
                debug_logger.log_event(current_start_frame, "Action", f"using precomputed action window for latent frames {current_start_frame}-{current_start_frame + current_num_frames - 1}")

            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            if profile:
                torch.cuda.synchronize()
                diffusion_start.record()
            valid_len = current_start_frame
            if valid_len >= len(window_ids) and evict_mode:
                if debug_logger is not None:
                    debug_logger.log_event(current_start_frame, "ORB", f"checking window={window_ids}")

                evict_middle, window_ids, sim_min = schedule_stableworld_window_tri_9(
                    window_ids=window_ids,
                    videos=videos,         
                    sim_threshold=Threshold,
                    debug_logger=debug_logger,
                    experiment_recorder=experiment_recorder,
                    frame_index=current_start_frame,
                    similarity_estimator_name=similarity_estimator_name,
                    lightglue_spatial_alpha=lightglue_spatial_alpha,
                    depth_metric=depth_metric,
                    depth_model=depth_model,
                    depth_checkpoint=depth_checkpoint,
                    depth_cache_size=depth_cache_size,
                )
                if debug_logger is not None:
                    debug_logger.log_event(current_start_frame, "Similarity", f"similarity={sim_min:.4f}")
                    debug_logger.log_event(current_start_frame, "Memory Decision", f"evict_middle={evict_middle}, updated_window={window_ids}")

            for index, current_timestep in enumerate(self.denoising_step_list):
                if debug_logger is not None and index == 0:
                    debug_logger.log_event(current_start_frame, "World Model", f"denoising latent block with {len(self.denoising_step_list)} steps")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        evict_middle=evict_middle
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        evict_middle=evict_middle
                    )

            B, C, F_blk, H, W = denoised_pred.shape   # 例如 [1,16,3,44,80]
            assert B == 1


            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.args.context_noise

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                evict_middle=evict_middle
            )
            if debug_logger is not None:
                debug_logger.log_event(current_start_frame, "KV Update", f"context pass wrote latent frames {current_start_frame}-{current_start_frame + current_num_frames - 1}, evict_middle={evict_middle}")

            current_start_frame += current_num_frames
            denoised_pred = denoised_pred.transpose(1,2)
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]
            if debug_logger is not None:
                try:
                    current_frame_id = generated_start_frame
                    current_img = get_decoded_frame_by_latent(videos, current_frame_id, sub=2)
                    current_path = os.path.join(debug_logger.frames_dir, f"frame_{generated_start_frame:04d}_current_{current_frame_id:04d}.png")
                    save_debug_frame(current_img, current_path)
                except Exception as exc:
                    debug_logger.log_event(current_start_frame, "Frame", f"failed to save current frame: {exc}")
                debug_logger.log_event(current_start_frame, "Frame", f"decoded latent block, total_decoded_blocks={len(videos)}")
                debug_logger.log_event(current_start_frame, "Next Frame", f"next_start_frame={current_start_frame}")

        if debug_logger is not None:
            debug_logger.save()
        if experiment_recorder is not None:
            experiment_recorder.close()

        if return_latents:
            return output
        else:
            return videos

    # def inference(
    #     self,
    #     noise: torch.Tensor,
    #     conditional_dict,
    #     initial_latent = None,
    #     return_latents = False,
    #     mode = 'universal',
    #     profile = False,
    # ) -> torch.Tensor:
    #     """
    #     Perform inference on the given noise and text prompts.
    #     Inputs:
    #         noise (torch.Tensor): The input noise tensor of shape
    #             (batch_size, num_output_frames, num_channels, height, width).
    #         text_prompts (List[str]): The list of text prompts.
    #         initial_latent (torch.Tensor): The initial latent tensor of shape
    #             (batch_size, num_input_frames, num_channels, height, width).
    #             If num_input_frames is 1, perform image to video.
    #             If num_input_frames is greater than 1, perform video extension.
    #         return_latents (bool): Whether to return the latents.
    #     Outputs:
    #         video (torch.Tensor): The generated video tensor of shape
    #             (batch_size, num_output_frames, num_channels, height, width).
    #             It is normalized to be in the range [0, 1].
    #     """
        
    #     assert noise.shape[1] == 16
    #     batch_size, num_channels, num_frames, height, width = noise.shape
        
    #     assert num_frames % self.num_frame_per_block == 0
    #     num_blocks = num_frames // self.num_frame_per_block

    #     num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
    #     num_output_frames = num_frames + num_input_frames  # add the initial latent frames

    #     output = torch.zeros(
    #         [batch_size, num_channels, num_output_frames, height, width],
    #         device=noise.device,
    #         dtype=noise.dtype
    #     )
    #     videos = []
    #     vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
    #     for j in range(len(vae_cache)):
    #         vae_cache[j] = None

    #     self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache=None
    #     # Step 1: Initialize KV cache to all zeros
    #     if self.kv_cache1 is None:
    #         self._initialize_kv_cache(
    #             batch_size=batch_size,
    #             dtype=noise.dtype,
    #             device=noise.device
    #         )
    #         self._initialize_kv_cache_mouse_and_keyboard(
    #             batch_size=batch_size,
    #             dtype=noise.dtype,
    #             device=noise.device
    #         )
            
    #         self._initialize_crossattn_cache(
    #             batch_size=batch_size,
    #             dtype=noise.dtype,
    #             device=noise.device
    #         )
    #     else:
    #         # reset cross attn cache
    #         for block_index in range(self.num_transformer_blocks):
    #             self.crossattn_cache[block_index]["is_init"] = False
    #         # reset kv cache
    #         for block_index in range(len(self.kv_cache1)):
    #             self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #     # Step 2: Cache context feature
    #     current_start_frame = 0
    #     if initial_latent is not None:
    #         timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
    #         # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
    #         assert num_input_frames % self.num_frame_per_block == 0
    #         num_input_blocks = num_input_frames // self.num_frame_per_block

    #         for _ in range(num_input_blocks):
    #             current_ref_latents = \
    #                 initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
    #             output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                
    #             self.generator(
    #                 noisy_image_or_video=current_ref_latents,
    #                 conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
    #                 timestep=timestep * 0,
    #                 kv_cache=self.kv_cache1,
    #                 kv_cache_mouse=self.kv_cache_mouse,
    #                 kv_cache_keyboard=self.kv_cache_keyboard,
    #                 crossattn_cache=self.crossattn_cache,
    #                 current_start=current_start_frame * self.frame_seq_length,
    #             )
    #             current_start_frame += self.num_frame_per_block


    #     # Step 3: Temporal denoising loop
    #     all_num_frames = [self.num_frame_per_block] * num_blocks
    #     if profile:
    #         diffusion_start = torch.cuda.Event(enable_timing=True)
    #         diffusion_end = torch.cuda.Event(enable_timing=True)
    #     for current_num_frames in tqdm(all_num_frames):

    #         noisy_input = noise[
    #             :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

    #         # Step 3.1: Spatial denoising loop
    #         if profile:
    #             torch.cuda.synchronize()
    #             diffusion_start.record()
    #         for index, current_timestep in enumerate(self.denoising_step_list):
    #             # set current timestep
    #             timestep = torch.ones(
    #                 [batch_size, current_num_frames],
    #                 device=noise.device,
    #                 dtype=torch.int64) * current_timestep

    #             if index < len(self.denoising_step_list) - 1:
    #                 _, denoised_pred = self.generator(
    #                     noisy_image_or_video=noisy_input,
    #                     conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
    #                     timestep=timestep,
    #                     kv_cache=self.kv_cache1,
    #                     kv_cache_mouse=self.kv_cache_mouse,
    #                     kv_cache_keyboard=self.kv_cache_keyboard,
    #                     crossattn_cache=self.crossattn_cache,
    #                     current_start=current_start_frame * self.frame_seq_length
    #                 )
    #                 next_timestep = self.denoising_step_list[index + 1]
    #                 noisy_input = self.scheduler.add_noise(
    #                     rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
    #                     torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
    #                     next_timestep * torch.ones(
    #                         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
    #                 )
    #                 noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
    #             else:
    #                 # for getting real output
    #                 _, denoised_pred = self.generator(
    #                     noisy_image_or_video=noisy_input,
    #                     conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
    #                     timestep=timestep,
    #                     kv_cache=self.kv_cache1,
    #                     kv_cache_mouse=self.kv_cache_mouse,
    #                     kv_cache_keyboard=self.kv_cache_keyboard,
    #                     crossattn_cache=self.crossattn_cache,
    #                     current_start=current_start_frame * self.frame_seq_length
    #                 )

    #         # Step 3.2: record the model's output
    #         output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

    #         # Step 3.3: rerun with timestep zero to update KV cache using clean context
    #         context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
    #         self.generator(
    #             noisy_image_or_video=denoised_pred,
    #             conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
    #             timestep=context_timestep,
    #             kv_cache=self.kv_cache1,
    #             kv_cache_mouse=self.kv_cache_mouse,
    #             kv_cache_keyboard=self.kv_cache_keyboard,
    #             crossattn_cache=self.crossattn_cache,
    #             current_start=current_start_frame * self.frame_seq_length,
    #         )

    #         # Step 3.4: update the start and end frame indices
    #         current_start_frame += current_num_frames

    #         denoised_pred = denoised_pred.transpose(1,2)
    #         video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
    #         videos += [video]

    #         if profile:
    #             torch.cuda.synchronize()
    #             diffusion_end.record()
    #             diffusion_time = diffusion_start.elapsed_time(diffusion_end)
    #             print(f"diffusion_time: {diffusion_time}", flush=True)
    #             fps = video.shape[1]*1000/ diffusion_time
    #             print(f"  - FPS: {fps:.2f}")

    #     if return_latents:
    #         return output
    #     else:
    #         return videos

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

        

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache


class CausalInferenceStreamingPipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device="cuda",
            vae_decoder=None,
            generator=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.vae_decoder = vae_decoder

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880 # 1590 # HW/4

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        output_folder = None,
        name = None,
        mode = 'universal'
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        
        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None
        # Set up profiling if requested
        self.kv_cache1=self.kv_cache_keyboard=self.kv_cache_mouse=self.crossattn_cache=None
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=True),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            current_actions = get_current_action(mode=mode)
            new_act, conditional_dict = cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=current_actions, mode=mode)
            # Step 3.1: Spatial denoising loop

            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=new_act,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Step 3.4: update the start and end frame indices
            denoised_pred = denoised_pred.transpose(1,2)
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]
            video = rearrange(video, "B T C H W -> B T H W C")
            video = ((video.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video = np.ascontiguousarray(video)
            mouse_icon = 'assets/images/mouse.png'
            if mode != 'templerun':
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                    conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                )
            else:
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy()
                )
            process_video(video.astype(np.uint8), output_folder+f'/{name}_current.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
            current_start_frame += current_num_frames

            if input("Continue? (Press `n` to break)").strip() == "n":
                break
                
        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)
        mouse_icon = 'assets/images/mouse.png'
        if mode != 'templerun':
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
            )
        else:
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy()
            )
        process_video(video.astype(np.uint8), output_folder+f'/{name}_icon.mp4', config, mouse_icon, mouse_scale=0.1, mode=mode)
        process_video(video.astype(np.uint8), output_folder+f'/{name}.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)

        if return_latents:
            return output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

        

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
