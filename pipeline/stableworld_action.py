from dataclasses import dataclass, asdict
import csv
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


@dataclass
class PoseState:
    frame_index: int
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    delta_x: float = 0.0
    delta_z: float = 0.0
    delta_yaw: float = 0.0
    delta_pitch: float = 0.0
    movement_magnitude: float = 0.0
    rotation_magnitude: float = 0.0
    pose_confidence: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "frame_index": int(self.frame_index),
            "pose_x": float(self.x),
            "pose_z": float(self.z),
            "pose_yaw": float(self.yaw),
            "pose_pitch": float(self.pitch),
            "delta_x": float(self.delta_x),
            "delta_z": float(self.delta_z),
            "delta_yaw": float(self.delta_yaw),
            "delta_pitch": float(self.delta_pitch),
            "movement_magnitude": float(self.movement_magnitude),
            "rotation_magnitude": float(self.rotation_magnitude),
            "pose_confidence": float(self.pose_confidence),
        }


@dataclass
class ActionState:
    frame_index: int
    intent_state: str
    intent_confidence: float
    rotation_speed: float
    movement_speed: float
    keyboard_vector: List[float]
    mouse_vector: List[float]
    pose_state: PoseState | None = None


class PoseTracker:
    def __init__(
        self,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        pitch_limit: float = 89.0,
    ):
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.pitch_limit = float(pitch_limit)
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, action_state: ActionState) -> PoseState:
        delta_x, delta_z = self._translation_delta(action_state.keyboard_vector, action_state.intent_state)
        delta_yaw, delta_pitch = self._rotation_delta(action_state.mouse_vector)
        self.x += delta_x
        self.z += delta_z
        self.yaw += delta_yaw
        self.pitch = float(np.clip(self.pitch + delta_pitch, -self.pitch_limit, self.pitch_limit))
        movement = float(np.sqrt(delta_x * delta_x + delta_z * delta_z))
        rotation = float(np.sqrt(delta_yaw * delta_yaw + delta_pitch * delta_pitch))
        confidence = float(np.clip(max(action_state.intent_confidence, min(1.0, movement + rotation)), 0.0, 1.0))
        return PoseState(
            frame_index=action_state.frame_index,
            x=float(self.x),
            z=float(self.z),
            yaw=float(self.yaw),
            pitch=float(self.pitch),
            delta_x=float(delta_x),
            delta_z=float(delta_z),
            delta_yaw=float(delta_yaw),
            delta_pitch=float(delta_pitch),
            movement_magnitude=movement,
            rotation_magnitude=rotation,
            pose_confidence=confidence,
        )

    def _translation_delta(self, keyboard_vec: List[float], intent_state: str) -> tuple[float, float]:
        if not keyboard_vec:
            return 0.0, 0.0
        speed = max(float(np.linalg.norm(keyboard_vec)), 0.0) * self.translation_scale
        if intent_state == "Run":
            speed *= 1.5
        if len(keyboard_vec) == 4:
            return (
                (float(keyboard_vec[3]) - float(keyboard_vec[2])) * self.translation_scale,
                (float(keyboard_vec[0]) - float(keyboard_vec[1])) * self.translation_scale,
            )
        if len(keyboard_vec) == 2:
            return 0.0, (float(keyboard_vec[0]) - float(keyboard_vec[1])) * self.translation_scale
        if len(keyboard_vec) == 7:
            return (float(keyboard_vec[6]) - float(keyboard_vec[5])) * self.translation_scale, 0.0
        if intent_state in {"Forward", "Walk", "Run"}:
            return 0.0, speed
        if intent_state == "Backward":
            return 0.0, -speed
        if intent_state == "Left":
            return -speed, 0.0
        if intent_state == "Right":
            return speed, 0.0
        return 0.0, 0.0

    def _rotation_delta(self, mouse_vec: List[float]) -> tuple[float, float]:
        if not mouse_vec:
            return 0.0, 0.0
        if len(mouse_vec) > 1:
            return float(mouse_vec[1]) * self.rotation_scale, float(mouse_vec[0]) * self.rotation_scale
        return float(mouse_vec[0]) * self.rotation_scale, 0.0


class ActionParser:
    def __init__(
        self,
        rotation_threshold: float = 0.02,
        movement_threshold: float = 0.1,
        run_threshold: float = 1.5,
    ):
        self.rotation_threshold = rotation_threshold
        self.movement_threshold = movement_threshold
        self.run_threshold = run_threshold

    @staticmethod
    def _mean_tensor(x: Optional[torch.Tensor]) -> List[float]:
        if x is None:
            return []
        y = x.detach().float().cpu()
        if y.ndim == 3:
            y = y[0]
        if y.ndim == 2:
            y = y.mean(dim=0)
        return y.numpy().astype(float).tolist()

    def parse(self, frame_index: int, keyboard: Optional[torch.Tensor], mouse: Optional[torch.Tensor]) -> ActionState:
        keyboard_vec = self._mean_tensor(keyboard)
        mouse_vec = self._mean_tensor(mouse)
        rotation_speed = float(np.linalg.norm(mouse_vec)) if mouse_vec else 0.0
        movement_speed = float(np.linalg.norm(keyboard_vec)) if keyboard_vec else 0.0

        intent = "Idle"
        confidence = 0.5

        if rotation_speed > self.rotation_threshold:
            dx = mouse_vec[1] if len(mouse_vec) > 1 else mouse_vec[0]
            intent = "Turn Right" if dx > 0 else "Turn Left"
            confidence = min(1.0, rotation_speed / max(self.rotation_threshold * 5.0, 1e-6))
        elif movement_speed > self.movement_threshold:
            dominant = int(np.argmax(np.abs(keyboard_vec))) if keyboard_vec else -1
            if len(keyboard_vec) == 4:
                intent = ["Forward", "Backward", "Left", "Right"][dominant]
            elif len(keyboard_vec) == 2:
                intent = ["Forward", "Backward"][dominant]
            elif len(keyboard_vec) == 7:
                intent = ["Idle", "Jump", "Unknown", "Turn Left", "Turn Right", "Left", "Right"][dominant]
            else:
                intent = "Walk"
            if movement_speed >= self.run_threshold:
                intent = "Run" if intent in {"Forward", "Walk"} else intent
            elif intent == "Forward":
                intent = "Walk"
            confidence = min(1.0, movement_speed / max(self.movement_threshold * 5.0, 1e-6))

        return ActionState(
            frame_index=int(frame_index),
            intent_state=intent,
            intent_confidence=float(confidence),
            rotation_speed=rotation_speed,
            movement_speed=movement_speed,
            keyboard_vector=keyboard_vec,
            mouse_vector=mouse_vec,
        )


class ActionBenchmark:
    @staticmethod
    def summarize(states: List[ActionState], bucket_size: int = 100) -> List[Dict]:
        buckets = {}
        for state in states:
            bucket = (state.frame_index // bucket_size) * bucket_size
            buckets.setdefault(bucket, []).append(state)

        rows = []
        for bucket in sorted(buckets):
            items = buckets[bucket]
            action_triggers = sum(1 for item in items if item.intent_state != "Idle")
            rows.append({
                "frame_bucket_start": bucket,
                "frame_bucket_end": bucket + bucket_size - 1,
                "action_triggers": int(action_triggers),
                "average_rotation": float(np.mean([x.rotation_speed for x in items])),
                "average_movement": float(np.mean([x.movement_speed for x in items])),
                "average_intent_confidence": float(np.mean([x.intent_confidence for x in items])),
            })
        return rows


class ActionVisualizer:
    @staticmethod
    def save(states: List[ActionState], output_path: str):
        if not states:
            return
        width = 1200
        height = 640
        margin_left = 80
        margin_right = 40
        plot_w = width - margin_left - margin_right
        canvas = np.zeros((height, width, 3), dtype=np.uint8) + 255

        xs = np.array([s.frame_index for s in states], dtype=np.float32)
        rotations = np.array([s.rotation_speed for s in states], dtype=np.float32)
        movements = np.array([s.movement_speed for s in states], dtype=np.float32)
        confidences = np.array([s.intent_confidence for s in states], dtype=np.float32)
        x_min = float(xs.min())
        x_max = float(max(xs.max(), x_min + 1.0))

        def sx(x):
            return int(margin_left + ((float(x) - x_min) / (x_max - x_min)) * plot_w)

        def draw_series(values, y_top, y_bottom, color, label):
            max_v = max(float(values.max()), 1e-6)
            pts = []
            for x, value in zip(xs, values):
                px = sx(x)
                py = int(y_bottom - (float(value) / max_v) * (y_bottom - y_top))
                pts.append((px, py))
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (20, y_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.line(canvas, (margin_left, y_bottom), (width - margin_right, y_bottom), (210, 210, 210), 1)

        draw_series(rotations, 50, 190, (220, 80, 40), "Mouse Rotation")
        draw_series(movements, 250, 390, (50, 150, 70), "Movement")
        draw_series(confidences, 450, 590, (80, 90, 220), "Intent Confidence")

        last_label_x = -999
        for state in states:
            x = sx(state.frame_index)
            if x - last_label_x < 80:
                continue
            last_label_x = x
            cv2.putText(canvas, state.intent_state[:12], (x, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

        cv2.imwrite(output_path, canvas)


class ActionIntentEngine:
    def __init__(self, bucket_size: int = 100):
        self.parser = ActionParser()
        self.pose_tracker = PoseTracker()
        self.bucket_size = bucket_size
        self.states: List[ActionState] = []

    def record(
        self,
        frame_index: int,
        conditional_dict: dict,
        current_start_frame: int,
        num_frame_per_block: int,
        mode: str = "universal",
    ) -> ActionState:
        end = 1 + 4 * (current_start_frame + num_frame_per_block - 1)
        start = max(0, end - 4 * num_frame_per_block)
        keyboard = conditional_dict.get("keyboard_cond", None)
        mouse = conditional_dict.get("mouse_cond", None) if mode != "templerun" else None
        keyboard_slice = keyboard[:, start:end] if keyboard is not None else None
        mouse_slice = mouse[:, start:end] if mouse is not None else None
        state = self.parser.parse(frame_index, keyboard_slice, mouse_slice)
        state.pose_state = self.pose_tracker.update(state)
        self.states.append(state)
        return state

    def save(self, output_dir: str):
        self.save_states(self.states, output_dir, self.bucket_size)

    @staticmethod
    def save_states(states: List[ActionState], output_dir: str, bucket_size: int = 100):
        os.makedirs(output_dir, exist_ok=True)
        timeline_path = os.path.join(output_dir, "action_timeline.csv")
        with open(timeline_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "frame_index",
                "intent_state",
                "intent_confidence",
                "rotation_speed",
                "movement_speed",
                "keyboard_vector",
                "mouse_vector",
                "pose_x",
                "pose_z",
                "pose_yaw",
                "pose_pitch",
                "delta_x",
                "delta_z",
                "delta_yaw",
                "delta_pitch",
                "pose_movement_magnitude",
                "pose_rotation_magnitude",
                "pose_confidence",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for state in states:
                row = asdict(state)
                row.pop("pose_state", None)
                row.pop("movement_magnitude", None)
                row.pop("rotation_magnitude", None)
                row["keyboard_vector"] = " ".join(f"{x:.6f}" for x in state.keyboard_vector)
                row["mouse_vector"] = " ".join(f"{x:.6f}" for x in state.mouse_vector)
                pose = state.pose_state or PoseState(frame_index=state.frame_index)
                row.update({
                    "pose_x": f"{pose.x:.6f}",
                    "pose_z": f"{pose.z:.6f}",
                    "pose_yaw": f"{pose.yaw:.6f}",
                    "pose_pitch": f"{pose.pitch:.6f}",
                    "delta_x": f"{pose.delta_x:.6f}",
                    "delta_z": f"{pose.delta_z:.6f}",
                    "delta_yaw": f"{pose.delta_yaw:.6f}",
                    "delta_pitch": f"{pose.delta_pitch:.6f}",
                    "pose_movement_magnitude": f"{pose.movement_magnitude:.6f}",
                    "pose_rotation_magnitude": f"{pose.rotation_magnitude:.6f}",
                    "pose_confidence": f"{pose.pose_confidence:.6f}",
                })
                writer.writerow(row)

        summary_path = os.path.join(output_dir, "action_summary_100f.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "frame_bucket_start",
                "frame_bucket_end",
                "action_triggers",
                "average_rotation",
                "average_movement",
                "average_intent_confidence",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in ActionBenchmark.summarize(states, bucket_size):
                writer.writerow(row)

        ActionVisualizer.save(states, os.path.join(output_dir, "action_timeline.png"))
