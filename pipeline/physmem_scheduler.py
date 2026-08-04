from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .stableworld_memory import MemoryBuffer, MemoryDecision, MemoryPolicy, MemoryScheduler
from .stableworld_similarity import SimilarityResult
from .stableworld_action import PoseState


@dataclass
class MemoryProposal:
    state: str
    source: str
    appearance_score: float
    confidence: float


@dataclass
class EvidenceValidation:
    semantic: str = "neutral"
    geometry: str = "neutral"
    intent: str = "neutral"
    semantic_consistency: float = 0.0
    geometry_consistency: float = 0.0
    intent_explanation: str = "unknown"
    world_change_probability: float = 0.0
    reasons: List[str] = field(default_factory=list)

    @property
    def support_count(self) -> int:
        return sum(1 for item in (self.semantic, self.geometry, self.intent) if item == "support")

    @property
    def reject_count(self) -> int:
        return sum(1 for item in (self.semantic, self.geometry, self.intent) if item == "reject")


@dataclass
class PoseValidation:
    event: str = "unknown"
    validation: str = "neutral"
    pose_distance: float = 0.0
    yaw_delta: float = 0.0
    nearest_frame_id: int | None = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class ViewState:
    state: str = "Unknown"
    confidence: float = 0.0
    pose_distance: float = 0.0
    yaw_delta: float = 0.0
    nearest_frame_id: int | None = None
    frame_gap: int = 0
    reasons: List[str] = field(default_factory=list)


@dataclass
class KeyPoseAnchor:
    frame_id: int
    pose: PoseState
    anchor_type: str
    view_state: str
    confidence: float
    reason: str = ""


@dataclass
class MemorySelection:
    mode: str = "current_window"
    target_frame_id: int | None = None
    anchor_type: str = ""
    reason: str = ""
    protected_frame_ids: List[int] = field(default_factory=list)
    target_window_ids: List[int] = field(default_factory=list)
    source_anchor_frame_id: int | None = None
    loop_closure_detected: bool = False
    turn_result: str = ""
    policy_mode: str = ""
    allow_memory_query: bool = False
    allow_hard_retrieve: bool = False
    allow_soft_reuse: bool = False
    query_result: str = ""
    candidate_count: int = 0
    best_candidate_frame_id: int | None = None
    best_candidate_score: float = 0.0
    best_candidate_pose_distance: float = 0.0
    best_candidate_yaw_delta: float = 0.0
    best_candidate_frame_gap: int = 0
    rank_reason: str = ""


@dataclass
class TrajectoryState:
    motion_state: str = "Idle"
    direction: str = "none"
    consecutive_motion_frames: int = 0
    accumulated_distance: float = 0.0
    yaw_stable: bool = True
    should_progress: bool = False
    confidence: float = 0.0
    reason: str = ""


@dataclass
class RevisitGateResult:
    allow_retrieve: bool = False
    allow_soft_reuse: bool = False
    force_progress: bool = False
    protect_current: bool = False
    reason: str = ""


@dataclass
class ActionModeState:
    mode: str = "Unknown"
    confidence: float = 0.0
    reason: str = ""


@dataclass
class TurnState:
    state: str = "Idle"
    confidence: float = 0.0
    duration: int = 0
    direction: str = "none"
    reason: str = ""


@dataclass
class MemoryCandidate:
    frame_id: int | None = None
    candidate_type: str = "none"
    pose_distance: float = 0.0
    yaw_delta: float = 0.0
    confidence: float = 0.0
    source: str = ""
    validated: bool = False
    reject_reason: str = ""


@dataclass
class SchedulingPolicy:
    policy_mode: str = "default"
    allow_memory_query: bool = False
    allow_hard_retrieve: bool = False
    allow_soft_reuse: bool = False
    allow_insert: bool = True
    allow_transition: bool = False
    allow_novel_anchor: bool = False
    reason: str = ""


@dataclass
class PosePathEntry:
    frame_id: int
    pose: PoseState
    window_ids: List[int]
    view_state: str
    action_mode: str
    turn_state: str
    memory_state: str
    selection_mode: str
    stability_score: float = 0.0


@dataclass
class HistoricalWindowCandidate:
    anchor_frame_id: int
    window_ids: List[int]
    pose_distance: float
    yaw_delta: float
    frame_gap: int
    view_state: str
    action_mode: str
    window_quality: float
    transition_penalty: float
    novel_penalty: float
    score: float
    reason: str = ""
    fidelity_score: float = 0.0
    stability_score: float = 0.0
    tier0_score: float = 0.0
    completeness: float = 0.0


class PoseMemory:
    def __init__(self, max_size: int = 512):
        self.max_size = int(max_size)
        self.poses: Dict[int, PoseState] = {}
        self.order: List[int] = []

    def insert(self, frame_id: int, pose: PoseState | None):
        if pose is None:
            return
        frame_id = int(frame_id)
        if frame_id not in self.poses:
            self.order.append(frame_id)
        self.poses[frame_id] = pose
        while len(self.order) > self.max_size:
            old = self.order.pop(0)
            self.poses.pop(old, None)

    def get(self, frame_id: int) -> PoseState | None:
        return self.poses.get(int(frame_id))

    def nearest(self, pose: PoseState | None) -> tuple[int | None, PoseState | None, float]:
        if pose is None or not self.poses:
            return None, None, float("inf")
        best_id = None
        best_pose = None
        best_distance = float("inf")
        for frame_id, candidate in self.poses.items():
            distance = self.distance(pose, candidate)
            if distance < best_distance:
                best_id = frame_id
                best_pose = candidate
                best_distance = distance
        return best_id, best_pose, float(best_distance)

    @staticmethod
    def distance(a: PoseState | None, b: PoseState | None) -> float:
        if a is None or b is None:
            return float("inf")
        dx = float(a.x) - float(b.x)
        dz = float(a.z) - float(b.z)
        return float(np.sqrt(dx * dx + dz * dz))

    @staticmethod
    def yaw_delta(a: PoseState | None, b: PoseState | None) -> float:
        if a is None or b is None:
            return float("inf")
        raw = abs(float(a.yaw) - float(b.yaw)) % (2.0 * np.pi)
        return float(min(raw, 2.0 * np.pi - raw))


@dataclass
class StrategyStabilityConfig:
    replace_cooldown: int = 5
    evict_cooldown: int = 8
    insert_enter_probability: float = 0.35
    replace_enter_probability: float = 0.55
    replace_exit_probability: float = 0.40
    evict_enter_probability: float = 0.80
    evict_exit_probability: float = 0.65
    allow_direct_keep_to_evict: bool = False
    viewpoint_rotation_threshold: float = 0.15
    viewpoint_translation_threshold: float = 0.35
    forward_progress_threshold: float = 0.20
    yaw_stable_threshold: float = 0.20
    revisit_distance_threshold: float = 0.75
    revisit_yaw_threshold: float = 0.35
    known_view_distance_threshold: float = 0.85
    known_view_yaw_threshold: float = 0.45
    novel_view_distance_threshold: float = 1.75
    novel_view_yaw_threshold: float = 0.95
    view_transition_rotation_threshold: float = 0.15
    revisit_min_frame_gap: int = 12
    key_anchor_min_frame_gap: int = 6
    key_anchor_distance_threshold: float = 0.75
    key_anchor_yaw_threshold: float = 0.35
    trajectory_progress_min_frames: int = 2
    trajectory_progress_distance: float = 0.25
    trajectory_reset_rotation_threshold: float = 0.35
    turn_start_frames: int = 1
    turn_stabilization_frames: int = 2
    turn_rotation_threshold: float = 0.02


@dataclass
class HierarchicalRankingConfig:
    """v4.6 (M2) hierarchical pose-priority historical-window retrieval settings.

    - Stage 1 Yaw Gate: hard filter on |yaw delta| (view direction must match).
    - Stage 2 Position Ranking: keep the top-k nearest candidates.
    - Stage 3 Fidelity Ranking: frame age / Tier0 / generation quality.
    - Stage 4 Window Stability: completeness + historical state stability.
    """
    yaw_gate_threshold: float = 0.35
    position_top_k: int = 8
    max_age: int = 150
    age_weight: float = 0.5
    tier0_weight: float = 0.3
    quality_weight: float = 0.2
    w_view: float = 0.25
    w_pose: float = 0.20
    w_fidelity: float = 0.30
    w_stability: float = 0.25
    transition_penalty: float = 0.10
    novel_penalty: float = 0.10
    pose_scale: float = 0.75
    min_score: float = 0.35
    tier0_enabled: bool = True


class KeyPoseMemory:
    def __init__(self, max_size: int = 64, stability: StrategyStabilityConfig | None = None):
        self.max_size = int(max_size)
        self.stability = stability or StrategyStabilityConfig()
        self.anchors: Dict[int, KeyPoseAnchor] = {}
        self.order: List[int] = []

    def insert(self, frame_id: int, pose: PoseState | None, anchor_type: str, view_state: ViewState, reason: str = "") -> KeyPoseAnchor | None:
        if pose is None:
            return None
        frame_id = int(frame_id)
        existing_id, existing_anchor, existing_distance = self.nearest(pose)
        existing_yaw = PoseMemory.yaw_delta(pose, existing_anchor.pose) if existing_anchor is not None else float("inf")
        frame_gap = frame_id - int(existing_id) if existing_id is not None else self.stability.key_anchor_min_frame_gap
        is_distinct = (
            existing_id is None
            or frame_gap >= self.stability.key_anchor_min_frame_gap
            and (
                existing_distance >= self.stability.key_anchor_distance_threshold
                or existing_yaw >= self.stability.key_anchor_yaw_threshold
                or anchor_type == "revisit_anchor"
            )
        )
        if not is_distinct:
            return None

        anchor = KeyPoseAnchor(
            frame_id=frame_id,
            pose=pose,
            anchor_type=anchor_type,
            view_state=view_state.state,
            confidence=float(view_state.confidence),
            reason=reason,
        )
        if frame_id not in self.anchors:
            self.order.append(frame_id)
        self.anchors[frame_id] = anchor
        while len(self.order) > self.max_size:
            old = self.order.pop(0)
            self.anchors.pop(old, None)
        return anchor

    def maybe_insert(
        self,
        frame_id: int,
        pose: PoseState | None,
        view_state: ViewState,
        pose_validation: PoseValidation,
        memory_state: str,
        revisit_gate: RevisitGateResult | None = None,
        action_mode: ActionModeState | None = None,
        turn_state: TurnState | None = None,
    ) -> KeyPoseAnchor | None:
        if pose is None:
            return None
        revisit_gate = revisit_gate or RevisitGateResult()
        action_mode = action_mode or ActionModeState()
        turn_state = turn_state or TurnState()
        if not self.anchors:
            return self.insert(frame_id, pose, "initial_anchor", view_state, "first_pose")
        if action_mode.mode == "Locomotion Only":
            return None
        if turn_state.state == "PostTurnStabilization" and memory_state == "INSERT":
            return self.insert(frame_id, pose, "stabilized_view_anchor", view_state, "post_turn_stabilization")
        if revisit_gate.force_progress and memory_state == "INSERT":
            return self.insert(frame_id, pose, "forward_anchor", view_state, revisit_gate.reason)
        if view_state.state == "Novel View" and memory_state == "INSERT":
            return self.insert(frame_id, pose, "novel_view_anchor", view_state, "novel_view_insert")
        if view_state.state == "Revisit" and revisit_gate.allow_retrieve:
            return self.insert(frame_id, pose, "revisit_anchor", view_state, "pose_revisit")
        if view_state.state == "View Transition":
            return self.insert(frame_id, pose, "turn_anchor", view_state, "view_transition")
        if pose_validation.event in {"forward_progression", "lateral_motion"} and memory_state in {"KEEP", "INSERT"}:
            return self.insert(frame_id, pose, "forward_anchor", view_state, pose_validation.event)
        return None

    def nearest(self, pose: PoseState | None, anchor_types: set[str] | None = None) -> tuple[int | None, KeyPoseAnchor | None, float]:
        if pose is None or not self.anchors:
            return None, None, float("inf")
        best_id = None
        best_anchor = None
        best_distance = float("inf")
        for frame_id, anchor in self.anchors.items():
            if anchor_types is not None and anchor.anchor_type not in anchor_types:
                continue
            distance = PoseMemory.distance(pose, anchor.pose)
            if distance < best_distance:
                best_id = frame_id
                best_anchor = anchor
                best_distance = distance
        return best_id, best_anchor, float(best_distance)


class MemoryPolicySelector:
    def select(
        self,
        action_mode: ActionModeState,
        turn_state: TurnState,
        view_state: ViewState,
        trajectory_state: TrajectoryState,
        revisit_gate: RevisitGateResult,
    ) -> SchedulingPolicy:
        # M1 (v4.6): a stable revisit outranks locomotion / turn protection,
        # so "turned back to a previously seen pose" can actually query memory.
        if view_state.state == "Revisit" and revisit_gate.allow_retrieve:
            return SchedulingPolicy(
                policy_mode="RevisitPolicy",
                allow_memory_query=True,
                allow_hard_retrieve=True,
                allow_soft_reuse=True,
                reason=f"stable_revisit_priority:{revisit_gate.reason or 'revisit'}",
            )
        if action_mode.mode == "Locomotion Only":
            return SchedulingPolicy(
                policy_mode="LocomotionPolicy",
                allow_memory_query=False,
                allow_insert=True,
                reason="locomotion_orb_dominant",
            )
        if turn_state.state == "TurnStart":
            return SchedulingPolicy(
                policy_mode="TurnStartPolicy",
                allow_memory_query=False,
                allow_transition=True,
                reason="protect_source_window",
            )
        if turn_state.state == "TurnInProgress":
            return SchedulingPolicy(
                policy_mode="ViewTransitionPolicy",
                allow_memory_query=False,
                allow_transition=True,
                reason="transition_no_retrieve",
            )
        if turn_state.state == "PostTurnStabilization":
            allow_hard = view_state.state == "Revisit"
            return SchedulingPolicy(
                policy_mode="PostTurnPolicy",
                allow_memory_query=view_state.state in {"Known View", "Revisit"},
                allow_hard_retrieve=allow_hard,
                allow_soft_reuse=view_state.state in {"Known View", "Revisit"},
                allow_insert=True,
                allow_novel_anchor=view_state.state == "Novel View",
                reason=f"post_turn:{view_state.state}",
            )
        if view_state.state == "Revisit":
            return SchedulingPolicy(
                policy_mode="RevisitPolicy",
                allow_memory_query=True,
                allow_hard_retrieve=True,
                allow_soft_reuse=True,
                reason=revisit_gate.reason or "revisit",
            )
        if view_state.state == "Known View":
            return SchedulingPolicy(
                policy_mode="KnownViewPolicy",
                allow_memory_query=True,
                allow_soft_reuse=True,
                reason=revisit_gate.reason or "known_view",
            )
        if view_state.state == "Novel View":
            return SchedulingPolicy(
                policy_mode="NovelViewPolicy",
                allow_memory_query=False,
                allow_insert=True,
                allow_novel_anchor=True,
                reason="novel_view_no_retrieve",
            )
        if view_state.state == "View Transition":
            return SchedulingPolicy(
                policy_mode="ViewTransitionPolicy",
                allow_memory_query=False,
                allow_transition=True,
                reason="view_transition_no_retrieve",
            )
        if trajectory_state.should_progress:
            return SchedulingPolicy(
                policy_mode="ProgressPolicy",
                allow_memory_query=False,
                allow_insert=True,
                reason=trajectory_state.reason,
            )
        return SchedulingPolicy(policy_mode="DefaultPolicy", reason=f"view={view_state.state}:mode={action_mode.mode}")


class PosePathMemory:
    def __init__(self, max_size: int = 512):
        self.max_size = int(max_size)
        self.entries: Dict[int, PosePathEntry] = {}
        self.order: List[int] = []

    def insert(
        self,
        frame_id: int,
        pose: PoseState | None,
        window_ids: list[int],
        view_state: ViewState,
        action_mode: ActionModeState,
        turn_state: TurnState,
        memory_state: str,
        memory_selection: MemorySelection,
        stability_score: float,
    ) -> None:
        if pose is None:
            return
        frame_id = int(frame_id)
        if frame_id not in self.entries:
            self.order.append(frame_id)
        self.entries[frame_id] = PosePathEntry(
            frame_id=frame_id,
            pose=pose,
            window_ids=list(window_ids),
            view_state=view_state.state,
            action_mode=action_mode.mode,
            turn_state=turn_state.state,
            memory_state=memory_state,
            selection_mode=memory_selection.mode,
            stability_score=float(stability_score),
        )
        while len(self.order) > self.max_size:
            old = self.order.pop(0)
            self.entries.pop(old, None)

    def candidates(
        self,
        pose: PoseState | None,
        current_frame_id: int,
        stability: StrategyStabilityConfig,
        max_candidates: int = 16,
        initial_window_ids: list[int] | None = None,
        ranking_config: HierarchicalRankingConfig | None = None,
    ) -> list[HistoricalWindowCandidate]:
        if pose is None:
            return []
        ranking = ranking_config or HierarchicalRankingConfig()
        initial_set = {int(fid) for fid in (initial_window_ids or [])}
        candidates: list[HistoricalWindowCandidate] = []
        for entry in self.entries.values():
            frame_gap = int(current_frame_id) - int(entry.frame_id)
            if frame_gap <= 0:
                continue
            pose_distance = PoseMemory.distance(pose, entry.pose)
            yaw_delta = PoseMemory.yaw_delta(pose, entry.pose)
            # M2 Stage 1: Yaw Gate (hard constraint) -- view direction must match.
            if yaw_delta > ranking.yaw_gate_threshold:
                continue
            transition_penalty = 1.0 if entry.view_state == "View Transition" or entry.turn_state == "TurnInProgress" else 0.0
            novel_penalty = 0.5 if entry.view_state == "Novel View" and entry.turn_state != "StableNewView" else 0.0
            quality = float(np.clip(entry.stability_score, 0.0, 1.0))
            window_ids = list(entry.window_ids)
            completeness = (
                sum(1 for fid in window_ids if int(fid) < int(current_frame_id)) / len(window_ids)
                if window_ids else 0.0
            )
            tier0_score = (
                float(1.0)
                if ranking.tier0_enabled and initial_set and any(int(fid) in initial_set for fid in window_ids[:3])
                else 0.0
            )
            age_score = min(frame_gap / max(float(ranking.max_age), 1.0), 1.0)
            # M2 Stage 3 (pre-computed here): Fidelity = age + Tier0 + generation quality.
            fidelity_score = float(
                np.clip(
                    ranking.age_weight * age_score
                    + ranking.tier0_weight * tier0_score
                    + ranking.quality_weight * quality,
                    0.0,
                    1.0,
                )
            )
            # M2 Stage 4 (pre-computed here): Window Stability = completeness + state stability.
            state_stability = 1.0 if entry.memory_state in {"KEEP", "INSERT"} else 0.5
            stability_score = float(np.clip(0.5 * completeness + 0.5 * state_stability, 0.0, 1.0))
            candidates.append(
                HistoricalWindowCandidate(
                    anchor_frame_id=entry.frame_id,
                    window_ids=window_ids,
                    pose_distance=float(pose_distance),
                    yaw_delta=float(yaw_delta),
                    frame_gap=int(frame_gap),
                    view_state=entry.view_state,
                    action_mode=entry.action_mode,
                    window_quality=quality,
                    transition_penalty=float(transition_penalty),
                    novel_penalty=float(novel_penalty),
                    score=float(pose_distance),
                    reason=f"pose={pose_distance:.4f}:yaw={yaw_delta:.4f}:gap={frame_gap}:view={entry.view_state}",
                    fidelity_score=float(fidelity_score),
                    stability_score=float(stability_score),
                    tier0_score=float(tier0_score),
                    completeness=float(completeness),
                )
            )
        # M2 Stage 2: Position Ranking -> keep the top-k nearest candidates.
        candidates.sort(key=lambda item: item.pose_distance)
        top_k = max(1, min(int(ranking.position_top_k), max_candidates))
        return candidates[:top_k]


class HistoricalWindowRanker:
    def __init__(self, min_score: float | None = None):
        # None -> fall back to HierarchicalRankingConfig.min_score.
        self.min_score = min_score

    def best(
        self,
        candidates: list[HistoricalWindowCandidate],
        ranking_config: HierarchicalRankingConfig | None = None,
        initial_window_ids: list[int] | None = None,
    ) -> HistoricalWindowCandidate | None:
        if not candidates:
            return None
        ranking = ranking_config or HierarchicalRankingConfig()
        threshold = ranking.min_score if self.min_score is None else float(self.min_score)
        yaw_gate = ranking.yaw_gate_threshold
        pose_scale = ranking.pose_scale
        for candidate in candidates:
            s_view = float(np.clip(1.0 - candidate.yaw_delta / max(yaw_gate, 1e-6), 0.0, 1.0))
            s_pose = float(np.clip(1.0 - candidate.pose_distance / max(pose_scale, 1e-6), 0.0, 1.0))
            final_score = (
                ranking.w_view * s_view
                + ranking.w_pose * s_pose
                + ranking.w_fidelity * candidate.fidelity_score
                + ranking.w_stability * candidate.stability_score
                - ranking.transition_penalty * candidate.transition_penalty
                - ranking.novel_penalty * candidate.novel_penalty
            )
            candidate.score = float(np.clip(final_score, 0.0, 1.0))
            candidate.reason = (
                f"yaw={candidate.yaw_delta:.4f}:pose={candidate.pose_distance:.4f}:"
                f"gap={candidate.frame_gap}:fid={candidate.fidelity_score:.3f}:"
                f"stab={candidate.stability_score:.3f}:tier0={candidate.tier0_score:.1f}:"
                f"score={candidate.score:.3f}"
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        best_candidate = candidates[0]
        if best_candidate.score < threshold:
            return None
        return best_candidate


class TrajectoryTracker:
    def __init__(self, stability: StrategyStabilityConfig):
        self.stability = stability
        self.last_direction = "none"
        self.consecutive_motion_frames = 0
        self.accumulated_distance = 0.0

    def update(self, pose_state: PoseState | None) -> TrajectoryState:
        if pose_state is None:
            self._reset()
            return TrajectoryState(reason="pose_missing")

        local_x = float(getattr(pose_state, "local_delta_x", pose_state.delta_x))
        local_z = float(getattr(pose_state, "local_delta_z", pose_state.delta_z))
        delta_yaw = abs(float(pose_state.delta_yaw))
        distance = float(np.sqrt(local_x * local_x + local_z * local_z))
        direction = self._dominant_direction(local_x, local_z, distance)

        if direction == "none" or delta_yaw >= self.stability.trajectory_reset_rotation_threshold:
            self._reset()
            return TrajectoryState(
                motion_state="Idle" if direction == "none" else "View Motion",
                direction=direction,
                yaw_stable=delta_yaw < self.stability.trajectory_reset_rotation_threshold,
                reason=f"reset:direction={direction}:delta_yaw={delta_yaw:.4f}",
            )

        if direction == self.last_direction:
            self.consecutive_motion_frames += 1
            self.accumulated_distance += distance
        else:
            self.last_direction = direction
            self.consecutive_motion_frames = 1
            self.accumulated_distance = distance

        should_progress = (
            self.consecutive_motion_frames >= self.stability.trajectory_progress_min_frames
            and self.accumulated_distance >= self.stability.trajectory_progress_distance
        )
        confidence = min(
            1.0,
            0.5 * min(self.consecutive_motion_frames / max(self.stability.trajectory_progress_min_frames, 1), 1.0)
            + 0.5 * min(self.accumulated_distance / max(self.stability.trajectory_progress_distance, 1e-6), 1.0),
        )
        return TrajectoryState(
            motion_state="Progressing" if should_progress else "Starting",
            direction=direction,
            consecutive_motion_frames=int(self.consecutive_motion_frames),
            accumulated_distance=float(self.accumulated_distance),
            yaw_stable=True,
            should_progress=bool(should_progress),
            confidence=float(confidence),
            reason=f"direction={direction}:distance={distance:.4f}",
        )

    def _reset(self):
        self.last_direction = "none"
        self.consecutive_motion_frames = 0
        self.accumulated_distance = 0.0

    @staticmethod
    def _dominant_direction(local_x: float, local_z: float, distance: float) -> str:
        if distance <= 1e-6:
            return "none"
        if abs(local_z) >= abs(local_x):
            return "forward" if local_z > 0 else "backward"
        return "right" if local_x > 0 else "left"


class ActionModeClassifier:
    def classify(self, pose_state: PoseState | None, intent_state: str = "Unknown") -> ActionModeState:
        if pose_state is None:
            return ActionModeState(mode="Unknown", confidence=0.0, reason="pose_missing")
        movement = float(getattr(pose_state, "movement_magnitude", 0.0) or 0.0)
        rotation = float(getattr(pose_state, "rotation_magnitude", 0.0) or 0.0)
        has_move = movement > 0.10
        has_view = rotation > 0.02
        if has_move and has_view:
            return ActionModeState(mode="Viewpoint Locomotion", confidence=float(np.clip(max(movement, rotation), 0.0, 1.0)), reason=f"move={movement:.4f}:view={rotation:.4f}:{intent_state}")
        if has_move:
            return ActionModeState(mode="Locomotion Only", confidence=float(np.clip(movement, 0.0, 1.0)), reason=f"move={movement:.4f}:{intent_state}")
        if has_view:
            return ActionModeState(mode="View Rotation Only", confidence=float(np.clip(rotation / 0.18, 0.0, 1.0)), reason=f"view={rotation:.4f}:{intent_state}")
        return ActionModeState(mode="Idle", confidence=1.0, reason=f"idle:{intent_state}")


class TurnStateMachine:
    def __init__(self, stability: StrategyStabilityConfig):
        self.stability = stability
        self.turn_duration = 0
        self.stabilization_remaining = 0
        self.last_direction = "none"

    def update(self, action_mode: ActionModeState, pose_state: PoseState | None) -> TurnState:
        rotation = float(getattr(pose_state, "rotation_magnitude", 0.0) or 0.0) if pose_state is not None else 0.0
        delta_yaw = float(getattr(pose_state, "delta_yaw", 0.0) or 0.0) if pose_state is not None else 0.0
        direction = "right" if delta_yaw > 0 else "left" if delta_yaw < 0 else self.last_direction
        is_turning = action_mode.mode in {"View Rotation Only", "Viewpoint Locomotion"} and rotation >= self.stability.turn_rotation_threshold

        if is_turning:
            self.turn_duration += 1
            self.stabilization_remaining = self.stability.turn_stabilization_frames
            self.last_direction = direction
            if self.turn_duration <= self.stability.turn_start_frames:
                return TurnState(
                    state="TurnStart",
                    confidence=float(np.clip(rotation / max(self.stability.turn_rotation_threshold * 4.0, 1e-6), 0.0, 1.0)),
                    duration=int(self.turn_duration),
                    direction=direction,
                    reason=f"rotation={rotation:.4f}",
                )
            return TurnState(
                state="TurnInProgress",
                confidence=float(np.clip(rotation / max(self.stability.turn_rotation_threshold * 4.0, 1e-6), 0.0, 1.0)),
                duration=int(self.turn_duration),
                direction=direction,
                reason=f"rotation={rotation:.4f}",
            )

        if self.stabilization_remaining > 0:
            self.stabilization_remaining -= 1
            self.turn_duration = 0
            return TurnState(
                state="PostTurnStabilization",
                confidence=float((self.stabilization_remaining + 1) / max(self.stability.turn_stabilization_frames, 1)),
                duration=int(self.stability.turn_stabilization_frames - self.stabilization_remaining),
                direction=self.last_direction,
                reason="mouse_stopped_after_turn",
            )

        self.turn_duration = 0
        return TurnState(state="StableNewView" if self.last_direction != "none" else "Idle", confidence=1.0, direction=self.last_direction, reason="stable")


class PoseAwareMemorySelector:
    def __init__(self, stability: StrategyStabilityConfig):
        self.stability = stability

    def select(
        self,
        view_state: ViewState,
        key_pose_memory: KeyPoseMemory,
        pose_state: PoseState | None,
        revisit_gate: RevisitGateResult | None = None,
        action_mode: ActionModeState | None = None,
        turn_state: TurnState | None = None,
        source_anchor: KeyPoseAnchor | None = None,
        fallback_source_frame_id: int | None = None,
        scheduling_policy: SchedulingPolicy | None = None,
        pose_path_memory: PosePathMemory | None = None,
        historical_ranker: HistoricalWindowRanker | None = None,
        current_frame_id: int | None = None,
        use_pose_path_memory: bool = True,
        ranking_config: HierarchicalRankingConfig | None = None,
        initial_window_ids: list[int] | None = None,
    ) -> MemorySelection:
        if pose_state is None:
            return MemorySelection(reason="pose_missing")

        revisit_gate = revisit_gate or RevisitGateResult()
        action_mode = action_mode or ActionModeState()
        turn_state = turn_state or TurnState()
        scheduling_policy = scheduling_policy or SchedulingPolicy()
        source_anchor_id = source_anchor.frame_id if source_anchor is not None else fallback_source_frame_id
        protected_ids = [int(source_anchor_id)] if source_anchor_id is not None else []
        policy_kwargs = {
            "policy_mode": scheduling_policy.policy_mode,
            "allow_memory_query": scheduling_policy.allow_memory_query,
            "allow_hard_retrieve": scheduling_policy.allow_hard_retrieve,
            "allow_soft_reuse": scheduling_policy.allow_soft_reuse,
        }

        # M1 (v4.6): stable revisit must be able to query memory even when the
        # frame also looks like locomotion / turn-in-progress.
        stable_revisit = (
            view_state.state == "Revisit"
            and revisit_gate.allow_retrieve
            and scheduling_policy.allow_memory_query
        )
        if stable_revisit:
            query = self._query_memory_selection(
                view_state,
                pose_state,
                pose_path_memory,
                historical_ranker,
                current_frame_id,
                scheduling_policy,
                protected_ids,
                source_anchor_id,
                source_anchor,
                key_pose_memory,
                turn_state,
                use_pose_path_memory,
                ranking_config=ranking_config,
                initial_window_ids=initial_window_ids,
            )
            if query is not None:
                return query
            # no usable historical window -> fall through to ordinary mode logic

        if action_mode.mode == "Locomotion Only":
            return MemorySelection(mode="orb_locomotion", reason=action_mode.reason, **policy_kwargs)
        if turn_state.state == "TurnStart":
            return MemorySelection(
                mode="protect_current",
                reason=action_mode.reason,
                protected_frame_ids=protected_ids,
                source_anchor_frame_id=source_anchor_id,
                turn_result="TurnStart",
                **policy_kwargs,
            )

        if scheduling_policy.allow_memory_query:
            query = self._query_memory_selection(
                view_state,
                pose_state,
                pose_path_memory,
                historical_ranker,
                current_frame_id,
                scheduling_policy,
                protected_ids,
                source_anchor_id,
                source_anchor,
                key_pose_memory,
                turn_state,
                use_pose_path_memory,
                ranking_config=ranking_config,
                initial_window_ids=initial_window_ids,
            )
            if query is not None:
                return query

        if turn_state.state == "TurnInProgress":
            return MemorySelection(
                mode="turn_transition",
                reason=turn_state.reason,
                protected_frame_ids=protected_ids,
                source_anchor_frame_id=source_anchor_id,
                turn_result="UnstableTransition",
                **policy_kwargs,
            )
        if turn_state.state == "PostTurnStabilization":
            return MemorySelection(
                mode="stabilize_view",
                reason=turn_state.reason,
                protected_frame_ids=protected_ids,
                source_anchor_frame_id=source_anchor_id,
                turn_result="NovelViewCandidate",
                **policy_kwargs,
            )
        if turn_state.state == "StableNewView" and action_mode.mode == "Idle":
            return MemorySelection(
                mode="stable_new_view",
                reason=turn_state.reason,
                protected_frame_ids=protected_ids,
                source_anchor_frame_id=source_anchor_id,
                turn_result="StableNewView",
                **policy_kwargs,
            )
        if revisit_gate.force_progress and action_mode.mode == "Viewpoint Locomotion":
            return MemorySelection(mode="hybrid_locomotion", reason=revisit_gate.reason, **policy_kwargs)
        if revisit_gate.force_progress:
            return MemorySelection(mode="progress_anchor", reason=revisit_gate.reason, **policy_kwargs)
        if revisit_gate.protect_current:
            return MemorySelection(mode="protect_current", reason=revisit_gate.reason, protected_frame_ids=protected_ids, source_anchor_frame_id=source_anchor_id, **policy_kwargs)

        nearest_id, nearest_anchor, nearest_distance = key_pose_memory.nearest(pose_state)
        if view_state.state == "Revisit" and nearest_anchor is not None and revisit_gate.allow_retrieve:
            return MemorySelection(
                mode="hard_retrieve_anchor",
                target_frame_id=nearest_anchor.frame_id,
                anchor_type=nearest_anchor.anchor_type,
                reason=f"revisit_nearest_anchor:{nearest_distance:.4f}",
                protected_frame_ids=protected_ids + [int(nearest_anchor.frame_id)],
                source_anchor_frame_id=source_anchor_id,
                loop_closure_detected=True,
                turn_result="ReturnToKnownView",
                **policy_kwargs,
            )

        if view_state.state == "Known View" and nearest_anchor is not None and revisit_gate.allow_soft_reuse:
            yaw_delta = PoseMemory.yaw_delta(pose_state, nearest_anchor.pose)
            if nearest_distance <= self.stability.known_view_distance_threshold and yaw_delta <= self.stability.known_view_yaw_threshold:
                return MemorySelection(
                    mode="soft_reuse_anchor",
                    target_frame_id=nearest_anchor.frame_id,
                    anchor_type=nearest_anchor.anchor_type,
                    reason=f"known_view_anchor:{nearest_distance:.4f}:{yaw_delta:.4f}",
                    protected_frame_ids=protected_ids + [int(nearest_anchor.frame_id)],
                    source_anchor_frame_id=source_anchor_id,
                    loop_closure_detected=True,
                    turn_result="ReturnToKnownView",
                    **policy_kwargs,
                )

        if view_state.state == "Novel View":
            return MemorySelection(
                mode="create_anchor",
                reason="novel_view",
                protected_frame_ids=protected_ids,
                source_anchor_frame_id=source_anchor_id,
                turn_result="NovelViewArrival",
                **policy_kwargs,
            )

        if view_state.state == "View Transition":
            return MemorySelection(
                mode="protect_current",
                reason="view_transition",
                protected_frame_ids=protected_ids,
                source_anchor_frame_id=source_anchor_id,
                turn_result="UnstableTransition",
                **policy_kwargs,
            )

        return MemorySelection(
            reason=f"view_state={view_state.state}",
            protected_frame_ids=protected_ids,
            source_anchor_frame_id=source_anchor_id,
            **policy_kwargs,
        )

    def _query_memory_selection(
        self,
        view_state: ViewState,
        pose_state: PoseState,
        pose_path_memory: PosePathMemory | None,
        historical_ranker: HistoricalWindowRanker | None,
        current_frame_id: int | None,
        scheduling_policy: SchedulingPolicy,
        protected_ids: list[int],
        source_anchor_id: int | None,
        source_anchor: KeyPoseAnchor | None,
        key_pose_memory: KeyPoseMemory,
        turn_state: TurnState,
        use_pose_path_memory: bool,
        ranking_config: HierarchicalRankingConfig | None = None,
        initial_window_ids: list[int] | None = None,
    ) -> MemorySelection | None:
        if use_pose_path_memory:
            window_selection = self._pose_path_window_selection(
                view_state,
                pose_state,
                pose_path_memory,
                historical_ranker,
                current_frame_id,
                scheduling_policy,
                protected_ids,
                source_anchor_id,
                ranking_config=ranking_config,
                initial_window_ids=initial_window_ids,
            )
            if window_selection is not None:
                return window_selection
        loop_selection = self._loop_closure_selection(
            view_state,
            pose_state,
            source_anchor,
            key_pose_memory,
            turn_state,
            scheduling_policy,
        )
        if loop_selection is not None:
            loop_selection.protected_frame_ids = list(dict.fromkeys(protected_ids + loop_selection.protected_frame_ids))
            loop_selection.source_anchor_frame_id = source_anchor_id
            return loop_selection
        return None

    def _pose_path_window_selection(
        self,
        view_state: ViewState,
        pose_state: PoseState,
        pose_path_memory: PosePathMemory | None,
        historical_ranker: HistoricalWindowRanker | None,
        current_frame_id: int | None,
        scheduling_policy: SchedulingPolicy,
        protected_ids: list[int],
        source_anchor_id: int | None,
        ranking_config: HierarchicalRankingConfig | None = None,
        initial_window_ids: list[int] | None = None,
    ) -> MemorySelection | None:
        if pose_path_memory is None or historical_ranker is None or current_frame_id is None:
            return None
        candidates = pose_path_memory.candidates(
            pose_state,
            current_frame_id,
            self.stability,
            initial_window_ids=initial_window_ids,
            ranking_config=ranking_config,
        )
        best = historical_ranker.best(
            candidates,
            ranking_config=ranking_config,
            initial_window_ids=initial_window_ids,
        )
        if best is None:
            return None
        mode = "retrieve_window" if scheduling_policy.allow_hard_retrieve or view_state.state == "Revisit" else "soft_reuse_window"
        return MemorySelection(
            mode=mode,
            target_frame_id=best.anchor_frame_id,
            target_window_ids=list(best.window_ids),
            anchor_type="pose_path_window",
            reason=f"pose_path_window:{best.reason}",
            protected_frame_ids=list(dict.fromkeys(protected_ids + best.window_ids[:3])),
            source_anchor_frame_id=source_anchor_id,
            loop_closure_detected=True,
            turn_result="ReturnToKnownView",
            policy_mode=scheduling_policy.policy_mode,
            allow_memory_query=scheduling_policy.allow_memory_query,
            allow_hard_retrieve=scheduling_policy.allow_hard_retrieve,
            allow_soft_reuse=scheduling_policy.allow_soft_reuse,
            query_result="matched_window",
            candidate_count=len(candidates),
            best_candidate_frame_id=best.anchor_frame_id,
            best_candidate_score=float(best.score),
            best_candidate_pose_distance=float(best.pose_distance),
            best_candidate_yaw_delta=float(best.yaw_delta),
            best_candidate_frame_gap=int(best.frame_gap),
            rank_reason=best.reason,
        )

    def _loop_closure_selection(
        self,
        view_state: ViewState,
        pose_state: PoseState,
        source_anchor: KeyPoseAnchor | None,
        key_pose_memory: KeyPoseMemory,
        turn_state: TurnState,
        scheduling_policy: SchedulingPolicy,
    ) -> MemorySelection | None:
        if turn_state.state not in {"TurnInProgress", "PostTurnStabilization", "StableNewView"}:
            return None

        candidates: list[KeyPoseAnchor] = []
        if source_anchor is not None:
            candidates.append(source_anchor)
        nearest_id, nearest_anchor, nearest_distance = key_pose_memory.nearest(pose_state)
        if nearest_anchor is not None and all(anchor.frame_id != nearest_anchor.frame_id for anchor in candidates):
            candidates.append(nearest_anchor)

        best_anchor = None
        best_distance = float("inf")
        best_yaw = float("inf")
        for anchor in candidates:
            distance = PoseMemory.distance(pose_state, anchor.pose)
            yaw_delta = PoseMemory.yaw_delta(pose_state, anchor.pose)
            if distance <= self.stability.known_view_distance_threshold and yaw_delta <= self.stability.known_view_yaw_threshold:
                score = distance + yaw_delta
                if score < best_distance + best_yaw:
                    best_anchor = anchor
                    best_distance = distance
                    best_yaw = yaw_delta

        if best_anchor is None:
            return None

        mode = "hard_retrieve_anchor" if view_state.state == "Revisit" or turn_state.state != "TurnInProgress" else "soft_reuse_anchor"
        return MemorySelection(
            mode=mode,
            target_frame_id=best_anchor.frame_id,
            anchor_type=best_anchor.anchor_type,
            reason=f"loop_closure:{turn_state.state}:{best_distance:.4f}:{best_yaw:.4f}",
            protected_frame_ids=[int(best_anchor.frame_id)],
            loop_closure_detected=True,
            turn_result="ReturnToKnownView",
            policy_mode=scheduling_policy.policy_mode,
            allow_memory_query=scheduling_policy.allow_memory_query,
            allow_hard_retrieve=scheduling_policy.allow_hard_retrieve,
            allow_soft_reuse=scheduling_policy.allow_soft_reuse,
            query_result="matched_anchor",
            candidate_count=1,
            best_candidate_frame_id=best_anchor.frame_id,
            best_candidate_pose_distance=float(best_distance),
            best_candidate_yaw_delta=float(best_yaw),
            rank_reason="anchor_fallback",
        )


class RevisitGate:
    def __init__(self, stability: StrategyStabilityConfig):
        self.stability = stability

    def evaluate(
        self,
        view_state: ViewState,
        trajectory_state: TrajectoryState,
        pose_state: PoseState | None,
        action_mode: ActionModeState | None = None,
    ) -> RevisitGateResult:
        if pose_state is None:
            return RevisitGateResult(reason="pose_missing")
        action_mode = action_mode or ActionModeState()
        if action_mode.mode == "Locomotion Only":
            return RevisitGateResult(reason="locomotion_only_orb_dominant")
        if view_state.state == "View Transition":
            return RevisitGateResult(protect_current=True, reason="view_transition")
        if action_mode.mode == "View Rotation Only":
            return RevisitGateResult(protect_current=True, reason="view_rotation_only")
        if trajectory_state.should_progress and action_mode.mode == "Viewpoint Locomotion":
            return RevisitGateResult(force_progress=True, reason=f"hybrid_{trajectory_state.direction}_progress")
        if trajectory_state.should_progress and action_mode.mode not in {"Locomotion Only"}:
            return RevisitGateResult(force_progress=True, reason=f"trajectory_{trajectory_state.direction}_progress")
        movement = float(getattr(pose_state, "movement_magnitude", 0.0) or 0.0)
        if view_state.state == "Revisit":
            if movement <= self.stability.forward_progress_threshold:
                return RevisitGateResult(allow_retrieve=True, reason="stable_revisit")
            return RevisitGateResult(reason=f"revisit_blocked_by_motion:{movement:.4f}")
        if view_state.state == "Known View":
            if movement <= self.stability.forward_progress_threshold:
                return RevisitGateResult(allow_soft_reuse=True, reason="stable_known_view")
            return RevisitGateResult(reason=f"known_view_blocked_by_motion:{movement:.4f}")
        if view_state.state == "Novel View":
            return RevisitGateResult(force_progress=True, reason="novel_view_progress")
        return RevisitGateResult(reason=f"view_state={view_state.state}")


class ProposalEngine:
    def __init__(self, policy: MemoryPolicy):
        self.policy = policy

    def propose(self, similarity: SimilarityResult, fusion_result: Any = None) -> MemoryProposal:
        appearance_score = self._appearance_score(similarity, fusion_result)
        state = "KEEP" if appearance_score >= self.policy.stable_score else "INSERT"
        return MemoryProposal(
            state=state,
            source="orb_appearance",
            appearance_score=appearance_score,
            confidence=float(np.clip(similarity.confidence, 0.0, 1.0)),
        )

    @staticmethod
    def _appearance_score(similarity: SimilarityResult, fusion_result: Any = None) -> float:
        if fusion_result is not None:
            evidence_scores = getattr(fusion_result, "evidence_scores", {}) or {}
            if "appearance" in evidence_scores:
                return float(np.clip(evidence_scores["appearance"], 0.0, 1.0))
        return float(np.clip(similarity.similarity, 0.0, 1.0))


class EvidenceValidator:
    def __init__(self, policy: MemoryPolicy):
        self.policy = policy

    def validate(
        self,
        proposal: MemoryProposal,
        fusion_result: Any = None,
        geometry_confidence: float = 1.0,
        intent_state: str = "Unknown",
        intent_confidence: float = 0.0,
    ) -> EvidenceValidation:
        report = fusion_result.evidence_report() if fusion_result is not None and hasattr(fusion_result, "evidence_report") else {}
        scores = getattr(fusion_result, "evidence_scores", {}) if fusion_result is not None else {}
        semantic = float(report.get("semantic_consistency", scores.get("semantic", proposal.appearance_score) or 0.0))
        geometry = float(report.get("geometry_consistency", scores.get("geometry", 1.0) or 1.0))
        explanation = self._normalize_intent_explanation(str(report.get("intent_explanation", self._intent_explanation(intent_state))))
        world_change = float(report.get("world_change_probability", 1.0 - proposal.appearance_score))

        validation = EvidenceValidation(
            semantic_consistency=float(np.clip(semantic, 0.0, 1.0)),
            geometry_consistency=float(np.clip(geometry, 0.0, 1.0)),
            intent_explanation=explanation,
            world_change_probability=float(np.clip(world_change, 0.0, 1.0)),
        )
        validation.semantic = self._validate_consistency(validation.semantic_consistency)
        validation.geometry = self._validate_geometry(validation.geometry_consistency, geometry_confidence)
        validation.intent = self._validate_intent(proposal, explanation, intent_confidence, validation.world_change_probability)
        validation.reasons = self._reasons(proposal, validation, geometry_confidence, intent_state, intent_confidence)
        return validation

    def _validate_consistency(self, value: float) -> str:
        if value >= self.policy.stable_score:
            return "support"
        if value < self.policy.refresh_score:
            return "reject"
        return "neutral"

    def _validate_geometry(self, value: float, confidence: float) -> str:
        if confidence < self.policy.low_geometry_confidence:
            return "reject"
        return self._validate_consistency(value)

    @staticmethod
    def _validate_intent(
        proposal: MemoryProposal,
        explanation: str,
        intent_confidence: float,
        world_change_probability: float,
    ) -> str:
        if explanation == "viewpoint_change" and intent_confidence >= 0.30:
            return "support" if proposal.state == "KEEP" else "reject"
        if explanation == "viewpoint_locomotion" and intent_confidence >= 0.30:
            return "neutral"
        if explanation in {"locomotion", "vertical_motion"} and intent_confidence >= 0.50:
            if proposal.state == "KEEP":
                return "support"
            return "neutral"
        return "neutral"

    @staticmethod
    def _normalize_intent_explanation(explanation: str) -> str:
        if explanation == "camera_motion":
            return "viewpoint_change"
        return explanation

    @staticmethod
    def _intent_explanation(intent_state: str) -> str:
        if intent_state in {"Idle", ""}:
            return "idle"
        if intent_state in {"Turn Left", "Turn Right"}:
            return "viewpoint_change"
        if intent_state == "Viewpoint Locomotion":
            return "viewpoint_locomotion"
        if intent_state in {"Forward", "Backward", "Left", "Right", "Walk", "Run"}:
            return "locomotion"
        if intent_state == "Jump":
            return "vertical_motion"
        return "unknown"

    @staticmethod
    def _reasons(
        proposal: MemoryProposal,
        validation: EvidenceValidation,
        geometry_confidence: float,
        intent_state: str,
        intent_confidence: float,
    ) -> List[str]:
        return [
            f"proposal={proposal.state}",
            f"appearance={proposal.appearance_score:.4f}",
            f"semantic={validation.semantic}:{validation.semantic_consistency:.4f}",
            f"geometry={validation.geometry}:{validation.geometry_consistency:.4f}",
            f"geometry_confidence={geometry_confidence:.4f}",
            f"intent={validation.intent}:{intent_state}:{intent_confidence:.4f}:{validation.intent_explanation}",
            f"world_change={validation.world_change_probability:.4f}",
        ]


class PoseValidator:
    def __init__(self, stability: StrategyStabilityConfig):
        self.stability = stability

    def validate(
        self,
        pose_state: PoseState | None,
        pose_memory: PoseMemory,
        reference_frame_id: int,
        middle_frame_id: int,
        view_state: ViewState | None = None,
    ) -> PoseValidation:
        if pose_state is None:
            return PoseValidation(event="unknown", validation="neutral", reasons=["pose=missing"])

        if view_state is not None:
            if view_state.state == "View Transition":
                return PoseValidation(
                    event="view_transition",
                    validation="reject_hard_update",
                    pose_distance=view_state.pose_distance,
                    yaw_delta=view_state.yaw_delta,
                    nearest_frame_id=view_state.nearest_frame_id,
                    reasons=view_state.reasons,
                )
            if view_state.state == "Novel View":
                return PoseValidation(
                    event="novel_view",
                    validation="support_insert",
                    pose_distance=view_state.pose_distance,
                    yaw_delta=view_state.yaw_delta,
                    nearest_frame_id=view_state.nearest_frame_id,
                    reasons=view_state.reasons,
                )
            if view_state.state == "Revisit":
                return PoseValidation(
                    event="revisit",
                    validation="support_keep",
                    pose_distance=view_state.pose_distance,
                    yaw_delta=view_state.yaw_delta,
                    nearest_frame_id=view_state.nearest_frame_id,
                    reasons=view_state.reasons,
                )

        reference_pose = pose_memory.get(reference_frame_id)
        middle_pose = pose_memory.get(middle_frame_id)
        nearest_id, nearest_pose, nearest_distance = pose_memory.nearest(pose_state)
        yaw_delta = PoseMemory.yaw_delta(pose_state, nearest_pose)
        delta_position = float(np.sqrt(pose_state.delta_x * pose_state.delta_x + pose_state.delta_z * pose_state.delta_z))
        delta_yaw = abs(float(pose_state.delta_yaw))
        if delta_yaw >= self.stability.viewpoint_rotation_threshold and delta_position > self.stability.viewpoint_translation_threshold:
            return PoseValidation(
                event="viewpoint_locomotion",
                validation="support_insert",
                pose_distance=nearest_distance,
                yaw_delta=yaw_delta,
                nearest_frame_id=nearest_id,
                reasons=[f"delta_yaw={delta_yaw:.4f}", f"delta_position={delta_position:.4f}"],
            )

        if delta_yaw >= self.stability.viewpoint_rotation_threshold and delta_position <= self.stability.viewpoint_translation_threshold:
            return PoseValidation(
                event="viewpoint_change",
                validation="reject_hard_update",
                pose_distance=nearest_distance,
                yaw_delta=yaw_delta,
                nearest_frame_id=nearest_id,
                reasons=[f"delta_yaw={delta_yaw:.4f}", f"delta_position={delta_position:.4f}"],
            )
        if pose_state.delta_z >= self.stability.forward_progress_threshold and delta_yaw <= self.stability.yaw_stable_threshold:
            return PoseValidation(
                event="forward_progression",
                validation="support_insert",
                pose_distance=PoseMemory.distance(pose_state, reference_pose),
                yaw_delta=PoseMemory.yaw_delta(pose_state, reference_pose),
                nearest_frame_id=nearest_id,
                reasons=[f"delta_z={pose_state.delta_z:.4f}", f"delta_yaw={delta_yaw:.4f}"],
            )
        if abs(pose_state.delta_x) >= self.stability.forward_progress_threshold and delta_yaw <= self.stability.yaw_stable_threshold:
            return PoseValidation(
                event="lateral_motion",
                validation="support_insert",
                pose_distance=PoseMemory.distance(pose_state, middle_pose),
                yaw_delta=PoseMemory.yaw_delta(pose_state, middle_pose),
                nearest_frame_id=nearest_id,
                reasons=[f"delta_x={pose_state.delta_x:.4f}", f"delta_yaw={delta_yaw:.4f}"],
            )
        if nearest_id is not None and nearest_distance <= self.stability.revisit_distance_threshold and yaw_delta <= self.stability.revisit_yaw_threshold:
            return PoseValidation(
                event="pose_revisit",
                validation="support_keep",
                pose_distance=nearest_distance,
                yaw_delta=yaw_delta,
                nearest_frame_id=nearest_id,
                reasons=[f"nearest={nearest_id}", f"distance={nearest_distance:.4f}", f"yaw_delta={yaw_delta:.4f}"],
            )
        return PoseValidation(
            event="pose_continuous",
            validation="neutral",
            pose_distance=nearest_distance,
            yaw_delta=yaw_delta,
            nearest_frame_id=nearest_id,
            reasons=[f"distance={nearest_distance:.4f}", f"yaw_delta={yaw_delta:.4f}"],
        )


class ViewStateClassifier:
    def __init__(self, stability: StrategyStabilityConfig):
        self.stability = stability

    def classify(self, pose_state: PoseState | None, pose_memory: PoseMemory) -> ViewState:
        if pose_state is None:
            return ViewState(state="Unknown", confidence=0.0, reasons=["pose=missing"])

        nearest_id, nearest_pose, nearest_distance = pose_memory.nearest(pose_state)
        yaw_delta = PoseMemory.yaw_delta(pose_state, nearest_pose)
        frame_gap = 0 if nearest_id is None else max(0, int(pose_state.frame_index) - int(nearest_id))
        delta_yaw = abs(float(pose_state.delta_yaw))
        delta_position = float(np.sqrt(pose_state.delta_x * pose_state.delta_x + pose_state.delta_z * pose_state.delta_z))

        base = {
            "pose_distance": nearest_distance,
            "yaw_delta": yaw_delta,
            "nearest_frame_id": nearest_id,
            "frame_gap": frame_gap,
        }
        if delta_yaw >= self.stability.view_transition_rotation_threshold:
            return ViewState(
                state="View Transition",
                confidence=float(np.clip(delta_yaw / max(self.stability.view_transition_rotation_threshold * 2.0, 1e-6), 0.0, 1.0)),
                reasons=[f"delta_yaw={delta_yaw:.4f}", f"delta_position={delta_position:.4f}"],
                **base,
            )

        if (
            nearest_id is not None
            and frame_gap >= self.stability.revisit_min_frame_gap
            and nearest_distance <= self.stability.revisit_distance_threshold
            and yaw_delta <= self.stability.revisit_yaw_threshold
        ):
            distance_score = 1.0 - min(nearest_distance / max(self.stability.revisit_distance_threshold, 1e-6), 1.0)
            yaw_score = 1.0 - min(yaw_delta / max(self.stability.revisit_yaw_threshold, 1e-6), 1.0)
            return ViewState(
                state="Revisit",
                confidence=float(np.clip(0.5 * distance_score + 0.5 * yaw_score, 0.0, 1.0)),
                reasons=[f"nearest={nearest_id}", f"frame_gap={frame_gap}", f"distance={nearest_distance:.4f}", f"yaw_delta={yaw_delta:.4f}"],
                **base,
            )

        if (
            nearest_id is None
            or nearest_distance >= self.stability.novel_view_distance_threshold
            or yaw_delta >= self.stability.novel_view_yaw_threshold
        ):
            distance_score = 0.0 if nearest_id is None else min(nearest_distance / max(self.stability.novel_view_distance_threshold, 1e-6), 1.0)
            yaw_score = 0.0 if nearest_id is None else min(yaw_delta / max(self.stability.novel_view_yaw_threshold, 1e-6), 1.0)
            return ViewState(
                state="Novel View",
                confidence=float(np.clip(max(distance_score, yaw_score), 0.0, 1.0)),
                reasons=[f"nearest={nearest_id}", f"distance={nearest_distance:.4f}", f"yaw_delta={yaw_delta:.4f}"],
                **base,
            )

        if nearest_distance <= self.stability.known_view_distance_threshold and yaw_delta <= self.stability.known_view_yaw_threshold:
            distance_score = 1.0 - min(nearest_distance / max(self.stability.known_view_distance_threshold, 1e-6), 1.0)
            yaw_score = 1.0 - min(yaw_delta / max(self.stability.known_view_yaw_threshold, 1e-6), 1.0)
            return ViewState(
                state="Known View",
                confidence=float(np.clip(0.5 * distance_score + 0.5 * yaw_score, 0.0, 1.0)),
                reasons=[f"nearest={nearest_id}", f"distance={nearest_distance:.4f}", f"yaw_delta={yaw_delta:.4f}"],
                **base,
            )

        return ViewState(
            state="Novel View",
            confidence=0.5,
            reasons=[f"ambiguous_distance={nearest_distance:.4f}", f"ambiguous_yaw_delta={yaw_delta:.4f}"],
            **base,
        )


class PhysMemStateMachine:
    ORDER = {"KEEP": 0, "INSERT": 1, "REPLACE": 2, "EVICT": 3, "REFRESH": 1}

    def __init__(self, policy: MemoryPolicy, stability: StrategyStabilityConfig | None = None):
        self.policy = policy
        self.stability = stability or StrategyStabilityConfig()
        self.last_state = "KEEP"
        self.replace_cooldown_remaining = 0
        self.evict_cooldown_remaining = 0

    def transition(
        self,
        proposal: MemoryProposal,
        validation: EvidenceValidation,
        pose_validation: PoseValidation | None = None,
        view_state: ViewState | None = None,
        memory_selection: MemorySelection | None = None,
        trajectory_state: TrajectoryState | None = None,
        action_mode: ActionModeState | None = None,
        turn_state: TurnState | None = None,
    ) -> tuple[str, str]:
        pose_validation = pose_validation or PoseValidation()
        view_state = view_state or ViewState()
        memory_selection = memory_selection or MemorySelection()
        trajectory_state = trajectory_state or TrajectoryState()
        action_mode = action_mode or ActionModeState()
        turn_state = turn_state or TurnState()
        candidate, reason = self._candidate_transition(proposal, validation, pose_validation, view_state, memory_selection, trajectory_state, action_mode, turn_state)
        constrained, constraint_reason = self._apply_transition_constraints(candidate, proposal, validation, pose_validation, view_state, memory_selection, trajectory_state, action_mode, turn_state)
        self._update_cooldown(constrained)
        previous = self.last_state
        self.last_state = constrained
        if constraint_reason:
            return constrained, f"{reason}|{constraint_reason}|mode={action_mode.mode}|turn={turn_state.state}|view={view_state.state}|selection={memory_selection.mode}|trajectory={trajectory_state.motion_state}:{trajectory_state.direction}|prev={previous}"
        return constrained, f"{reason}|mode={action_mode.mode}|turn={turn_state.state}|view={view_state.state}|selection={memory_selection.mode}|trajectory={trajectory_state.motion_state}:{trajectory_state.direction}|prev={previous}"

    def _candidate_transition(
        self,
        proposal: MemoryProposal,
        validation: EvidenceValidation,
        pose_validation: PoseValidation,
        view_state: ViewState,
        memory_selection: MemorySelection,
        trajectory_state: TrajectoryState,
        action_mode: ActionModeState,
        turn_state: TurnState,
    ) -> tuple[str, str]:
        if memory_selection.loop_closure_detected:
            return "KEEP", f"selection_loop_closure_{memory_selection.mode}"
        if memory_selection.mode in {"hard_retrieve_anchor", "soft_reuse_anchor", "retrieve_window", "soft_reuse_window"}:
            return "KEEP", f"selection_{memory_selection.mode}"
        if action_mode.mode == "Locomotion Only":
            return proposal.state, f"action_mode_locomotion_orb_{proposal.state.lower()}"
        if turn_state.state == "TurnStart":
            return "KEEP", "action_mode_view_rotation_protect"
        if turn_state.state == "TurnInProgress":
            return "INSERT", "turn_in_progress_soft_insert"
        if turn_state.state == "PostTurnStabilization":
            return "INSERT", "post_turn_stabilize_view"
        if turn_state.state == "StableNewView" and action_mode.mode == "Idle":
            return proposal.state, f"stable_new_view_orb_{proposal.state.lower()}"
        if action_mode.mode == "Viewpoint Locomotion" and trajectory_state.should_progress and proposal.state == "INSERT":
            return "INSERT", f"action_mode_hybrid_{trajectory_state.direction}_insert"
        if trajectory_state.should_progress and view_state.state != "View Transition" and action_mode.mode != "Locomotion Only":
            return "INSERT", f"trajectory_{trajectory_state.direction}_progress"

        if memory_selection.mode in {"progress_anchor", "hybrid_locomotion"}:
            return "INSERT", f"selection_{memory_selection.mode}"
        if memory_selection.mode == "protect_current":
            return "KEEP", "selection_protect_current"
        if memory_selection.mode == "create_anchor" and proposal.state == "INSERT":
            return "INSERT", "selection_create_anchor"

        if view_state.state == "View Transition":
            return "KEEP", "view_transition_protect_memory"
        if view_state.state == "Revisit":
            return "KEEP", "view_revisit_retrieve_memory"
        if view_state.state == "Novel View" and proposal.state == "INSERT":
            return "INSERT", "view_novel_insert_anchor"

        if validation.geometry == "reject":
            return "REFRESH", "proposal_validation_geometry_reject"

        if pose_validation.validation == "reject_hard_update" and proposal.state == "INSERT":
            return "KEEP", f"pose_{pose_validation.event}_protect_memory"

        if proposal.state == "KEEP":
            if validation.intent == "reject" and validation.world_change_probability >= self.stability.replace_enter_probability:
                return "REPLACE", "proposal_keep_rejected_by_world_change"
            if validation.reject_count >= 2:
                return "INSERT", "proposal_keep_soft_reject"
            return "KEEP", "proposal_keep_validated"

        if validation.intent == "reject" and validation.intent_explanation == "viewpoint_change":
            return "KEEP", "proposal_insert_explained_by_camera_motion"
        if pose_validation.validation == "support_insert":
            return "INSERT", f"pose_{pose_validation.event}_preserve_progression"
        if pose_validation.validation == "support_keep":
            return "KEEP", f"pose_{pose_validation.event}_preserve_memory"
        if validation.semantic == "support" and validation.geometry == "support":
            return "INSERT", "proposal_insert_validated"
        physical_support_count = sum(1 for item in (validation.semantic, validation.geometry) if item == "support")
        if validation.world_change_probability >= self.stability.evict_enter_probability and physical_support_count >= 2:
            return "EVICT", "proposal_insert_hard_world_change"
        if validation.world_change_probability >= self.stability.replace_enter_probability:
            return "REPLACE", "proposal_insert_world_change"
        return "INSERT", "proposal_insert_conservative"

    def _apply_transition_constraints(
        self,
        candidate: str,
        proposal: MemoryProposal,
        validation: EvidenceValidation,
        pose_validation: PoseValidation,
        view_state: ViewState,
        memory_selection: MemorySelection,
        trajectory_state: TrajectoryState,
        action_mode: ActionModeState,
        turn_state: TurnState,
    ) -> tuple[str, str]:
        if turn_state.state in {"TurnInProgress", "PostTurnStabilization"} and candidate in {"REPLACE", "EVICT"}:
            return "INSERT", f"turn_{turn_state.state.lower()}_blocks_hard_update"
        if action_mode.mode == "Locomotion Only":
            if candidate in {"REPLACE", "EVICT", "REFRESH"}:
                return proposal.state, f"locomotion_orb_blocks_{candidate.lower()}"
            return candidate, ""
        if trajectory_state.should_progress and candidate in {"REPLACE", "EVICT", "KEEP"} and view_state.state != "View Transition":
            return "INSERT", f"trajectory_{trajectory_state.direction}_blocks_pullback"

        if memory_selection.mode in {"progress_anchor", "hybrid_locomotion"} and candidate in {"KEEP", "REPLACE", "EVICT"}:
            return "INSERT", "selection_progress_anchor_blocks_pullback"
        if memory_selection.mode in {"hard_retrieve_anchor", "soft_reuse_anchor", "retrieve_window", "soft_reuse_window", "protect_current"} and candidate in {"REPLACE", "EVICT"}:
            return "KEEP", f"selection_{memory_selection.mode}_blocks_hard_update"
        if memory_selection.mode == "create_anchor" and candidate in {"REPLACE", "EVICT"}:
            return "INSERT", "selection_create_anchor_prefers_insert"

        if view_state.state in {"View Transition", "Revisit"} and candidate in {"REPLACE", "EVICT"}:
            return "KEEP", f"view_{view_state.state.lower().replace(' ', '_')}_blocks_hard_update"
        if view_state.state == "Novel View" and candidate in {"REPLACE", "EVICT"}:
            return "INSERT", "view_novel_prefers_insert"

        if pose_validation.validation == "reject_hard_update" and candidate in {"REPLACE", "EVICT"}:
            return "KEEP", f"pose_{pose_validation.event}_blocks_hard_update"
        if pose_validation.validation == "support_insert" and candidate == "EVICT":
            return "INSERT", f"pose_{pose_validation.event}_blocks_evict"
        if pose_validation.validation == "support_keep" and candidate in {"REPLACE", "EVICT"}:
            return "KEEP", f"pose_{pose_validation.event}_blocks_hard_update"

        if candidate == "REPLACE" and self.replace_cooldown_remaining > 0:
            return "INSERT", f"replace_cooldown={self.replace_cooldown_remaining}"
        if candidate == "EVICT" and self.evict_cooldown_remaining > 0:
            return "REPLACE", f"evict_cooldown={self.evict_cooldown_remaining}"

        if candidate == "EVICT" and self.last_state == "KEEP" and not self.stability.allow_direct_keep_to_evict:
            return "REPLACE", "blocked_direct_keep_to_evict"

        if candidate == "REPLACE" and self.last_state == "REPLACE":
            if validation.world_change_probability < self.stability.replace_exit_probability:
                return "INSERT", f"replace_hysteresis_exit={self.stability.replace_exit_probability:.2f}"

        if candidate == "EVICT" and self.last_state == "EVICT":
            if validation.world_change_probability < self.stability.evict_exit_probability:
                return "REPLACE", f"evict_hysteresis_exit={self.stability.evict_exit_probability:.2f}"

        if self.ORDER.get(candidate, 0) > self.ORDER.get(self.last_state, 0) + 1:
            if candidate == "EVICT":
                return "REPLACE", "stepwise_transition_to_evict"
            return "INSERT", "stepwise_transition"

        if candidate == "INSERT" and validation.world_change_probability < self.stability.insert_enter_probability:
            if trajectory_state.should_progress or memory_selection.mode == "progress_anchor":
                return "INSERT", "trajectory_progress_bypasses_insert_hysteresis"
            if proposal.state == "KEEP":
                return "KEEP", f"insert_hysteresis_enter={self.stability.insert_enter_probability:.2f}"

        return candidate, ""

    def _update_cooldown(self, state: str):
        self.replace_cooldown_remaining = max(0, self.replace_cooldown_remaining - 1)
        self.evict_cooldown_remaining = max(0, self.evict_cooldown_remaining - 1)
        if state == "REPLACE":
            self.replace_cooldown_remaining = self.stability.replace_cooldown
        if state == "EVICT":
            self.evict_cooldown_remaining = self.stability.evict_cooldown


class PhysMemScheduler(MemoryScheduler):
    def __init__(
        self,
        sim_threshold: float,
        policy: MemoryPolicy | None = None,
        stability: StrategyStabilityConfig | None = None,
        use_pose_memory: bool = True,
        use_pose_path_memory: bool = True,
        ranking_config: HierarchicalRankingConfig | None = None,
    ):
        super().__init__(sim_threshold=sim_threshold)
        self.policy = policy or MemoryPolicy(stable_score=sim_threshold)
        self.stability = stability or StrategyStabilityConfig()
        self.use_pose_memory = bool(use_pose_memory)
        self.use_pose_path_memory = bool(use_pose_path_memory)
        self.ranking_config = ranking_config or HierarchicalRankingConfig()
        self.initial_window_ids: list[int] | None = None
        self.proposal_engine = ProposalEngine(self.policy)
        self.validator = EvidenceValidator(self.policy)
        self.pose_memory = PoseMemory()
        self.key_pose_memory = KeyPoseMemory(stability=self.stability)
        self.pose_path_memory = PosePathMemory()
        self.policy_selector = MemoryPolicySelector()
        self.historical_ranker = HistoricalWindowRanker(min_score=self.ranking_config.min_score)
        self.pose_validator = PoseValidator(self.stability)
        self.view_state_classifier = ViewStateClassifier(self.stability)
        self.memory_selector = PoseAwareMemorySelector(self.stability)
        self.revisit_gate = RevisitGate(self.stability)
        self.trajectory_tracker = TrajectoryTracker(self.stability)
        self.action_mode_classifier = ActionModeClassifier()
        self.turn_state_machine = TurnStateMachine(self.stability)
        self.state_machine = PhysMemStateMachine(self.policy, self.stability)
        self.last_view_state = ViewState(state="Not Evaluated", confidence=0.0)
        self.last_key_anchor: KeyPoseAnchor | None = None
        self.last_memory_selection = MemorySelection()
        self.last_trajectory_state = TrajectoryState()
        self.last_revisit_gate = RevisitGateResult(reason="not_evaluated")
        self.last_action_mode = ActionModeState(mode="Not Evaluated")
        self.last_turn_state = TurnState(state="Not Evaluated")
        self.active_source_anchor: KeyPoseAnchor | None = None
        self.last_scheduling_policy = SchedulingPolicy(policy_mode="Not Evaluated")

    def schedule(
        self,
        memory_buffer: MemoryBuffer,
        similarity: SimilarityResult,
        fusion_result: Any = None,
        unified_memory_score: float | None = None,
        geometry_confidence: float = 1.0,
        intent_state: str = "Unknown",
        intent_confidence: float = 0.0,
        pose_state: PoseState | None = None,
    ) -> MemoryDecision:
        if self.initial_window_ids is None:
            self.initial_window_ids = list(memory_buffer.snapshot())
        proposal = self.proposal_engine.propose(similarity, fusion_result)
        validation = self.validator.validate(
            proposal,
            fusion_result=fusion_result,
            geometry_confidence=geometry_confidence,
            intent_state=intent_state,
            intent_confidence=intent_confidence,
        )
        if self.use_pose_memory:
            view_state = self.view_state_classifier.classify(pose_state, self.pose_memory)
            trajectory_state = self.trajectory_tracker.update(pose_state)
            action_mode = self.action_mode_classifier.classify(pose_state, intent_state)
            turn_state = self.turn_state_machine.update(action_mode, pose_state)
            revisit_gate = self.revisit_gate.evaluate(view_state, trajectory_state, pose_state, action_mode)
            scheduling_policy = self.policy_selector.select(action_mode, turn_state, view_state, trajectory_state, revisit_gate)
            self._update_source_anchor(memory_buffer, pose_state, view_state, action_mode, turn_state)
            memory_selection = self.memory_selector.select(
                view_state,
                self.key_pose_memory,
                pose_state,
                revisit_gate,
                action_mode,
                turn_state,
                source_anchor=self.active_source_anchor,
                fallback_source_frame_id=memory_buffer.reference_frame_id,
                scheduling_policy=scheduling_policy,
                pose_path_memory=self.pose_path_memory,
                historical_ranker=self.historical_ranker,
                current_frame_id=memory_buffer.current_frame_id,
                use_pose_path_memory=self.use_pose_path_memory,
                ranking_config=self.ranking_config,
                initial_window_ids=self.initial_window_ids,
            )
            pose_validation = self.pose_validator.validate(
                pose_state,
                self.pose_memory,
                reference_frame_id=memory_buffer.reference_frame_id,
                middle_frame_id=memory_buffer.middle_frame_id,
                view_state=view_state,
            )
        else:
            view_state = ViewState(state="Pose Disabled", confidence=0.0, reasons=["use_pose_memory=false"])
            trajectory_state = TrajectoryState(motion_state="Pose Disabled", reason="use_pose_memory=false")
            revisit_gate = RevisitGateResult(reason="use_pose_memory=false")
            action_mode = ActionModeState(mode="Pose Disabled", reason="use_pose_memory=false")
            turn_state = TurnState(state="Pose Disabled", reason="use_pose_memory=false")
            memory_selection = MemorySelection(mode="pose_disabled", reason="use_pose_memory=false")
            pose_validation = PoseValidation(event="pose_disabled", validation="neutral", reasons=["use_pose_memory=false"])
            scheduling_policy = SchedulingPolicy(policy_mode="PoseDisabledPolicy", reason="use_pose_memory=false")
        self.last_view_state = view_state
        self.last_memory_selection = memory_selection
        self.last_trajectory_state = trajectory_state
        self.last_revisit_gate = revisit_gate
        self.last_action_mode = action_mode
        self.last_turn_state = turn_state
        self.last_scheduling_policy = scheduling_policy
        memory_state, transition = self.state_machine.transition(
            proposal,
            validation,
            pose_validation,
            view_state,
            memory_selection,
            trajectory_state,
            action_mode,
            turn_state,
        )
        self.last_key_anchor = None
        if self.use_pose_memory:
            self.pose_memory.insert(memory_buffer.current_frame_id, pose_state)
            self.last_key_anchor = self.key_pose_memory.maybe_insert(
                memory_buffer.current_frame_id,
                pose_state,
                view_state,
                pose_validation,
                memory_state,
                revisit_gate,
                action_mode,
                turn_state,
            )
            self.pose_path_memory.insert(
                memory_buffer.current_frame_id,
                pose_state,
                memory_buffer.snapshot(),
                view_state,
                action_mode,
                turn_state,
                memory_state,
                memory_selection,
                stability_score=float(proposal.appearance_score),
            )
        keep_ids, delete_range, refresh_ids, insert_count, kv_policy, evict_middle = self._strategy_plan(memory_buffer, memory_state, memory_selection)
        score = float(
            unified_memory_score
            if unified_memory_score is not None
            else getattr(fusion_result, "unified_memory_score", proposal.appearance_score)
        )
        decision_name = f"physmem_{memory_state.lower()}"
        return MemoryDecision(
            evict_middle=evict_middle,
            delete_range=delete_range,
            decision=decision_name,
            similarity=float(similarity.similarity),
            confidence=float(similarity.confidence),
            matching_points=int(similarity.matching_points),
            memory_state=memory_state,
            policy=self.policy.name,
            transition=transition,
            unified_memory_score=score,
            geometry_confidence=float(geometry_confidence),
            intent_state=intent_state,
            keep_ids=keep_ids,
            delete_ids=delete_range,
            refresh_ids=refresh_ids,
            insert_count=insert_count,
            kv_policy=kv_policy,
        )

    def _update_source_anchor(
        self,
        memory_buffer: MemoryBuffer,
        pose_state: PoseState | None,
        view_state: ViewState,
        action_mode: ActionModeState,
        turn_state: TurnState,
    ) -> None:
        if pose_state is None:
            return
        if turn_state.state == "TurnStart":
            anchor_frame_id = int(memory_buffer.reference_frame_id)
            anchor_pose = self.pose_memory.get(anchor_frame_id) or pose_state
            self.active_source_anchor = KeyPoseAnchor(
                frame_id=anchor_frame_id,
                pose=anchor_pose,
                anchor_type="source_view_anchor",
                view_state=view_state.state,
                confidence=float(view_state.confidence),
                reason=f"turn_start:{action_mode.mode}",
            )
            self.key_pose_memory.anchors[anchor_frame_id] = self.active_source_anchor
            if anchor_frame_id not in self.key_pose_memory.order:
                self.key_pose_memory.order.append(anchor_frame_id)
            return
        if turn_state.state in {"Idle", "StableNewView"} and action_mode.mode in {"Idle", "Locomotion Only"}:
            self.active_source_anchor = None

    def _strategy_plan(
        self,
        memory_buffer: MemoryBuffer,
        memory_state: str,
        memory_selection: MemorySelection | None = None,
    ) -> tuple[list[int], list[int], list[int], int, str, int]:
        ids = memory_buffer.snapshot()
        if memory_state == "KEEP":
            if memory_selection is not None and memory_selection.mode in {"retrieve_window", "soft_reuse_window"} and memory_selection.target_window_ids:
                keep_ids = self._compose_retrieved_window(ids, memory_selection.target_window_ids, memory_selection.protected_frame_ids)
                delete_ids = [frame_id for frame_id in ids if frame_id not in set(keep_ids)]
                kv_policy = "retrieve_pose_window" if memory_selection.mode == "retrieve_window" else "soft_reuse_pose_window"
                return keep_ids, delete_ids, [], max(0, len(ids) - len(keep_ids)), kv_policy, 0
            delete_ids = ids[3:6]
            keep_ids = ids[:3] + ids[6:]
            if memory_selection is not None and memory_selection.target_frame_id is not None:
                target_id = int(memory_selection.target_frame_id)
                if target_id in ids:
                    keep_ids = self._protect_target(ids, keep_ids, delete_ids, target_id)
                    delete_ids = [frame_id for frame_id in ids if frame_id not in set(keep_ids)]
                elif memory_selection.mode == "hard_retrieve_anchor":
                    keep_ids = [target_id] + keep_ids
            if memory_selection is not None:
                keep_ids = self._protect_frames(ids, keep_ids, memory_selection.protected_frame_ids)
                delete_ids = [frame_id for frame_id in ids if frame_id not in set(keep_ids)]
            return keep_ids, delete_ids, [], 3, "preserve_anchor", 1
        if memory_state == "REFRESH":
            delete_ids = [ids[2], ids[5]]
            keep_ids = [frame_id for frame_id in ids if frame_id not in set(delete_ids)]
            return keep_ids, delete_ids, delete_ids, 2, "refresh_uncertain", 1
        if memory_state == "INSERT":
            if memory_selection is not None and memory_selection.mode == "turn_transition":
                delete_ids, keep_ids = self._delete_oldest_unprotected(ids, memory_selection.protected_frame_ids, 2)
                return keep_ids, delete_ids, [], 2, "append_transition", 0
            if memory_selection is not None and memory_selection.mode == "stabilize_view":
                delete_ids, keep_ids = self._delete_oldest_unprotected(ids, memory_selection.protected_frame_ids, 3)
                return keep_ids, delete_ids, [], 3, "stabilize_view_anchor", 0
            if memory_selection is not None and memory_selection.mode in {"orb_locomotion", "progress_anchor", "hybrid_locomotion"}:
                delete_ids, keep_ids = self._delete_oldest_unprotected(ids, memory_selection.protected_frame_ids, 3)
                return keep_ids, delete_ids, [], 3, "append_locomotion", 0
            delete_ids = [ids[1], ids[3]]
            keep_ids = [frame_id for frame_id in ids if frame_id not in set(delete_ids)]
            return keep_ids, delete_ids, [], 2, "append_compact", 0
        if memory_state == "EVICT":
            keep_ids = ids[-6:]
            delete_ids = ids[:-6]
            return keep_ids, delete_ids, [], 3, "hard_evict", 0
        delete_ids = ids[:3]
        keep_ids = ids[3:]
        return keep_ids, delete_ids, [], 3, "replace_stale", 0

    @staticmethod
    def _protect_target(ids: list[int], keep_ids: list[int], delete_ids: list[int], target_frame_id: int) -> list[int]:
        if target_frame_id in keep_ids:
            return keep_ids
        protected = list(keep_ids)
        if protected:
            protected.pop(len(protected) // 2)
        protected.append(target_frame_id)
        order = {frame_id: index for index, frame_id in enumerate(ids)}
        return sorted(set(protected), key=lambda frame_id: order.get(frame_id, len(order)))

    @staticmethod
    def _protect_frames(ids: list[int], keep_ids: list[int], protected_frame_ids: list[int]) -> list[int]:
        protected_ids = [int(frame_id) for frame_id in protected_frame_ids if frame_id is not None]
        if not protected_ids:
            return keep_ids
        order = {frame_id: index for index, frame_id in enumerate(ids)}
        merged = list(dict.fromkeys(list(keep_ids) + protected_ids))
        in_window = [frame_id for frame_id in merged if frame_id in order]
        external = [frame_id for frame_id in merged if frame_id not in order]
        return external + sorted(in_window, key=lambda frame_id: order.get(frame_id, len(order)))

    @staticmethod
    def _delete_oldest_unprotected(ids: list[int], protected_frame_ids: list[int], count: int) -> tuple[list[int], list[int]]:
        protected = {int(frame_id) for frame_id in protected_frame_ids if frame_id is not None}
        delete_ids: list[int] = []
        for frame_id in ids:
            if frame_id in protected:
                continue
            delete_ids.append(frame_id)
            if len(delete_ids) >= count:
                break
        if len(delete_ids) < count:
            for frame_id in ids:
                if frame_id not in delete_ids:
                    delete_ids.append(frame_id)
                    if len(delete_ids) >= count:
                        break
        delete_set = set(delete_ids)
        keep_ids = [frame_id for frame_id in ids if frame_id not in delete_set]
        return delete_ids, keep_ids

    @staticmethod
    def _compose_retrieved_window(current_ids: list[int], target_window_ids: list[int], protected_frame_ids: list[int]) -> list[int]:
        current_recent = current_ids[-3:]
        target_core = list(dict.fromkeys(list(protected_frame_ids) + list(target_window_ids)))
        kept: list[int] = []
        for frame_id in target_core + current_recent:
            if frame_id not in kept:
                kept.append(frame_id)
        if len(kept) >= len(current_ids):
            return kept[:len(current_ids)]
        return kept
