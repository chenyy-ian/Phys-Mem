from dataclasses import dataclass
from typing import Dict, List

import torch

from .stableworld_action import ActionIntentEngine, ActionState, PoseState
from .stableworld_fusion import EvidenceValue, FusionEngine, FusionResult
from .stableworld_similarity import SimilarityResult, build_similarity_estimator


@dataclass
class EvidenceBundle:
    primary_similarity: SimilarityResult
    evidences: Dict[str, EvidenceValue]
    similarity_results: Dict[str, SimilarityResult]
    action_state: ActionState
    pose_state: PoseState | None
    fusion_result: FusionResult


class EvidenceCollector:
    def __init__(
        self,
        primary_estimator_name: str = "orb",
        evidence_mode: str = "single",
        lightglue_spatial_alpha: float = 0.0,
        depth_metric: str = "l1",
        depth_model: str = "vits",
        depth_checkpoint: str | None = None,
        depth_cache_size: int = 256,
        fusion_engine: FusionEngine | None = None,
        action_engine: ActionIntentEngine | None = None,
    ):
        self.primary_estimator_name = (primary_estimator_name or "orb").lower()
        self.evidence_mode = (evidence_mode or "single").lower()
        self.lightglue_spatial_alpha = lightglue_spatial_alpha
        self.depth_metric = depth_metric
        self.depth_model = depth_model
        self.depth_checkpoint = depth_checkpoint
        self.depth_cache_size = depth_cache_size
        self.fusion_engine = fusion_engine or FusionEngine()
        self.action_engine = action_engine or ActionIntentEngine()
        self._estimators = {}

    def _estimator(self, name: str):
        if name not in self._estimators:
            self._estimators[name] = build_similarity_estimator(
                name,
                lightglue_spatial_alpha=self.lightglue_spatial_alpha,
                depth_metric=self.depth_metric,
                depth_model=self.depth_model,
                depth_checkpoint=self.depth_checkpoint,
                depth_cache_size=self.depth_cache_size,
            )
        return self._estimators[name]

    def _active_estimators(self) -> List[str]:
        if self.evidence_mode in {"multi", "fusion", "physmem"}:
            return ["orb", "lightglue", "depth"]
        return [self.primary_estimator_name]

    @staticmethod
    def _action_explanation(action_state: ActionState) -> str:
        if getattr(action_state, "action_explanation", None):
            return action_state.action_explanation
        if action_state.intent_state in {"Turn Left", "Turn Right"} or action_state.rotation_speed > 0:
            return "viewpoint_change"
        if action_state.intent_state in {"Forward", "Backward", "Left", "Right", "Walk", "Run"}:
            return "locomotion"
        if action_state.intent_state == "Jump":
            return "vertical_motion"
        if action_state.intent_state == "Idle":
            return "idle"
        return "unknown"

    @staticmethod
    def _result_to_evidence(name: str, result: SimilarityResult) -> EvidenceValue:
        if name == "orb":
            return EvidenceValue(
                name="appearance",
                score=float(result.similarity),
                confidence=float(result.confidence),
                available=True,
                metadata={"estimator": name, "matching_points": int(result.matching_points)},
            )
        if name == "lightglue":
            semantic_similarity = float(result.debug.get("semantic_similarity", result.similarity))
            return EvidenceValue(
                name="semantic",
                score=semantic_similarity,
                confidence=float(result.confidence),
                available=True,
                metadata={"estimator": name, "matching_points": int(result.matching_points)},
            )
        if name == "depth":
            geometry_similarity = float(result.debug.get("geometry_similarity", result.similarity))
            return EvidenceValue(
                name="geometry",
                score=geometry_similarity,
                confidence=float(result.confidence),
                available=True,
                metadata={"estimator": name, "matching_points": int(result.matching_points)},
            )
        return EvidenceValue(name=name, score=float(result.similarity), confidence=float(result.confidence), available=True)

    def collect(
        self,
        reference_frame: torch.Tensor,
        middle_frame: torch.Tensor,
        conditional_dict: dict,
        frame_index: int,
        current_start_frame: int,
        num_frame_per_block: int,
        mode: str,
        return_debug: bool = False,
    ) -> EvidenceBundle:
        action_state = self.action_engine.record(
            frame_index=frame_index,
            conditional_dict=conditional_dict,
            current_start_frame=current_start_frame,
            num_frame_per_block=num_frame_per_block,
            mode=mode,
        )

        similarity_results: Dict[str, SimilarityResult] = {}
        evidences: Dict[str, EvidenceValue] = {}
        for estimator_name in self._active_estimators():
            result = self._estimator(estimator_name).compute_similarity(
                reference_frame,
                middle_frame,
                return_debug=return_debug,
            )
            similarity_results[estimator_name] = result
            evidence = self._result_to_evidence(estimator_name, result)
            evidences[evidence.name] = evidence

        primary_result = similarity_results.get(self.primary_estimator_name)
        if primary_result is None:
            primary_result = next(iter(similarity_results.values()))

        intent_score = FusionEngine.intent_to_memory_score(action_state.intent_state, action_state.intent_confidence)
        action_explanation = self._action_explanation(action_state)
        pose_state = action_state.pose_state
        pose_dict = pose_state.as_dict() if pose_state is not None else {}
        evidences["intent"] = EvidenceValue(
            name="intent",
            score=intent_score,
            confidence=float(action_state.intent_confidence),
            available=True,
            metadata={
                "intent_state": action_state.intent_state,
                "action_explanation": action_explanation,
                "view_intent": getattr(action_state, "view_intent", "None"),
                "move_intent": getattr(action_state, "move_intent", "None"),
                "has_view_motion": bool(getattr(action_state, "has_view_motion", False)),
                "has_move_motion": bool(getattr(action_state, "has_move_motion", False)),
                "is_viewpoint_locomotion": action_explanation == "viewpoint_locomotion",
                "is_viewpoint_change": action_explanation == "viewpoint_change",
                "is_world_change_evidence": False,
                "rotation_speed": float(action_state.rotation_speed),
                "movement_speed": float(action_state.movement_speed),
                "is_view_rotation": bool(pose_state is not None and abs(pose_state.delta_yaw) > 0 and pose_state.movement_magnitude < 0.25),
                "is_forward_progression": bool(pose_state is not None and pose_state.delta_z > 0 and abs(pose_state.delta_yaw) < 0.25),
                **pose_dict,
            },
        )
        fusion_result = self.fusion_engine.fuse(evidences)

        return EvidenceBundle(
            primary_similarity=primary_result,
            evidences=evidences,
            similarity_results=similarity_results,
            action_state=action_state,
            pose_state=pose_state,
            fusion_result=fusion_result,
        )
