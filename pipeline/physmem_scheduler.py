from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .stableworld_memory import MemoryBuffer, MemoryDecision, MemoryPolicy, MemoryScheduler
from .stableworld_similarity import SimilarityResult


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
class StrategyStabilityConfig:
    replace_cooldown: int = 5
    evict_cooldown: int = 8
    insert_enter_probability: float = 0.35
    replace_enter_probability: float = 0.55
    replace_exit_probability: float = 0.40
    evict_enter_probability: float = 0.80
    evict_exit_probability: float = 0.65
    allow_direct_keep_to_evict: bool = False


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


class PhysMemStateMachine:
    ORDER = {"KEEP": 0, "INSERT": 1, "REPLACE": 2, "EVICT": 3, "REFRESH": 1}

    def __init__(self, policy: MemoryPolicy, stability: StrategyStabilityConfig | None = None):
        self.policy = policy
        self.stability = stability or StrategyStabilityConfig()
        self.last_state = "KEEP"
        self.replace_cooldown_remaining = 0
        self.evict_cooldown_remaining = 0

    def transition(self, proposal: MemoryProposal, validation: EvidenceValidation) -> tuple[str, str]:
        candidate, reason = self._candidate_transition(proposal, validation)
        constrained, constraint_reason = self._apply_transition_constraints(candidate, proposal, validation)
        self._update_cooldown(constrained)
        previous = self.last_state
        self.last_state = constrained
        if constraint_reason:
            return constrained, f"{reason}|{constraint_reason}|prev={previous}"
        return constrained, f"{reason}|prev={previous}"

    def _candidate_transition(self, proposal: MemoryProposal, validation: EvidenceValidation) -> tuple[str, str]:
        if validation.geometry == "reject":
            return "REFRESH", "proposal_validation_geometry_reject"

        if proposal.state == "KEEP":
            if validation.intent == "reject" and validation.world_change_probability >= self.stability.replace_enter_probability:
                return "REPLACE", "proposal_keep_rejected_by_world_change"
            if validation.reject_count >= 2:
                return "INSERT", "proposal_keep_soft_reject"
            return "KEEP", "proposal_keep_validated"

        if validation.intent == "reject" and validation.intent_explanation == "viewpoint_change":
            return "KEEP", "proposal_insert_explained_by_camera_motion"
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
    ) -> tuple[str, str]:
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
    ):
        super().__init__(sim_threshold=sim_threshold)
        self.policy = policy or MemoryPolicy(stable_score=sim_threshold)
        self.stability = stability or StrategyStabilityConfig()
        self.proposal_engine = ProposalEngine(self.policy)
        self.validator = EvidenceValidator(self.policy)
        self.state_machine = PhysMemStateMachine(self.policy, self.stability)

    def schedule(
        self,
        memory_buffer: MemoryBuffer,
        similarity: SimilarityResult,
        fusion_result: Any = None,
        unified_memory_score: float | None = None,
        geometry_confidence: float = 1.0,
        intent_state: str = "Unknown",
        intent_confidence: float = 0.0,
    ) -> MemoryDecision:
        proposal = self.proposal_engine.propose(similarity, fusion_result)
        validation = self.validator.validate(
            proposal,
            fusion_result=fusion_result,
            geometry_confidence=geometry_confidence,
            intent_state=intent_state,
            intent_confidence=intent_confidence,
        )
        memory_state, transition = self.state_machine.transition(proposal, validation)
        keep_ids, delete_range, refresh_ids, insert_count, kv_policy, evict_middle = self._strategy_plan(memory_buffer, memory_state)
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

    def _strategy_plan(self, memory_buffer: MemoryBuffer, memory_state: str) -> tuple[list[int], list[int], list[int], int, str, int]:
        ids = memory_buffer.snapshot()
        if memory_state == "KEEP":
            delete_ids = ids[3:6]
            keep_ids = ids[:3] + ids[6:]
            return keep_ids, delete_ids, [], 3, "preserve_anchor", 1
        if memory_state == "REFRESH":
            delete_ids = [ids[2], ids[5]]
            keep_ids = [frame_id for frame_id in ids if frame_id not in set(delete_ids)]
            return keep_ids, delete_ids, delete_ids, 2, "refresh_uncertain", 1
        if memory_state == "INSERT":
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
