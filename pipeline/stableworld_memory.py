from dataclasses import dataclass, field
from typing import Any, Dict, List

from .stableworld_similarity import SimilarityResult


@dataclass
class MemoryDecision:
    evict_middle: int
    delete_range: List[int]
    decision: str
    similarity: float
    confidence: float
    matching_points: int
    memory_state: str = "REPLACE"
    policy: str = "stableworld"
    transition: str = ""
    unified_memory_score: float | None = None
    geometry_confidence: float | None = None
    intent_state: str | None = None
    keep_ids: List[int] = field(default_factory=list)
    delete_ids: List[int] = field(default_factory=list)
    refresh_ids: List[int] = field(default_factory=list)
    insert_count: int = 3
    kv_policy: str = "legacy"

    @property
    def delete_frame(self) -> int:
        return int(self.delete_range[0])


class MemoryBuffer:
    def __init__(self, window_ids: list[int]):
        self.window_ids = list(window_ids)

    def __len__(self) -> int:
        return len(self.window_ids)

    def snapshot(self) -> list[int]:
        return list(self.window_ids)

    @property
    def last_id(self) -> int:
        return self.window_ids[-1]

    @property
    def reference_frame_id(self) -> int:
        return self.window_ids[2]

    @property
    def middle_frame_id(self) -> int:
        return self.window_ids[5]

    @property
    def current_frame_id(self) -> int:
        return self.last_id + 1

    def keep(self, ids: list[int]) -> list[int]:
        return list(ids)

    def delete(self, start: int, end: int) -> list[int]:
        return self.window_ids[start:end]

    def insert(self, kept: list[int], count: int = 3) -> list[int]:
        next_ids = [self.last_id + i + 1 for i in range(count)]
        return kept + next_ids

    def replace(self, new_ids: list[int]) -> list[int]:
        assert len(new_ids) == len(self.window_ids), f"Window length should remain {len(self.window_ids)}, got {len(new_ids)}"
        self.window_ids = list(new_ids)
        return self.snapshot()

    def apply_decision(self, decision: MemoryDecision) -> list[int]:
        if decision.evict_middle:
            kept = self.keep(self.window_ids[:3] + self.window_ids[6:])
        else:
            kept = self.keep(self.window_ids[3:])
        new_ids = self.insert(kept, count=3)
        return self.replace(new_ids)

    def apply_strategy(self, decision: MemoryDecision) -> list[int]:
        keep_ids = decision.keep_ids or [frame_id for frame_id in self.window_ids if frame_id not in set(decision.delete_ids)]
        keep_set = set(keep_ids)
        delete_set = set(decision.delete_ids)
        kept = [frame_id for frame_id in self.window_ids if frame_id in keep_set and frame_id not in delete_set]
        new_ids = self.insert(kept, count=decision.insert_count)
        if len(new_ids) > len(self.window_ids):
            new_ids = new_ids[-len(self.window_ids):]
        if len(new_ids) < len(self.window_ids):
            new_ids = self.insert(new_ids, count=len(self.window_ids) - len(new_ids))
        return self.replace(new_ids)


class MemoryScheduler:
    def __init__(self, sim_threshold: float):
        self.sim_threshold = sim_threshold

    def schedule(self, memory_buffer: MemoryBuffer, similarity: SimilarityResult) -> MemoryDecision:
        if similarity.similarity >= self.sim_threshold:
            return MemoryDecision(
                evict_middle=1,
                delete_range=memory_buffer.delete(3, 6),
                decision="delete_middle",
                similarity=float(similarity.similarity),
                confidence=float(similarity.confidence),
                matching_points=int(similarity.matching_points),
            )
        return MemoryDecision(
            evict_middle=0,
            delete_range=memory_buffer.delete(0, 3),
            decision="delete_oldest",
            similarity=float(similarity.similarity),
            confidence=float(similarity.confidence),
            matching_points=int(similarity.matching_points),
        )


@dataclass
class MemoryPolicy:
    name: str = "physmem_rule"
    stable_score: float = 0.78
    refresh_score: float = 0.58
    refresh_anchor_score: float = 0.66
    reliable_geometry_confidence: float = 0.88
    low_geometry_confidence: float = 0.35
    high_intent_confidence: float = 0.50
    active_intents: set[str] = field(default_factory=lambda: {
        "Walk",
        "Run",
        "Forward",
        "Backward",
        "Left",
        "Right",
        "Turn Left",
        "Turn Right",
        "Jump",
    })


@dataclass
class DecisionRule:
    name: str
    target_state: str
    priority: int

    def matches(self, context: Dict, policy: MemoryPolicy) -> bool:
        checks = {
            "intent_turn": lambda: (
                context["intent_state"] in {"Turn Left", "Turn Right"}
                and context["intent_confidence"] >= policy.high_intent_confidence
            ),
            "geometry_uncertain": lambda: context["geometry_confidence"] < policy.low_geometry_confidence,
            "anchor_unreliable": lambda: (
                context["unified_memory_score"] < policy.refresh_anchor_score
                and context["geometry_confidence"] >= policy.reliable_geometry_confidence
                and context["intent_state"] not in {"Turn Left", "Turn Right"}
            ),
            "stable_world": lambda: context["unified_memory_score"] >= policy.stable_score,
            "transitional_world": lambda: context["unified_memory_score"] >= policy.refresh_score,
            "active_motion": lambda: context["intent_state"] in policy.active_intents,
            "default": lambda: True,
        }
        return checks[self.name]()


class MemoryStateMachine:
    def __init__(self, policy: MemoryPolicy | None = None):
        self.policy = policy or MemoryPolicy()
        self.rules = sorted([
            DecisionRule("intent_turn", "EVICT", 100),
            DecisionRule("geometry_uncertain", "REFRESH", 90),
            DecisionRule("anchor_unreliable", "REFRESH", 85),
            DecisionRule("stable_world", "KEEP", 80),
            DecisionRule("transitional_world", "INSERT", 70),
            DecisionRule("active_motion", "REPLACE", 60),
            DecisionRule("default", "REPLACE", 0),
        ], key=lambda rule: rule.priority, reverse=True)

    def transition(self, context: Dict) -> tuple[str, str]:
        for rule in self.rules:
            if rule.matches(context, self.policy):
                return rule.target_state, rule.name
        return "REPLACE", "fallback"


class PhysMemScheduler(MemoryScheduler):
    def __init__(self, sim_threshold: float, policy: MemoryPolicy | None = None):
        super().__init__(sim_threshold=sim_threshold)
        self.policy = policy or MemoryPolicy(stable_score=sim_threshold)
        self.state_machine = MemoryStateMachine(self.policy)

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
        if fusion_result is not None:
            unified_memory_score = float(fusion_result.unified_memory_score)
        score = float(similarity.similarity if unified_memory_score is None else unified_memory_score)
        context = {
            "unified_memory_score": score,
            "geometry_confidence": float(geometry_confidence),
            "intent_state": intent_state or "Unknown",
            "intent_confidence": float(intent_confidence),
        }
        memory_state, transition = self.state_machine.transition(context)
        keep_ids, delete_range, refresh_ids, insert_count, kv_policy, evict_middle = self._strategy_plan(memory_buffer, memory_state)
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
            keep_ids = ids[-2:]
            delete_ids = ids[:-2]
            return keep_ids, delete_ids, [], len(delete_ids), "hard_evict", 0
        delete_ids = ids[:3]
        keep_ids = ids[3:]
        return keep_ids, delete_ids, [], 3, "replace_stale", 0
