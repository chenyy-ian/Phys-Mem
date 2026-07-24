from dataclasses import dataclass, field
from typing import Dict, List

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
    low_geometry_confidence: float = 0.35
    high_intent_confidence: float = 0.70
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
        unified_memory_score: float | None = None,
        geometry_confidence: float = 1.0,
        intent_state: str = "Unknown",
        intent_confidence: float = 0.0,
    ) -> MemoryDecision:
        score = float(similarity.similarity if unified_memory_score is None else unified_memory_score)
        context = {
            "unified_memory_score": score,
            "geometry_confidence": float(geometry_confidence),
            "intent_state": intent_state or "Unknown",
            "intent_confidence": float(intent_confidence),
        }
        memory_state, transition = self.state_machine.transition(context)
        evict_middle_by_state = {
            "KEEP": 1,
            "REFRESH": 1,
            "INSERT": 0,
            "REPLACE": 0,
            "EVICT": 0,
        }
        evict_middle = evict_middle_by_state[memory_state]
        delete_range = memory_buffer.delete(3, 6) if evict_middle else memory_buffer.delete(0, 3)
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
        )
