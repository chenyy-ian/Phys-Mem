from dataclasses import dataclass

from .stableworld_similarity import SimilarityResult


@dataclass
class MemoryDecision:
    evict_middle: int
    delete_range: list[int]
    decision: str
    similarity: float
    confidence: float
    matching_points: int

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
