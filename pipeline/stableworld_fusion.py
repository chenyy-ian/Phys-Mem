from dataclasses import dataclass, field
import csv
import os
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class FusionWeight:
    appearance: float = 0.25
    semantic: float = 0.25
    geometry: float = 0.25
    intent: float = 0.25

    def as_dict(self) -> Dict[str, float]:
        return {
            "appearance": float(self.appearance),
            "semantic": float(self.semantic),
            "geometry": float(self.geometry),
            "intent": float(self.intent),
        }


@dataclass
class FusionConfig:
    mode: str = "weighted"
    weights: FusionWeight = field(default_factory=FusionWeight)
    dynamic_adjustment: bool = True
    renormalize_available: bool = True
    min_confidence: float = 1e-6


@dataclass
class EvidenceValue:
    name: str
    score: float
    confidence: float
    available: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class FusionResult:
    unified_memory_score: float
    evidence_confidence: float
    evidence_weights: Dict[str, float]
    evidence_contributions: Dict[str, float]
    evidence_scores: Dict[str, float]
    mode: str


class WeightManager:
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def resolve(self, evidences: Dict[str, EvidenceValue]) -> Dict[str, float]:
        base_weights = self.config.weights.as_dict()
        resolved = {}
        for name, base_weight in base_weights.items():
            evidence = evidences.get(name)
            if evidence is None or not evidence.available:
                resolved[name] = 0.0 if self.config.renormalize_available else float(base_weight)
                continue
            if self.config.dynamic_adjustment:
                resolved[name] = float(base_weight) * max(float(evidence.confidence), self.config.min_confidence)
            else:
                resolved[name] = float(base_weight)

        total = sum(resolved.values())
        if total > 0:
            return {name: value / total for name, value in resolved.items()}
        return {name: 0.0 for name in base_weights}


class BaseFusionStrategy:
    def fuse(self, evidences: Dict[str, EvidenceValue], weights: Dict[str, float], config: FusionConfig) -> FusionResult:
        raise NotImplementedError


class WeightedFusionStrategy(BaseFusionStrategy):
    def fuse(self, evidences: Dict[str, EvidenceValue], weights: Dict[str, float], config: FusionConfig) -> FusionResult:
        contributions = {}
        scores = {}
        confidence_terms = []
        for name, weight in weights.items():
            evidence = evidences.get(name)
            score = float(evidence.score) if evidence is not None and evidence.available else 0.0
            confidence = float(evidence.confidence) if evidence is not None and evidence.available else 0.0
            scores[name] = score
            contributions[name] = float(weight) * score
            confidence_terms.append(float(weight) * confidence)

        unified_score = float(np.clip(sum(contributions.values()), 0.0, 1.0))
        evidence_confidence = float(np.clip(sum(confidence_terms), 0.0, 1.0))
        return FusionResult(
            unified_memory_score=unified_score,
            evidence_confidence=evidence_confidence,
            evidence_weights=weights,
            evidence_contributions=contributions,
            evidence_scores=scores,
            mode=config.mode,
        )


class RuleBasedFusionStrategy(BaseFusionStrategy):
    def fuse(self, evidences: Dict[str, EvidenceValue], weights: Dict[str, float], config: FusionConfig) -> FusionResult:
        weighted = WeightedFusionStrategy().fuse(evidences, weights, config)
        penalties = []
        for name in ("appearance", "semantic", "geometry"):
            evidence = evidences.get(name)
            if evidence is not None and evidence.available and evidence.score < 0.35:
                penalties.append((0.35 - evidence.score) * weights.get(name, 0.0))
        intent = evidences.get("intent")
        if intent is not None and intent.available and intent.score < 0.5:
            penalties.append((0.5 - intent.score) * weights.get("intent", 0.0))
        score = float(np.clip(weighted.unified_memory_score - sum(penalties), 0.0, 1.0))
        weighted.unified_memory_score = score
        return weighted


class LearnedFusionStrategy(BaseFusionStrategy):
    def fuse(self, evidences: Dict[str, EvidenceValue], weights: Dict[str, float], config: FusionConfig) -> FusionResult:
        return WeightedFusionStrategy().fuse(evidences, weights, config)


class FusionEngine:
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.weight_manager = WeightManager(self.config)
        self.strategies = {
            "weighted": WeightedFusionStrategy(),
            "rule": RuleBasedFusionStrategy(),
            "rule_based": RuleBasedFusionStrategy(),
            "learned": LearnedFusionStrategy(),
            "learned_placeholder": LearnedFusionStrategy(),
        }

    @staticmethod
    def intent_to_memory_score(intent_state: str, intent_confidence: float) -> float:
        if intent_state in {"Idle", "Unknown", ""}:
            return 1.0
        return float(np.clip(1.0 - intent_confidence, 0.0, 1.0))

    def fuse(self, evidences: Dict[str, EvidenceValue]) -> FusionResult:
        weights = self.weight_manager.resolve(evidences)
        strategy = self.strategies.get(self.config.mode, self.strategies["weighted"])
        return strategy.fuse(evidences, weights, self.config)


class FusionBenchmark:
    @staticmethod
    def summarize(records: List[Dict], bucket_size: int = 100) -> List[Dict]:
        buckets = {}
        for record in records:
            bucket = (int(record["frame_index"]) // bucket_size) * bucket_size
            buckets.setdefault(bucket, []).append(record)

        rows = []
        for bucket in sorted(buckets):
            items = buckets[bucket]
            rows.append({
                "frame_bucket_start": bucket,
                "frame_bucket_end": bucket + bucket_size - 1,
                "average_unified_memory_score": float(np.mean([x["unified_memory_score"] for x in items])),
                "average_evidence_confidence": float(np.mean([x["evidence_confidence"] for x in items])),
                "appearance_contribution": float(np.mean([x["appearance_contribution"] for x in items])),
                "semantic_contribution": float(np.mean([x["semantic_contribution"] for x in items])),
                "geometry_contribution": float(np.mean([x["geometry_contribution"] for x in items])),
                "intent_contribution": float(np.mean([x["intent_contribution"] for x in items])),
            })
        return rows


class FusionVisualizer:
    @staticmethod
    def _draw_bar_chart(labels: List[str], values: List[float], output_path: str, title: str):
        width = 900
        height = 520
        canvas = np.zeros((height, width, 3), dtype=np.uint8) + 255
        cv2.putText(canvas, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 40, 40), 2)
        max_value = max(max(values), 1e-6)
        colors = [(70, 120, 220), (80, 170, 90), (210, 120, 60), (170, 80, 180)]
        bar_w = 120
        gap = 70
        x0 = 80
        baseline = 430
        for idx, (label, value) in enumerate(zip(labels, values)):
            x = x0 + idx * (bar_w + gap)
            bar_h = int((float(value) / max_value) * 320)
            cv2.rectangle(canvas, (x, baseline - bar_h), (x + bar_w, baseline), colors[idx % len(colors)], -1)
            cv2.putText(canvas, f"{value:.3f}", (x, baseline - bar_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
            cv2.putText(canvas, label, (x, baseline + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        cv2.imwrite(output_path, canvas)

    @staticmethod
    def save(records: List[Dict], output_dir: str):
        if not records:
            return
        os.makedirs(output_dir, exist_ok=True)
        labels = ["appearance", "semantic", "geometry", "intent"]
        last = records[-1]
        FusionVisualizer._draw_bar_chart(
            labels,
            [last[f"{label}_weight"] for label in labels],
            os.path.join(output_dir, "fusion_weight_distribution.png"),
            "Evidence Weight Distribution",
        )
        FusionVisualizer._draw_bar_chart(
            labels,
            [last[f"{label}_contribution"] for label in labels],
            os.path.join(output_dir, "fusion_evidence_contribution.png"),
            "Evidence Contribution",
        )
        FusionVisualizer._draw_curve(records, output_dir)

    @staticmethod
    def _draw_curve(records: List[Dict], output_dir: str):
        width = 1200
        height = 680
        margin_left = 80
        margin_right = 40
        plot_w = width - margin_left - margin_right
        canvas = np.zeros((height, width, 3), dtype=np.uint8) + 255
        xs = np.array([x["frame_index"] for x in records], dtype=np.float32)
        x_min = float(xs.min())
        x_max = float(max(xs.max(), x_min + 1.0))

        def sx(x):
            return int(margin_left + ((float(x) - x_min) / (x_max - x_min)) * plot_w)

        def draw_series(key, y_top, y_bottom, color, label):
            values = np.array([x[key] for x in records], dtype=np.float32)
            max_v = max(float(values.max()), 1.0)
            pts = []
            for x, value in zip(xs, values):
                px = sx(x)
                py = int(y_bottom - (float(value) / max_v) * (y_bottom - y_top))
                pts.append((px, py))
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (20, y_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            cv2.line(canvas, (margin_left, y_bottom), (width - margin_right, y_bottom), (215, 215, 215), 1)

        draw_series("unified_memory_score", 50, 210, (65, 110, 220), "Unified Memory Score")
        draw_series("evidence_confidence", 260, 420, (80, 160, 80), "Evidence Confidence")
        draw_series("intent_score", 470, 630, (180, 80, 170), "Intent Memory Score")
        cv2.putText(canvas, "Frame Index", (520, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1)
        cv2.imwrite(os.path.join(output_dir, "fusion_evidence_curve.png"), canvas)
