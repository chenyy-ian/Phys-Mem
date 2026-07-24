from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


EXPERIMENT_RECORD_FIELDS = [
    "ExperimentID",
    "FrameID",
    "Timestamp",
    "CurrentFrameID",
    "ReferenceFrameID",
    "MiddleFrameID",
    "Similarity",
    "SimilarityType",
    "MatchingPoints",
    "Confidence",
    "Decision",
    "MemoryState",
    "MemoryID",
    "MemorySize",
    "ReplaceCount",
    "KeepCount",
    "Runtime",
    "CPUTime",
    "GPUTime",
    "VRAM",
    "FPS",
    "GeometryScore",
    "DepthConsistency",
    "CameraMotion",
    "ActionEmbedding",
    "ActionType",
    "ActionMagnitude",
    "FusionWeight",
    "FusionEntropy",
    "FusionDecision",
    "SemanticScore",
    "SemanticConfidence",
    "PhysicalScore",
    "PhysicalConfidence",
    "EvidenceType",
    "PolicyConfidence",
    "DecisionConfidence",
    "ExtraFields",
]


@dataclass
class ExperimentRecord:
    ExperimentID: str
    FrameID: int | None = None
    Timestamp: float = field(default_factory=time.time)
    CurrentFrameID: int | None = None
    ReferenceFrameID: int | None = None
    MiddleFrameID: int | None = None
    Similarity: float | None = None
    SimilarityType: str | None = None
    MatchingPoints: int | None = None
    Confidence: float | None = None
    Decision: str | None = None
    MemoryState: str | None = None
    MemoryID: str | None = None
    MemorySize: int | None = None
    ReplaceCount: int | None = None
    KeepCount: int | None = None
    Runtime: float | None = None
    CPUTime: float | None = None
    GPUTime: float | None = None
    VRAM: float | None = None
    FPS: float | None = None
    GeometryScore: float | None = None
    DepthConsistency: float | None = None
    CameraMotion: str | None = None
    ActionEmbedding: str | None = None
    ActionType: str | None = None
    ActionMagnitude: float | None = None
    FusionWeight: float | None = None
    FusionEntropy: float | None = None
    FusionDecision: str | None = None
    SemanticScore: float | None = None
    SemanticConfidence: float | None = None
    PhysicalScore: float | None = None
    PhysicalConfidence: float | None = None
    EvidenceType: str | None = None
    PolicyConfidence: float | None = None
    DecisionConfidence: float | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, experiment_id: str, **kwargs: Any) -> "ExperimentRecord":
        known = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in EXPERIMENT_RECORD_FIELDS}
        known.pop("ExperimentID", None)
        known.pop("ExtraFields", None)
        record = cls(ExperimentID=experiment_id, **known)
        record.extra_fields.update(kwargs)
        return record

    def to_dict(self) -> dict[str, Any]:
        row = {}
        for field_name in EXPERIMENT_RECORD_FIELDS:
            if field_name == "ExtraFields":
                row[field_name] = json.dumps(self.extra_fields, ensure_ascii=False) if self.extra_fields else ""
            else:
                row[field_name] = getattr(self, field_name)
        return row


class ExperimentDatabase:
    """
    File-backed raw experiment database. Stage10 metric scripts should read these
    records and compute paper metrics offline.
    """

    def __init__(self, root_dir: str = "outputs/experiment_tracking"):
        self.root_dir = root_dir
        self.experiments_dir = os.path.join(root_dir, "experiments")
        self.logs_dir = os.path.join(root_dir, "logs")
        self.results_dir = os.path.join(root_dir, "results")
        self.configs_dir = os.path.join(root_dir, "configs")
        self.metadata_dir = os.path.join(root_dir, "metadata")
        for path in [
            self.experiments_dir,
            self.logs_dir,
            self.results_dir,
            self.configs_dir,
            self.metadata_dir,
        ]:
            os.makedirs(path, exist_ok=True)

    def next_experiment_id(self) -> str:
        existing = []
        if os.path.isdir(self.experiments_dir):
            for name in os.listdir(self.experiments_dir):
                if name.startswith("Experiment") and name[len("Experiment"):].isdigit():
                    existing.append(int(name[len("Experiment"):]))
        return f"Experiment{(max(existing) + 1) if existing else 1:03d}"

    def create_experiment(self, experiment_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        experiment_id = experiment_id or self.next_experiment_id()
        for subdir in ["logs", "results", "configs", "metadata", "visualizations"]:
            os.makedirs(os.path.join(self.experiments_dir, experiment_id, subdir), exist_ok=True)
        if metadata:
            metadata_path = self.metadata_path(experiment_id)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        return experiment_id

    def record_path_csv(self, experiment_id: str) -> str:
        return os.path.join(self.experiments_dir, experiment_id, "logs", "experiment_records.csv")

    def record_path_json(self, experiment_id: str) -> str:
        return os.path.join(self.experiments_dir, experiment_id, "logs", "experiment_records.json")

    def metadata_path(self, experiment_id: str) -> str:
        return os.path.join(self.experiments_dir, experiment_id, "metadata", "metadata.json")

    def load_records(self, experiment_id: str) -> list[dict[str, Any]]:
        json_path = self.record_path_json(experiment_id)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        csv_path = self.record_path_csv(experiment_id)
        if not os.path.exists(csv_path):
            return []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


class FrameRecorder:
    def collect(self, frame_id: int | None = None, current_frame_id: int | None = None, **_: Any) -> dict[str, Any]:
        return {
            "FrameID": frame_id,
            "CurrentFrameID": current_frame_id if current_frame_id is not None else frame_id,
        }


class SimilarityRecorder:
    def collect(
        self,
        similarity: float | None = None,
        similarity_type: str | None = None,
        matching_points: int | None = None,
        confidence: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "Similarity": similarity,
            "SimilarityType": similarity_type,
            "MatchingPoints": matching_points,
            "Confidence": confidence,
        }


class MemoryRecorder:
    def collect(
        self,
        decision: str | None = None,
        memory_state: Iterable[int] | str | None = None,
        memory_id: str | None = None,
        memory_size: int | None = None,
        replace_count: int | None = None,
        keep_count: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if memory_state is not None and not isinstance(memory_state, str):
            memory_state = " ".join(str(x) for x in memory_state)
        return {
            "Decision": decision,
            "MemoryState": memory_state,
            "MemoryID": memory_id,
            "MemorySize": memory_size,
            "ReplaceCount": replace_count,
            "KeepCount": keep_count,
        }


class RuntimeRecorder:
    def collect(
        self,
        runtime: float | None = None,
        cpu_time: float | None = None,
        gpu_time: float | None = None,
        vram: float | None = None,
        fps: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "Runtime": runtime,
            "CPUTime": cpu_time,
            "GPUTime": gpu_time,
            "VRAM": vram,
            "FPS": fps,
        }


class GeometryRecorder:
    def collect(
        self,
        geometry_score: float | None = None,
        depth_consistency: float | None = None,
        camera_motion: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "GeometryScore": geometry_score,
            "DepthConsistency": depth_consistency,
            "CameraMotion": camera_motion,
        }


class ActionRecorder:
    def collect(
        self,
        action_embedding: str | None = None,
        action_type: str | None = None,
        action_magnitude: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "ActionEmbedding": action_embedding,
            "ActionType": action_type,
            "ActionMagnitude": action_magnitude,
        }


class FusionRecorder:
    def collect(
        self,
        fusion_weight: float | None = None,
        fusion_entropy: float | None = None,
        fusion_decision: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "FusionWeight": fusion_weight,
            "FusionEntropy": fusion_entropy,
            "FusionDecision": fusion_decision,
        }


class Exporter:
    def __init__(self, csv_path: str, json_path: str):
        self.csv_path = csv_path
        self.json_path = json_path
        self.records: list[ExperimentRecord] = []

    def append(self, record: ExperimentRecord) -> None:
        self.records.append(record)

    def flush(self) -> None:
        rows = [record.to_dict() for record in self.records]
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EXPERIMENT_RECORD_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


class TimelineView:
    def spec(self) -> dict[str, str]:
        return {"view": "timeline", "source": "ExperimentRecord"}


class CurveView:
    def spec(self) -> dict[str, str]:
        return {"view": "curve", "source": "ExperimentRecord"}


class HistogramView:
    def spec(self) -> dict[str, str]:
        return {"view": "histogram", "source": "ExperimentRecord"}


class HeatmapView:
    def spec(self) -> dict[str, str]:
        return {"view": "heatmap", "source": "ExperimentRecord"}


class ExperimentRecorder:
    def __init__(
        self,
        enabled: bool = True,
        database_root: str = "outputs/experiment_tracking",
        experiment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.enabled = enabled
        self.database = ExperimentDatabase(database_root)
        self.experiment_id = self.database.create_experiment(experiment_id, metadata=metadata or {}) if enabled else ""
        self.exporter = Exporter(
            self.database.record_path_csv(self.experiment_id),
            self.database.record_path_json(self.experiment_id),
        ) if enabled else None
        self.frame_recorder = FrameRecorder()
        self.similarity_recorder = SimilarityRecorder()
        self.memory_recorder = MemoryRecorder()
        self.runtime_recorder = RuntimeRecorder()
        self.geometry_recorder = GeometryRecorder()
        self.action_recorder = ActionRecorder()
        self.fusion_recorder = FusionRecorder()
        self.visualizations = {
            "timeline": TimelineView(),
            "curve": CurveView(),
            "histogram": HistogramView(),
            "heatmap": HeatmapView(),
        }

    def record(self, **kwargs: Any) -> None:
        if not self.enabled or self.exporter is None:
            return

        data = {}
        data.update(self.frame_recorder.collect(**kwargs))
        data.update(self.similarity_recorder.collect(**kwargs))
        data.update(self.memory_recorder.collect(**kwargs))
        data.update(self.runtime_recorder.collect(**kwargs))
        data.update(self.geometry_recorder.collect(**kwargs))
        data.update(self.action_recorder.collect(**kwargs))
        data.update(self.fusion_recorder.collect(**kwargs))

        passthrough = {
            key: value
            for key, value in kwargs.items()
            if key in EXPERIMENT_RECORD_FIELDS
        }
        data.update(passthrough)

        extras = {
            key: value
            for key, value in kwargs.items()
            if key not in EXPERIMENT_RECORD_FIELDS
        }
        data.update(extras)
        self.exporter.append(ExperimentRecord.from_kwargs(self.experiment_id, **data))

    def flush(self) -> None:
        if self.enabled and self.exporter is not None:
            self.exporter.flush()

    def close(self) -> None:
        self.flush()

    def visualization_interface(self) -> dict[str, dict[str, str]]:
        return {name: view.spec() for name, view in self.visualizations.items()}
