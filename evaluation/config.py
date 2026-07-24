from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExperimentMethod:
    name: str
    similarity_estimator: str = "orb"
    memory_scheduler: str = "stableworld"
    evidence_mode: str = "single"
    lightglue_spatial_alpha: float = 0.0
    depth_metric: str = "l1"
    fusion_mode: str = "weighted"
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "appearance": 0.25,
        "semantic": 0.25,
        "geometry": 0.25,
        "intent": 0.25,
    })
    notes: str = ""

    def command_args(self) -> List[str]:
        args = [
            "--evict_mode",
            "--stableworld_debug",
            "--similarity_estimator",
            self.similarity_estimator,
            "--memory_scheduler",
            self.memory_scheduler,
            "--evidence_mode",
            self.evidence_mode,
            "--fusion_mode",
            self.fusion_mode,
            "--fusion_weight_appearance",
            str(self.fusion_weights.get("appearance", 0.25)),
            "--fusion_weight_semantic",
            str(self.fusion_weights.get("semantic", 0.25)),
            "--fusion_weight_geometry",
            str(self.fusion_weights.get("geometry", 0.25)),
            "--fusion_weight_intent",
            str(self.fusion_weights.get("intent", 0.25)),
        ]
        if self.similarity_estimator == "lightglue":
            args += ["--lightglue_spatial_alpha", str(self.lightglue_spatial_alpha)]
        if self.similarity_estimator == "depth":
            args += ["--depth_metric", self.depth_metric]
        return args


@dataclass
class ExperimentConfig:
    output_root: str = "outputs/evaluation"
    inference_script: str = "inference.py"
    config_path: str = "configs/inference_yaml/inference_universal.yaml"
    image_path: str = "demo_images/universal/0011.png"
    pretrained_model_path: str = "Matrix-Game-2.0"
    checkpoint_path: str = ""
    depth_checkpoint: str = ""
    num_output_frames: int = 150
    seed: int = 0
    threshold: float = 0.78
    methods: List[ExperimentMethod] = field(default_factory=list)
    scenarios: List[str] = field(default_factory=lambda: [
        "idle",
        "slow_walk",
        "fast_run",
        "turn_in_place",
        "strafe",
        "near_door",
        "repeated_texture",
    ])


def default_experiment_suite() -> List[ExperimentMethod]:
    return [
        ExperimentMethod(
            name="baseline_matrix_game",
            similarity_estimator="orb",
            memory_scheduler="stableworld",
            notes="Matrix-Game2.0 without Phys-Mem scheduling; run with evict disabled manually when needed.",
        ),
        ExperimentMethod(name="stableworld_orb", similarity_estimator="orb"),
        ExperimentMethod(name="semantic_lightglue", similarity_estimator="lightglue"),
        ExperimentMethod(
            name="semantic_lightglue_penalty",
            similarity_estimator="lightglue",
            lightglue_spatial_alpha=0.001,
        ),
        ExperimentMethod(name="physical_depth", similarity_estimator="depth"),
        ExperimentMethod(name="depth_action", similarity_estimator="depth"),
        ExperimentMethod(name="fusion_debug", similarity_estimator="depth", evidence_mode="multi", fusion_mode="weighted"),
        ExperimentMethod(name="physmem", similarity_estimator="depth", memory_scheduler="physmem", evidence_mode="multi"),
    ]
