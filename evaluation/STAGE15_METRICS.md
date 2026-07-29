# Stage15 Evaluation Metrics

Stage15 is an offline evaluation layer. It does not modify ExperimentRecord,
Memory Scheduler, Memory Buffer, Generator, KV Policy, ORB, LightGlue, Depth, or
Fusion. Public metrics are executed through official implementations whenever
available.

## Existing Logs

Directly computable from current logs:

- Runtime
- MCS: Memory Consistency Score proxy from frame similarity
- PCS: Phys-Mem Consistency Score proxy from memory/fusion consistency
- FRR: False Refresh/Refresh-like Rate proxy from recorded decisions
- MSS: Memory Stability Score
- MRR: Memory Replacement Rate
- AFS: Average Fusion Score
- StrategyDistribution
- StrategyTransitionSmoothness

Need extra offline inference/evaluation:

- VBench Background Consistency
- VBench Subject Consistency
- VBench Motion Smoothness
- VBench Temporal Flickering
- VBench Imaging Quality
- STREAM Temporal Score
- STREAM Spatial Score
- DOVER Technical Quality
- DOVER Aesthetic Quality
- DOVER Overall Quality

Unsupported from logs alone:

- Any no-reference or learned perceptual video quality metric that requires
  reading generated video frames.
- Any text-video alignment metric that requires prompts not saved with the
  current run.
- Any human preference score without a human study protocol.

## Public Metrics

| Metric | Source | Official implementation | Input | Output | Current support |
| --- | --- | --- | --- | --- | --- |
| Background Consistency | VBench: Comprehensive Benchmark Suite for Video Generative Models | https://github.com/Vchitect/VBench | Generated videos, optional prompt/config files | JSON/report from official VBench | `run_vbench.py` wrapper |
| Subject Consistency | VBench | https://github.com/Vchitect/VBench | Generated videos, optional prompt/config files | JSON/report from official VBench | `run_vbench.py` wrapper |
| Motion Smoothness | VBench | https://github.com/Vchitect/VBench | Generated videos | JSON/report from official VBench | `run_vbench.py` wrapper |
| Temporal Flickering | VBench | https://github.com/Vchitect/VBench | Generated videos | JSON/report from official VBench | `run_vbench.py` wrapper |
| Imaging Quality | VBench | https://github.com/Vchitect/VBench | Generated videos | JSON/report from official VBench | `run_vbench.py` wrapper |
| Temporal Score | STREAM: Measuring Temporal Consistency in Video Generation | https://openreview.net/forum?id=fZwY0JQZes, https://github.com/pro2nit/STREAM | Generated videos and official STREAM assets | Official STREAM score file | `run_stream.py` command wrapper |
| Spatial Score | STREAM | https://openreview.net/forum?id=fZwY0JQZes, https://github.com/pro2nit/STREAM | Generated videos and official STREAM assets | Official STREAM score file | `run_stream.py` command wrapper |
| Technical Quality | DOVER: Evaluating Video Quality from Aesthetic and Technical Perspectives | https://github.com/VQAssessment/DOVER | Generated videos and official DOVER model | Official DOVER score file | `run_dover.py` command wrapper |
| Aesthetic Quality | DOVER | https://github.com/VQAssessment/DOVER | Generated videos and official DOVER model | Official DOVER score file | `run_dover.py` command wrapper |
| Overall Quality | DOVER | https://github.com/VQAssessment/DOVER | Generated videos and official DOVER model | Official DOVER score file | `run_dover.py` command wrapper |

## Recommended Table

| Method | VBench BackgroundConsistency ↑ | VBench TemporalFlickering ↓ | VBench MotionSmoothness ↑ | STREAM Temporal ↑ | STREAM Spatial ↑ | DOVER Technical ↑ | DOVER Overall ↑ | MCS ↑ | PCS ↑ | FRR ↓ | Runtime ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | | | | | | | | | | | |
| StableWorld ORB | | | | | | | | | | | |
| LightGlue | | | | | | | | | | | |
| LightGlue+Penalty | | | | | | | | | | | |
| Depth | | | | | | | | | | | |
| Fusion | | | | | | | | | | | |
| PhysMem | | | | | | | | | | | |

## Example Commands

VBench:

```bash
python evaluation/run_vbench.py \
  --video_dir outputs_v2/physmem_kv_stress \
  --output_dir evaluation/results/physmem/vbench
```

DOVER:

```bash
python evaluation/run_dover.py \
  --video_dir outputs_v2/physmem_kv_stress \
  --dover_repo /path/to/DOVER \
  --dover_command python evaluate.py --video_dir {video_dir} --output {output} \
  --output evaluation/results/physmem/dover.json
```

STREAM:

```bash
python evaluation/run_stream.py \
  --video_dir outputs_v2/physmem_kv_stress \
  --stream_repo /path/to/STREAM \
  --stream_command python evaluate.py --video_dir {video_dir} --output {output} \
  --output evaluation/results/physmem/stream.json
```

Memory metrics:

```bash
python evaluation/memory_metrics.py \
  --debug_dir outputs_v2/physmem_kv_stress/stableworld_debug \
  --output evaluation/results/physmem/memory_metrics.json
```

Aggregate:

```bash
python evaluation/aggregate_metrics.py \
  --method PhysMem \
  --debug_dir outputs_v2/physmem_kv_stress/stableworld_debug \
  --vbench_json evaluation/results/physmem/vbench/vbench_metrics.json \
  --stream_json evaluation/results/physmem/stream.json \
  --dover_json evaluation/results/physmem/dover.json \
  --output evaluation/results/paper_results.csv
```
