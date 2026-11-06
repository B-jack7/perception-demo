# Robot perception route — real reproducible demo

This project is a meeting-ready implementation of the **hybrid perception route** for stacked-object manipulation. It runs three real pretrained models and turns their outputs into an inspectable support graph:

1. **Grounding DINO Tiny** proposes task-relevant objects from an RGB image using an open-vocabulary text prompt.
2. **SAM 2.1 Hiera Tiny** converts the learned boxes into pixel-accurate masks.
3. **Depth Anything V2 Metric Indoor Small** estimates metric monocular depth.
4. An explicit geometric layer proposes `supported_by` relations and a safe removal order.

The demo contains no hand-authored boxes or masks. The final support relation is deliberately rule-based and exports its evidence, so it can be debugged and challenged in the meeting.

## Run

From this directory:

```bash
uv sync
uv run python run_demo.py
```

The first run downloads about 1–2 GB of public model weights into `.cache/huggingface`. Later runs reuse the local cache. CPU mode is the reproducible default and takes roughly tens of seconds on the current Apple Silicon Mac.

Optional MPS acceleration:

```bash
uv run python run_demo.py --device auto
```

The generated artifacts appear in `outputs/`:

- `01_grounding_dino_boxes.png` — raw learned proposals and confidence scores
- `02_sam2_masks.png` — learned pixel masks and object identity
- `03_metric_depth.png` — Depth Anything V2 output
- `04_support_graph.png` — geometric relation graph and removal plan
- `05_pipeline_summary.png` — one-slide live-demo summary
- `scene_graph.json` — machine-readable nodes, edges, evidence, model IDs, and timings
- `object_positions.json` — perception-only object positions for the interactive viewer
- `mask_*.png` and `metric_depth_m.npy` — raw intermediate products

## Interactive position viewer

After the model run finishes, open the local hover-to-inspect interface:

```bash
uv run python run_interactive.py
```

The browser view lets you switch between RGB and depth, toggle real SAM masks, and hover over an object to inspect its pixel centroid, estimated depth, approximate camera XYZ position, detector score, and mask quality. Use `--rerun` if you want the model inference and viewer launch in one command.

## What this proves—and what it does not

It proves that a practical, modular perception route can turn a real robot-scene RGB image into proposals, masks, depth estimates, and an explicit support hypothesis. It is suitable as a technical prototype and as a baseline architecture.

It is not yet a safety-certified manipulation stack. The sample lacks calibrated camera intrinsics, monocular depth is less reliable than a physical RGB-D sensor, and support edges should be stabilized across multiple frames before the robot executes a motion. Replacing the estimated depth with RealSense/Kinect depth is the natural next step; the graph layer can remain the same.

## Model and data sources

- Grounding DINO Tiny: <https://huggingface.co/IDEA-Research/grounding-dino-tiny>
- SAM 2.1 Hiera Tiny: <https://huggingface.co/facebook/sam2.1-hiera-tiny>
- Depth Anything V2 Metric Indoor Small: <https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf>
- CALVIN benchmark: <https://github.com/mees/calvin>
- Image pair used here: row 62 from the public CALVIN validation mirror at <https://huggingface.co/datasets/ShuaKang/calvin_abc_d_val>
