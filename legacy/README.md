# Legacy Prototype

This directory archives the earlier perception experiment based on:

- Grounding DINO object proposals
- SAM 2 masks
- Depth Anything V2 monocular metric-depth estimation
- Rule-based support relations

It is retained for reproducibility and comparison only. It is **not** the current proposed technical route.

Run the archived experiment from this directory:

```bash
uv sync
uv run python run_demo.py
```

The current project uses real RGB-D geometry as the intended metric source and treats semantic models as labels for already-separated 3D instances.
