#!/usr/bin/env python3
"""Run a real hybrid perception pipeline on a tabletop manipulation scene.

The learned components are Grounding DINO, SAM 2.1, and Depth Anything V2.
The support graph is an explicitly inspectable geometric reasoning layer built
from their outputs. Nothing in this script uses hand-authored object boxes or
masks.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID = "facebook/sam2.1-hiera-tiny"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

DISPLAY_COLORS = {
    "red_block": "#ef4444",
    "pink_block": "#ec4899",
    "blue_block": "#2563eb",
}

COLOR_PROTOTYPES_RGB = {
    "red_block": (220, 35, 35),
    "pink_block": (225, 35, 175),
    "blue_block": (35, 65, 215),
    "black_block": (25, 25, 25),
    "yellow_ball": (235, 220, 35),
    "gray_or_robot": (125, 125, 125),
    "wood_or_background": (145, 95, 55),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounding DINO → SAM 2.1 → metric depth → support graph"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_DIR / "data" / "calvin_unstack_before.jpg",
        help="RGB input image",
    )
    parser.add_argument(
        "--after-image",
        type=Path,
        default=PROJECT_DIR / "data" / "calvin_unstack_after.jpg",
        help="Optional comparison image showing the unstacked result",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "outputs", help="Output directory"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
        help="cpu is the most reproducible; auto uses MPS when available",
    )
    parser.add_argument(
        "--target",
        default="pink_block",
        help="Object requested for removal; blockers are planned first",
    )
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable on this machine")
    return requested


def tensor_inputs_to(inputs, device: str):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def nms(boxes: list[np.ndarray], scores: list[float], threshold: float = 0.60) -> list[int]:
    keep: list[int] = []
    for idx in np.argsort(scores)[::-1]:
        if all(box_iou(boxes[idx], boxes[other]) < threshold for other in keep):
            keep.append(int(idx))
    return keep


def run_detection(image: Image.Image, device: str) -> list[dict]:
    processor = AutoProcessor.from_pretrained(DETECTOR_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID).to(device)
    model.eval()
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    prompts: list[str] = []
    image_area = float(image.width * image.height)
    # The general prompt finds the scene objects; the second open-vocabulary
    # prompt resolves the small upper cube that otherwise merges with its
    # supporting cube. Both prompts are processed by the same learned model.
    for prompt in ("small object", "blue cube"):
        inputs = tensor_inputs_to(
            processor(images=image, text=[[prompt]], return_tensors="pt"), device
        )
        with torch.inference_mode():
            raw = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            raw,
            threshold=0.14 if prompt == "blue cube" else 0.18,
            text_threshold=0.14,
            target_sizes=[(image.height, image.width)],
        )[0]
        for box, score in zip(result["boxes"], result["scores"]):
            b = box.detach().cpu().numpy().astype(float)
            area_ratio = ((b[2] - b[0]) * (b[3] - b[1])) / image_area
            if 0.004 <= area_ratio <= 0.055:
                boxes.append(b)
                scores.append(float(score.detach().cpu()))
                prompts.append(prompt)

    selected = nms(boxes, scores)
    detections = [
        {
            "proposal_id": f"proposal_{rank + 1}",
            "box": boxes[idx].round(2).tolist(),
            "detector_score": round(scores[idx], 4),
            "detector_prompt": prompts[idx],
        }
        for rank, idx in enumerate(selected)
    ]
    del model, processor, inputs, raw
    gc.collect()
    return detections


def run_segmentation(image: Image.Image, detections: list[dict], device: str) -> tuple[np.ndarray, list[float]]:
    boxes = [item["box"] for item in detections]
    processor = Sam2Processor.from_pretrained(SEGMENTER_ID)
    model = Sam2Model.from_pretrained(SEGMENTER_ID).to(device)
    model.eval()
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
    inputs = tensor_inputs_to(inputs, device)
    with torch.inference_mode():
        raw = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        raw.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
    )[0]
    masks_np = masks[:, 0].numpy().astype(bool)
    quality = raw.iou_scores.detach().cpu().reshape(-1).numpy().astype(float).tolist()
    del model, processor, inputs, raw
    gc.collect()
    return masks_np, quality


def representative_rgb(image_np: np.ndarray, mask: np.ndarray, box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
    box_mask = np.zeros(mask.shape, dtype=bool)
    box_mask[y1:y2, x1:x2] = True
    pixels = image_np[mask & box_mask]
    if len(pixels) < 8:
        pixels = image_np[y1:y2, x1:x2].reshape(-1, 3)
    hsv = mcolors.rgb_to_hsv(pixels.astype(float) / 255.0)
    informative = (hsv[:, 1] > 0.35) | (hsv[:, 2] < 0.28)
    if informative.sum() >= 8:
        pixels = pixels[informative]
    return np.median(pixels, axis=0)


def color_distance(rgb: np.ndarray, prototype: tuple[int, int, int]) -> float:
    sample = mcolors.rgb_to_hsv(np.asarray(rgb, dtype=float)[None, :] / 255.0)[0]
    target = mcolors.rgb_to_hsv(np.asarray(prototype, dtype=float)[None, :] / 255.0)[0]
    hue_delta = min(abs(sample[0] - target[0]), 1.0 - abs(sample[0] - target[0]))
    return float(2.5 * hue_delta**2 + 1.2 * (sample[1] - target[1]) ** 2 + (sample[2] - target[2]) ** 2)


def label_and_filter_objects(
    image: Image.Image,
    detections: list[dict],
    masks: np.ndarray,
    mask_quality: list[float],
) -> tuple[list[dict], np.ndarray]:
    image_np = np.asarray(image)
    candidates: list[dict] = []
    for detection, mask, quality in zip(detections, masks, mask_quality):
        rgb = representative_rgb(image_np, mask, detection["box"])
        label = min(
            COLOR_PROTOTYPES_RGB,
            key=lambda name: color_distance(rgb, COLOR_PROTOTYPES_RGB[name]),
        )
        item = dict(detection)
        item.update(
            {
                "label": label,
                "mask_quality": round(float(quality), 4),
                "representative_rgb": [round(float(v), 1) for v in rgb],
            }
        )
        candidates.append(item)

    # Keep one high-quality instance for each task-relevant color. The location
    # and mask remain learned-model outputs; color identity is a transparent,
    # inspectable descriptor rather than another opaque model.
    chosen_indices: list[int] = []
    for label in DISPLAY_COLORS:
        matches = [i for i, item in enumerate(candidates) if item["label"] == label]
        if matches:
            chosen_indices.append(
                max(
                    matches,
                    key=lambda i: candidates[i]["detector_score"] * candidates[i]["mask_quality"],
                )
            )
    objects = [candidates[i] for i in chosen_indices]
    object_masks = masks[chosen_indices] if chosen_indices else np.empty((0, image.height, image.width), bool)
    objects.sort(key=lambda item: item["label"])
    if objects:
        # Reorder masks to match the sorted objects.
        label_to_mask = {candidates[i]["label"]: masks[i] for i in chosen_indices}
        object_masks = np.stack([label_to_mask[item["label"]] for item in objects])
    for idx, item in enumerate(objects):
        item["object_id"] = item["label"]
        item["mask_pixels"] = int(object_masks[idx].sum())
    return objects, object_masks


def run_depth(image: Image.Image, device: str) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(DEPTH_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(device)
    model.eval()
    inputs = tensor_inputs_to(processor(images=image, return_tensors="pt"), device)
    with torch.inference_mode():
        raw = model(**inputs)
    depth = processor.post_process_depth_estimation(
        raw, target_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"]
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    del model, processor, inputs, raw
    gc.collect()
    return depth_np


def enrich_geometry(objects: list[dict], masks: np.ndarray, depth: np.ndarray) -> None:
    height, width = depth.shape
    fx = fy = 0.8 * max(width, height)  # demo approximation; dataset intrinsics are absent
    cx, cy = width / 2.0, height / 2.0
    for item, mask in zip(objects, masks):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        z = float(np.median(depth[mask]))
        u, v = float(np.median(xs)), float(np.median(ys))
        item["centroid_px"] = [round(u, 2), round(v, 2)]
        item["median_depth_m"] = round(z, 4)
        item["approx_camera_xyz_m"] = [
            round((u - cx) * z / fx, 4),
            round((v - cy) * z / fy, 4),
            round(z, 4),
        ]
        item["mask_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def support_score(upper: dict, lower: dict) -> tuple[float, dict]:
    ux1, uy1, ux2, uy2 = upper["mask_bbox"]
    lx1, ly1, lx2, ly2 = lower["mask_bbox"]
    overlap = max(0, min(ux2, lx2) - max(ux1, lx1))
    overlap_ratio = overlap / max(1, min(ux2 - ux1, lx2 - lx1))
    vertical_gap = ly1 - uy2
    size_scale = max(1, min(uy2 - uy1, ly2 - ly1))
    depth_delta = abs(upper["median_depth_m"] - lower["median_depth_m"])
    upper_is_above = upper["centroid_px"][1] < lower["centroid_px"][1]
    # In a camera projection, a genuinely stacked object commonly overlaps the
    # supporting mask. A smooth contact likelihood therefore accepts a modest
    # negative gap instead of requiring two disjoint bounding boxes.
    contact = math.exp(-((vertical_gap / (1.25 * size_scale)) ** 2))
    depth_agreement = max(0.0, 1.0 - depth_delta / 0.45)
    score = overlap_ratio * contact * (0.55 + 0.45 * depth_agreement) if upper_is_above else 0.0
    evidence = {
        "horizontal_overlap_ratio": round(float(overlap_ratio), 3),
        "vertical_gap_px": int(vertical_gap),
        "depth_delta_m": round(float(depth_delta), 4),
        "support_score": round(float(score), 3),
    }
    return float(score), evidence


def build_scene_graph(objects: list[dict], target: str) -> dict:
    edges: list[dict] = []
    for upper in objects:
        best = None
        for lower in objects:
            if upper is lower:
                continue
            score, evidence = support_score(upper, lower)
            if best is None or score > best[0]:
                best = (score, lower, evidence)
        if best and best[0] >= 0.22:
            edges.append(
                {
                    "source": upper["object_id"],
                    "relation": "supported_by",
                    "target": best[1]["object_id"],
                    "evidence": best[2],
                }
            )

    blockers = [edge["source"] for edge in edges if edge["target"] == target]
    removal_order = blockers + ([target] if any(o["object_id"] == target for o in objects) else [])
    return {
        "nodes": [
            {
                key: value
                for key, value in obj.items()
                if key
                in {
                    "object_id",
                    "label",
                    "detector_score",
                    "mask_quality",
                    "representative_rgb",
                    "centroid_px",
                    "median_depth_m",
                    "approx_camera_xyz_m",
                    "mask_bbox",
                }
            }
            for obj in objects
        ],
        "edges": edges,
        "query": {"action": "remove", "target": target},
        "plan": {
            "blocked": bool(blockers),
            "blockers": blockers,
            "removal_order": removal_order,
            "explanation": (
                f"Move {', '.join(blockers)} first, then remove {target}."
                if blockers
                else f"No supported object was detected above {target}; remove it directly."
            ),
        },
        "limitations": [
            "Depth is monocular metric estimation, not a calibrated RGB-D measurement.",
            "XYZ uses approximate intrinsics because this image sample does not ship camera calibration.",
            "Support edges are geometric hypotheses and should be confirmed across frames before execution.",
        ],
    }


def save_detection_figure(image: Image.Image, detections: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(image)
    for idx, item in enumerate(detections, 1):
        x1, y1, x2, y2 = item["box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2, ec="#22c55e"))
        ax.text(
            x1,
            max(3, y1 - 2),
            f"P{idx} {item['detector_score']:.2f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "#166534", "alpha": 0.85, "pad": 2},
        )
    ax.set_title("Grounding DINO — learned open-vocabulary proposals\nprompts: ‘small object’ + ‘blue cube’")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_segmentation_figure(image: Image.Image, objects: list[dict], masks: np.ndarray, path: Path) -> None:
    base = np.asarray(image).astype(float) / 255.0
    overlay = base.copy()
    for item, mask in zip(objects, masks):
        color = np.asarray(mcolors.to_rgb(DISPLAY_COLORS[item["label"]]))
        overlay[mask] = 0.42 * overlay[mask] + 0.58 * color
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(overlay)
    for item in objects:
        x, y = item["centroid_px"] if "centroid_px" in item else (
            (item["box"][0] + item["box"][2]) / 2,
            (item["box"][1] + item["box"][3]) / 2,
        )
        ax.text(
            x,
            y,
            item["label"].replace("_", " "),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            bbox={"facecolor": "#111827", "alpha": 0.82, "pad": 2},
        )
    ax.set_title("SAM 2.1 — box-prompted pixel masks\ncolor identity from masked-pixel descriptor")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_depth_figure(image: Image.Image, depth: np.ndarray, objects: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    axes[0].imshow(image)
    axes[0].set_title("RGB input")
    axes[0].axis("off")
    depth_artist = axes[1].imshow(depth, cmap="magma_r")
    for item in objects:
        x, y = item["centroid_px"]
        axes[1].scatter([x], [y], c="cyan", s=26, edgecolors="black", linewidths=0.6)
        axes[1].text(x + 3, y, f"{item['label'].split('_')[0]}\n{item['median_depth_m']:.2f} m", color="cyan", fontsize=7)
    axes[1].set_title("Depth Anything V2 — indoor metric estimate")
    axes[1].axis("off")
    fig.colorbar(depth_artist, ax=axes[1], fraction=0.046, pad=0.04, label="estimated meters")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_aligned_depth_colormap(depth: np.ndarray, path: Path) -> None:
    """Save a pixel-aligned depth visualization for the interactive viewer."""
    low, high = np.percentile(depth, [2, 98])
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    rgba = plt.get_cmap("magma_r")(normalized)
    Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8)).save(path)


def save_graph_figure(image: Image.Image, graph: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].imshow(image)
    axes[0].set_title("Input scene")
    axes[0].axis("off")

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Geometric support graph + removal plan")
    nodes = graph["nodes"]
    positions: dict[str, tuple[float, float]] = {}
    ordered = sorted(nodes, key=lambda n: n["centroid_px"][1])
    for idx, node in enumerate(ordered):
        x = 0.25 if idx % 2 == 0 else 0.67
        y = 0.82 - 0.19 * idx
        positions[node["object_id"]] = (x, y)
        color = DISPLAY_COLORS.get(node["object_id"], "#64748b")
        ax.text(
            x,
            y,
            node["object_id"].replace("_", " "),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "white", "lw": 1.5},
        )
    for edge in graph["edges"]:
        start, end = positions[edge["source"]], positions[edge["target"]]
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                color="#0f172a",
                linewidth=1.8,
                connectionstyle="arc3,rad=0.12",
            )
        )
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my, "supported by", fontsize=8, ha="center", bbox={"facecolor": "white", "alpha": 0.9, "pad": 1})
    plan = graph["plan"]
    ax.text(
        0.04,
        0.05,
        "Query: remove pink block\nPlan: " + " → ".join(plan["removal_order"]) + "\n" + plan["explanation"],
        fontsize=10,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#e0f2fe", "edgecolor": "#0284c7"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_figure(
    image: Image.Image,
    after_image: Image.Image | None,
    objects: list[dict],
    masks: np.ndarray,
    depth: np.ndarray,
    graph: dict,
    path: Path,
) -> None:
    base = np.asarray(image).astype(float) / 255.0
    overlay = base.copy()
    for item, mask in zip(objects, masks):
        color = np.asarray(mcolors.to_rgb(DISPLAY_COLORS[item["label"]]))
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("1  RGB input")
    axes[1].imshow(overlay)
    axes[1].set_title("2  Grounded masks")
    axes[2].imshow(depth, cmap="magma_r")
    axes[2].set_title("3  Metric depth estimate")
    if after_image is not None:
        axes[3].imshow(after_image)
        axes[3].set_title("4  Desired result")
    else:
        axes[3].text(0.5, 0.5, " → ".join(graph["plan"]["removal_order"]), ha="center", va="center", fontsize=14)
        axes[3].set_title("4  Removal order")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        "Real hybrid perception demo: learned perception → explicit support reasoning\n"
        + graph["plan"]["explanation"],
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    image = Image.open(args.image).convert("RGB")
    after_image = Image.open(args.after_image).convert("RGB") if args.after_image.exists() else None
    started = time.perf_counter()
    timings: dict[str, float] = {}

    print(f"Input: {args.image}")
    print(f"Device: {device}")
    print("[1/4] Grounding DINO proposals")
    t0 = time.perf_counter()
    detections = run_detection(image, device)
    timings["grounding_dino_s"] = round(time.perf_counter() - t0, 3)
    if not detections:
        raise RuntimeError("Grounding DINO returned no usable proposals")
    save_detection_figure(image, detections, args.output_dir / "01_grounding_dino_boxes.png")

    print(f"[2/4] SAM 2.1 masks for {len(detections)} proposals")
    t0 = time.perf_counter()
    masks, quality = run_segmentation(image, detections, device)
    objects, object_masks = label_and_filter_objects(image, detections, masks, quality)
    timings["sam2_s"] = round(time.perf_counter() - t0, 3)
    if len(objects) < 2:
        raise RuntimeError("Fewer than two task objects survived mask/color filtering")

    print("[3/4] Depth Anything V2 metric depth")
    t0 = time.perf_counter()
    depth = run_depth(image, device)
    timings["depth_anything_v2_s"] = round(time.perf_counter() - t0, 3)
    enrich_geometry(objects, object_masks, depth)
    save_segmentation_figure(image, objects, object_masks, args.output_dir / "02_sam2_masks.png")
    save_depth_figure(image, depth, objects, args.output_dir / "03_metric_depth.png")
    save_aligned_depth_colormap(depth, args.output_dir / "depth_colormap.png")

    print("[4/4] Support graph and removal order")
    graph = build_scene_graph(objects, args.target)
    graph["models"] = {
        "detector": DETECTOR_ID,
        "segmenter": SEGMENTER_ID,
        "depth": DEPTH_ID,
    }
    graph["input"] = str(args.image.resolve())
    graph["timings_seconds"] = timings
    graph["runtime_seconds"] = round(time.perf_counter() - started, 3)
    positions = {
        "input": str(args.image.resolve()),
        "image_size_px": [image.width, image.height],
        "objects": graph["nodes"],
        "models": graph["models"],
        "timings_seconds": timings,
        "coordinate_notes": {
            "centroid_px": "Mask centroid in source-image pixel coordinates.",
            "median_depth_m": "Median monocular metric-depth estimate inside the SAM mask.",
            "approx_camera_xyz_m": "Approximate camera-frame XYZ using assumed intrinsics; replace with calibrated RGB-D for deployment.",
        },
    }
    (args.output_dir / "object_positions.json").write_text(
        json.dumps(positions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "scene_graph.json").write_text(
        json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(args.output_dir / "metric_depth_m.npy", depth)
    for item, mask in zip(objects, object_masks):
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            args.output_dir / f"mask_{item['object_id']}.png"
        )
    save_graph_figure(image, graph, args.output_dir / "04_support_graph.png")
    save_summary_figure(
        image,
        after_image,
        objects,
        object_masks,
        depth,
        graph,
        args.output_dir / "05_pipeline_summary.png",
    )

    print(json.dumps(graph["plan"], indent=2))
    print(f"Done in {graph['runtime_seconds']:.1f}s. Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
