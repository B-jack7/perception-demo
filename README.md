# Active Multi-View RGB-D Reconstruction

Interactive proposal and implementation workspace for TUM Robotics Praktikum Case 2.

## Live demo

**[Open the public interactive demo](https://b-jack7.github.io/perception-demo/)**

The demo follows one complete tabletop example:

1. Observe a cup, cube and bottle with the overhead Orbbec Gemini 335.
2. Move the wrist-mounted Gemini 305 to additional viewpoints.
3. Register partial clouds with robot transforms and constrained ICP.
4. Fuse the observations into a local voxel/TSDF map.
5. Remove the table and separate independent 3D object instances.
6. Fuse geometry, colour and multi-view CLIP/SigLIP evidence to recognize the objects.
7. Build a semantic 3D scene graph.
8. Resolve “move the cup on the left to the right side”, plan the action and update the changed map region.

The page supports automatic playback, direct step selection, single-view versus multi-view comparison, 3D orbit/zoom and click-to-inspect object metadata.

## Run locally on macOS

### Fastest option

Open `index.html` directly in a browser.

### Recommended option

Double-click `start_demo.command`, or run:

```bash
cd /Users/chen/Documents/Codex/2026-11-02/ni-k/perception_demo
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/>.

The demo is static and does not upload local data. The public GitHub Pages link can be shared with anyone and does not require a GitHub account.

## Current status

The current web scene is a deterministic simulation of the proposed pipeline. It demonstrates interfaces, information flow and expected intermediate outputs; it does not claim to be live Orbbec data.

Next implementation targets:

- Confirm access to Platform A and permission for pre-action wrist-camera motion.
- Record real RGB, aligned depth, CameraInfo, joint states and TF with rosbag2.
- Complete overhead and wrist hand-eye calibration.
- Replace synthetic surface samples with recorded point clouds.
- Evaluate registration, localization, recognition and spatial grounding accuracy.

## Repository layout

```text
index.html                 Public demo entry point
active_demo.css            Responsive visual design
active_demo.js             Interactive 3D pipeline simulation
interactive_viewer.html    Redirect for previously shared links
start_demo.command         macOS double-click launcher
docs/
  technical-route.md       Architecture, assumptions and research question
  implementation-roadmap.md
legacy/                    Archived Grounding DINO/SAM/monocular-depth prototype
```

## Technical documentation

- [Technical route](docs/technical-route.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Legacy prototype](legacy/README.md)

## Hardware assumption

The active multi-view route assumes Platform A:

- SO-101 leader/follower arm
- Orbbec Gemini 335 overhead RGB-D camera
- Orbbec Gemini 305 wrist-mounted RGB-D camera
- Jetson AGX Orin
- ROS 2, TF2 and optional MoveIt 2 integration

Platform allocation and access to wrist-camera scanning still require confirmation.
