# Active Multi-View RGB-D Reconstruction Demo

This repository presents an independent technical route for Case 2: active multi-view RGB-D reconstruction followed by 3D scene-graph grounding.

The meeting demo visualizes a complete tabletop example containing a cup, cube and bottle:

1. Capture an initial overhead RGB-D view with Orbbec Gemini 335.
2. Move the wrist-mounted Gemini 305 to additional safe viewpoints.
3. Transform each partial cloud into the robot-base frame using forward kinematics and hand-eye calibration.
4. Refine alignment with constrained ICP and fuse the observations with a voxel grid or TSDF.
5. Remove the table and cluster the cloud into independent object instances.
6. Fuse 3D shape evidence, multi-view RGB crops and CLIP/SigLIP scores to recognize the cup, cube and bottle.
7. Build a semantic 3D scene graph and convert “move the cup on the left to the right side” into a structured graph query.
8. Select the cup, plan the motion, execute it and refresh the changed part of the map.

## Open the demo

Open `index.html` directly, or serve the directory locally:

```bash
python3 -m http.server 8765
```

Then visit <http://127.0.0.1:8765/>.

The old `interactive_viewer.html` URL redirects to the new demo so previously shared links remain usable.

## Interactions

- Play the full route automatically or select an individual step.
- Compare active multi-view reconstruction with a single overhead view.
- Drag to orbit the reconstructed scene and scroll to zoom.
- Click a reconstructed object to inspect its position and dimensions.
- Watch the camera viewpoint, visible surface coverage, scene graph and planned robot path change across the route.

## Scope and honesty

The current web demo uses a deterministic synthetic tabletop scene. It explains the proposed architecture and does not claim to be live Orbbec output. The next implementation steps are real ROS 2 camera topics, hand-eye calibration, safe SO-101 viewpoints, point-cloud registration and evaluation on recorded rosbags.

This route assumes access to Platform A:

- Orbbec Gemini 335 overhead RGB-D camera
- Orbbec Gemini 305 wrist-mounted RGB-D camera
- SO-101 arm
- Jetson AGX Orin

Access to Platform A and permission to move the wrist camera before manipulation must still be confirmed.

## Legacy prototype

The previous Grounding DINO + SAM 2 + monocular-depth scripts and generated assets remain in the repository only as legacy material. They are not part of the current proposed route and are no longer linked from the meeting demo.
