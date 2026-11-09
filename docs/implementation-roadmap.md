# Implementation Roadmap

## Phase 1 — Offline geometry

- Load recorded RGB, depth and camera intrinsics.
- Back-project depth into a metric point cloud.
- Crop the robot workspace and remove the table with RANSAC.
- Cluster tabletop objects and visualize their coordinates.

## Phase 2 — Real Orbbec input

- Connect Gemini 335 RGB, aligned depth and CameraInfo topics.
- Record repeatable rosbags with timestamps, `/joint_states` and TF.
- Quantify depth validity and temporal noise.

## Phase 3 — Multi-view registration

- Calibrate the fixed overhead camera.
- Calibrate `T_end_effector_camera` for Gemini 305.
- Transform all views into `robot_base` using forward kinematics.
- Add constrained ICP and diagnose double edges or ghosting.

## Phase 4 — Reconstruction and instances

- Fuse observations with a voxel grid, then TSDF if needed.
- Preserve occupied, free and unknown regions separately.
- Add 3D region growing and contact-object separation.

## Phase 5 — Semantic recognition

- Project each 3D instance into all available RGB views.
- Extract colour and primitive-shape evidence.
- Compare multi-view crops with CLIP/SigLIP class prompts.
- Fuse evidence and expose confidence instead of forcing a label.

## Phase 6 — Scene graph and manipulation

- Compute left/right/front/behind/on-top relations in a declared reference frame.
- Parse language into a structured scene-graph query.
- Generate target and destination poses.
- Integrate MoveIt 2 or safe recorded waypoints.
- Reobserve after execution and update only the changed map region.

## Immediate decisions required

1. Is Group 2 assigned to Platform A?
2. May the wrist Gemini 305 move before manipulation?
3. Is hand-eye calibration already available?
4. Is an SO-101 MoveIt configuration provided?
5. Which frame defines human words such as “left” and “right”?

