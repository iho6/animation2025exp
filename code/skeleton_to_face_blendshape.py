"""
TODO: Implement conversion from MediaPipe skeleton (with face landmarks) to blendshape scores.

Goal:
- Take skeleton/landmark output (JSON) from skeleton_visualizer or MediaPipe Face Landmarker.
- Map facial keypoints to blendshape weights (e.g., for mouth open, smile, brow raise, etc.).
- Output blendshape scores per frame for use in facial animation (e.g., for 3D models in Blender, Unity, etc.).

Steps to implement:
1. Parse MediaPipe face landmarks from skeleton JSON.
2. Define mapping from landmark positions to blendshape weights.
3. Implement function to compute blendshape scores from landmarks.
4. Export results as a per-frame blendshape score file (e.g., JSON or CSV).

References:
- MediaPipe Face Landmarker documentation
- Common blendshape sets (ARKit, VRM, etc.)
- Example: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

"""

# TODO: Implement skeleton_to_face_blendshape pipeline

