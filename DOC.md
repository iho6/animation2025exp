# Project Documentation

---

## Run Code

### Extract Dominant Camera View from Multi-Shot Video

```bash
python code/smooth_scene_extractor.py path/input.mp4 path/output.mp4
```

* **input.mp4**: Input multi-shot video file.
* **output.mp4**: Output video file containing only the dominant consistent shots.

---

### Extract Skeleton from Video

```bash
python code/video_to_skeleton.py path/input_video.mp4 --out_json path/input.json --out_vid path/output.mp4
```

* **my\_clip.mp4**: Input video file path.
* **--out\_json**: Path to save extracted skeleton data in JSON format.
* **--out\_vid**: Path to save annotated output video with skeleton overlay.

### Visualize Skeleton JSON (2D & 3D)

Generate 2D and 3D skeleton animation videos from a skeleton JSON file:

```bash
python code/skeleton_visualizer.py data/output/skeleton.json
```

This will create:
- `skeleton2d.mp4` — 2D skeleton animation
- `skeleton3d.mp4` — 3D skeleton animation


## Misc Utils

### Preview a Video File


To generate and open an HTML preview for any video file, run:

```bash
python code/play_video.py path/input_video.mp4
```

To open the HTML preview again (without regenerating), use:

```bash
python code/play_video.py path/input_video.mp4 --open-html
```

## Installations

### Install Python Dependencies

To install or update all required Python libraries, run:

```bash
pip install -U -r requirements.txt
```

#### Python Package List

- **scenedetect**: Scene detection in videos.
- **opencv-python / opencv-python-headless**: Computer vision and video I/O.
- **torch**: PyTorch, deep learning framework.
- **torchvision**: Image models and transforms for PyTorch.
- **scikit-learn**: Machine learning algorithms (e.g., clustering).
- **numpy**: Numerical computing and array operations.
- **mediapipe**: Cross-platform ML solutions for computer vision (e.g., pose/skeleton extraction).

### Install Requirements for Misc Utils

- **xdg-utils** (Linux only): Provides `xdg-open` for opening files with the default application. Install with:
	```bash
	sudo apt-get update && sudo apt-get install -y xdg-utils
	```
