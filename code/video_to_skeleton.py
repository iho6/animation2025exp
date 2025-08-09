import cv2
import mediapipe as mp
import json
import argparse
from pathlib import Path
from tqdm import tqdm

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def video_to_skeleton(input_video, output_json, output_annotated=None,
                      model_complexity=2, min_det_conf=0.5, min_track_conf=0.5):
    import os
    ext = os.path.splitext(str(input_video))[1].lower()
    is_image = ext in ['.png', '.jpg', '.jpeg']
    results_list = []
    if is_image:
        # Single image mode
        frame = cv2.imread(str(input_video))
        if frame is None:
            raise RuntimeError(f"Cannot open image: {input_video}")
        height, width = frame.shape[:2]
        with mp_pose.Pose(static_image_mode=True,
                          model_complexity=model_complexity,
                          min_detection_confidence=min_det_conf,
                          min_tracking_confidence=min_track_conf) as pose:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            frame_data = {"frame_index": 0, "pose": None}
            if res.pose_landmarks:
                kps = []
                for lm in res.pose_landmarks.landmark:
                    kps.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })
                frame_data["pose"] = kps
            results_list.append(frame_data)
            # draw overlay
            if output_annotated:
                annotated = frame.copy()
                if res.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        res.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # Save as image if output_annotated ends with image ext, else as video
                out_ext = os.path.splitext(str(output_annotated))[1].lower()
                if out_ext in ['.png', '.jpg', '.jpeg']:
                    cv2.imwrite(str(output_annotated), annotated)
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(output_annotated), fourcc, 1, (width, height))
                    writer.write(annotated)
                    writer.release()
    else:
        # Video mode (original logic)
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_annotated:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_annotated), fourcc, fps, (width, height))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with mp_pose.Pose(static_image_mode=False,
                          model_complexity=model_complexity,
                          min_detection_confidence=min_det_conf,
                          min_tracking_confidence=min_track_conf) as pose:
            for fid in tqdm(range(total_frames), desc="Frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                frame_data = {"frame_index": fid, "pose": None}
                if res.pose_landmarks:
                    kps = []
                    for lm in res.pose_landmarks.landmark:
                        kps.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility
                        })
                    frame_data["pose"] = kps
                results_list.append(frame_data)
                # draw overlay
                if writer is not None:
                    annotated = frame.copy()
                    if res.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated,
                            res.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    writer.write(annotated)
        cap.release()
        if writer:
            writer.release()

    # Save JSON
    outp = Path(output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump({"video": str(input_video), "frames": results_list}, f)

    print(f"Saved {len(results_list)} frames to {output_json}")
    if output_annotated:
        print(f"Annotated output saved to {output_annotated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video -> 2D skeleton (MediaPipe Pose)")
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("--out_json", type=str, default=None, help="Path to output JSON (default: same as video, .json)")
    parser.add_argument("--out_vid", type=str, default=None, help="Annotated output video path (default: same as video, .mp4)")
    args = parser.parse_args()

    # Determine output base path
    input_path = Path(args.input_video)
    base = input_path.with_suffix("")
    # Handle/correct extensions and defaults
    out_json = args.out_json
    out_vid = args.out_vid
    if not out_json and not out_vid:
        out_json = str(base) + ".json"
        out_vid = str(base) + "_skeleton.mp4"
    elif out_json and not out_vid:
        # Correct extension if needed
        out_json_path = Path(out_json)
        if out_json_path.suffix != ".json":
            out_json = str(out_json_path.with_suffix(".json"))
        out_vid = str(Path(out_json).with_suffix("_skeleton.mp4"))
    elif out_vid and not out_json:
        out_vid_path = Path(out_vid)
        if out_vid_path.suffix != ".mp4":
            out_vid = str(out_vid_path.with_suffix(".mp4"))
        out_json = str(Path(out_vid).with_suffix(".json"))
    else:
        # Both provided, correct extensions if needed
        out_json_path = Path(out_json)
        if out_json_path.suffix != ".json":
            out_json = str(out_json_path.with_suffix(".json"))
        out_vid_path = Path(out_vid)
        if out_vid_path.suffix != ".mp4":
            out_vid = str(out_vid_path.with_suffix(".mp4"))

    video_to_skeleton(args.input_video, out_json, out_vid)
