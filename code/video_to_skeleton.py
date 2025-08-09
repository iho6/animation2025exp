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

    results_list = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=model_complexity,
                      min_detection_confidence=min_det_conf,
                      min_tracking_confidence=min_track_conf) as pose:

        for fid in tqdm(range(total_frames), desc="Frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR -> RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            frame_data = {"frame_index": fid, "pose": None}
            if res.pose_landmarks:
                # landmarks: x,y are normalized [0..1], z is relative, visibility is 0..1
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
        print(f"Annotated video saved to {output_annotated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video -> 2D skeleton (MediaPipe Pose)")
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("--out_json", type=str, default="skeletons.json", help="Path to output JSON")
    parser.add_argument("--out_vid", type=str, default=None, help="(Optional) annotated output video path (mp4)")
    args = parser.parse_args()

    video_to_skeleton(args.input_video, args.out_json, args.out_vid)
