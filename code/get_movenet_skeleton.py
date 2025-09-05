
import tensorflow as tf
import tensorflow_hub as hub
import cv2, json, argparse, os

def parse_args():
    parser = argparse.ArgumentParser(description="Extract MoveNet skeletons from video.")
    parser.add_argument('--in', dest='input_path', required=True, help='Input video path')
    parser.add_argument('--out', dest='output_path', required=True, help='Output JSON path')
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # Load MoveNet model
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    cap = cv2.VideoCapture(input_path)
    frame_id = 0
    skeleton_data = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
        img = tf.cast(img, dtype=tf.int32)

        outputs = movenet.signatures['serving_default'](img)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :].tolist()

        # Check if detection score is high enough (score is at index 2 of each keypoint)
        # If all scores are very low, skip this frame
        scores = [kp[2] for kp in keypoints]
        if max(scores) < 0.2:
            frame_id += 1
            continue

        skeleton_data[frame_id] = keypoints
        frame_id += 1

    cap.release()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(skeleton_data, f)

    print(f"Saved skeletons to {output_path}")

if __name__ == "__main__":
    main()
