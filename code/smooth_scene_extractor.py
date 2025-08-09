import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from sklearn.cluster import KMeans

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

import sys
import os

def detect_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    return scene_list  # list of (start, end) frame tuples

def extract_features_from_frame(frame, model, transform, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img).cpu().numpy().flatten()
    return features

def extract_shot_features(video_path, scenes, model, transform, device, sample_frames=5):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for (start, end) in scenes:
        start = int(start)
        end = int(end)
        start = max(0, start)
        end = min(end, frame_count-1)
        # sample frames evenly in the shot
        sample_points = np.linspace(start, end, num=sample_frames, dtype=int)
        shot_feats = []
        for fno in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if not ret:
                continue
            feat = extract_features_from_frame(frame, model, transform, device)
            shot_feats.append(feat)
        if len(shot_feats) > 0:
            avg_feat = np.mean(shot_feats, axis=0)
        else:
            avg_feat = np.zeros(512)  # fallback vector size for resnet18 last layer (will fix below)
        features.append(avg_feat)
    cap.release()
    return np.array(features)

def main(input_video, output_video):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained resnet18 and chop off final classifier layer, use avgpool output (512 dim)
    resnet18 = models.resnet18(pretrained=True)
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])  # remove last fc layer
    resnet18.eval().to(device)

    # Transform for input images
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    print("Detecting scenes...")
    scenes = detect_scenes(input_video)
    if len(scenes) == 0:
        print("No scenes detected! Exiting.")
        return

    print(f"Detected {len(scenes)} scenes. Extracting features...")
    shot_features = extract_shot_features(input_video, scenes, resnet18, transform, device)

    print("Clustering shots...")
    n_clusters = min(3, len(scenes))  # pick up to 3 clusters or number of shots
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(shot_features)
    labels = kmeans.labels_

    # Calculate total duration per cluster
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    durations = np.array([ (int(end) - int(start) + 1) / fps for (start, end) in scenes ])
    cluster_durations = []
    for c in range(n_clusters):
        cluster_durations.append(durations[labels == c].sum())

    dominant_cluster = int(np.argmax(cluster_durations))
    print(f"Dominant cluster: {dominant_cluster} (duration {cluster_durations[dominant_cluster]:.2f}s)")

    # Now write output video with only shots from dominant cluster
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for idx, (start, end) in enumerate(scenes):
        if labels[idx] != dominant_cluster:
            continue
        start = int(start)
        end = int(end)
        # Seek to the start of the shot once
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for fno in range(start, end+1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

    cap.release()
    out.release()
    print(f"Output video saved as {output_video}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dominant_view_extractor.py input_video.mp4 output_video.mp4")
    else:
        main(sys.argv[1], sys.argv[2])
