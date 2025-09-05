import sys
import argparse
import os
import json
import numpy as np

# Add Pose2Mesh lib and smplpytorch to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/pose2mesh/lib')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/pose2mesh/demo')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/pose2mesh/smplpytorch')))

# Add Pose2Mesh lib to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/pose2mesh/lib')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/pose2mesh/demo')))

import torch
from core.config import cfg
import models.pose2mesh_net as pose2mesh_net
from smpl import SMPL
from funcs_utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Convert skeleton JSON to mesh npy files (with verts and faces) using Pose2Mesh.")
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id (-1 for CPU)')
    parser.add_argument('--in', dest='input_json', required=True, help='Input skeleton JSON path')
    parser.add_argument('--out', dest='img_dir', required=True, help='Output PNG/MP4 directory')
    args = parser.parse_args()

    input_json = args.input_json
    img_dir = args.img_dir
    output_dir = os.path.join('data', 'input', 'mesh', 'full_mesh')
    npy_dir = os.path.join('pose2mesh_input')
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Load skeleton.json
    with open(input_json) as f:
        data = json.load(f)

    # Save intermediate keypoints as npy
    frame_keys = sorted(data.keys(), key=lambda x: int(x))
    # Use only COCO 17 joints (no pelvis/neck for Human36M weights)
    for frame_id in frame_keys:
        keypoints = data[frame_id]
        coco_points = [[kp[1], kp[0], kp[2]] for kp in keypoints]
        if len(coco_points) == 17:
            np.save(os.path.join(npy_dir, f"{frame_id}.npy"), np.array(coco_points))
        else:
            print(f"Frame {frame_id} has {len(coco_points)} keypoints, expected 17. Skipping.")

    print(f"Converted {input_json} → {npy_dir}/*.npy (COCO17)")

    # Prepare Pose2Mesh model for Human36M (official demo settings)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    smpl_model = SMPL()
    faces = smpl_model.face
    print(f"[DEBUG] smpl_model.face shape: {faces.shape}")
    if hasattr(smpl_model, 'layer') and 'neutral' in smpl_model.layer:
        verts = smpl_model.layer['neutral'].th_v_template.numpy()
        print(f"[DEBUG] smpl_model.layer['neutral'].th_v_template shape: {verts.shape}")
    joint_regressor = smpl_model.joint_regressor_h36m
    joint_num = 17  # Human36M
    from graph_utils import build_coarse_graphs
    skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
    )
    flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
    graph_Adj, graph_L, graph_perm, graph_perm_reverse = build_coarse_graphs(faces, joint_num, skeleton, flip_pairs, levels=9)
    model = pose2mesh_net.get_model(joint_num, graph_L)
    model = model.to(device)
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/pose2mesh/weights/pose2mesh_human36J_gt_train_human36/final.pth/final.pth'))
    checkpoint = load_checkpoint(load_dir=weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run inference and save only full_mesh files as 0.npy, 1.npy, ...
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')], key=lambda x: int(x.replace('.npy','')))
    # img_dir is now set by argument
    import matplotlib.pyplot as plt
    import cv2
    images = []
    for idx, fname in enumerate(npy_files):
        keypoints_path = os.path.join(npy_dir, fname)
        keypoints = np.load(keypoints_path)
        # Only use the first 17 joints for Human36M/COCO17
        if keypoints.shape[0] < joint_num:
            print(f"Warning: {fname} has {keypoints.shape[0]} joints, expected at least {joint_num}. Skipping.")
            continue
        keypoints_coco17 = keypoints[:joint_num, :]
        input_2d = torch.from_numpy(keypoints_coco17[:, :2]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            verts_pred, _ = model(input_2d)
            verts = verts_pred[0].cpu().numpy()
        # Save full mesh as idx.npy
        full_mesh = {'verts': verts, 'faces': faces}
        full_mesh_path = os.path.join(output_dir, f"{idx}.npy")
        np.save(full_mesh_path, full_mesh)
        print(f"Saved: {full_mesh_path}")
        # Visualization: simple scatter plot for now
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(verts[:,0], verts[:,1], verts[:,2], s=0.1)
        ax.set_axis_off()
        plt.tight_layout()
        img_path = os.path.join(img_dir, f"{idx}.png")
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        images.append(cv2.imread(img_path))

    # Chain PNGs into MP4
    if images:
        height, width, _ = images[0].shape
        video_path = os.path.join(img_dir, 'mesh_sequence.mp4')
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
        for img in images:
            out.write(img)
        out.release()
        print(f"MP4 video saved to {video_path}")

if __name__ == "__main__":
    main()