"""
visualize_mesh.py

Visualize mesh .npy files (from Pose2Mesh or similar) as .png images and optionally chain them into a video.

Usage:
    python visualize_mesh.py --input_dir pose2mesh_input/ --output_dir data/output/mesh_frames/ --make_video

Dependencies:
    - numpy
    - trimesh
    - matplotlib
    - imageio
    - opencv-python (for video)

"""
import os
import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import imageio
import cv2

def visualize_mesh_npy(npy_path, output_path, faces=None, elev=30, azim=45):
    data = np.load(npy_path, allow_pickle=True)
    if data.dtype == object:
        data = data.item()
    if isinstance(data, dict):
        verts = data.get('verts')
        if verts is None:
            verts = data.get('vertices')
        if verts is None:
            verts = data.get('v')
        faces_from_data = data.get('faces')
        if faces_from_data is not None:
            faces = faces_from_data
    else:
        verts = data
    if verts is None or faces is None:
        print(f"DEBUG: verts type: {type(verts)}, shape: {getattr(verts, 'shape', None)}")
        print(f"DEBUG: faces type: {type(faces)}, shape: {getattr(faces, 'shape', None)}")
        raise ValueError(f"Missing verts or faces in {npy_path}")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color='lightgrey', edgecolor='none', alpha=1.0)
    ax.view_init(elev, azim)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Visualize mesh .npy files as .png images and optionally make a video.')
    parser.add_argument('--input_dir', required=True, help='Directory with .npy mesh files')
    parser.add_argument('--output_dir', required=True, help='Directory to save .png images')
    parser.add_argument('--faces', default=None, help='Optional .npy file with faces array')
    parser.add_argument('--make_video', action='store_true', help='If set, chain images into a video')
    parser.add_argument('--video_name', default='mesh_video.mp4', help='Output video filename')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    faces = np.load(args.faces, allow_pickle=True) if args.faces else None
    npy_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.npy')])
    img_paths = []
    for fname in npy_files:
        npy_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname.replace('.npy', '.png'))
        try:
            visualize_mesh_npy(npy_path, out_path, faces=faces)
            img_paths.append(out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Failed to visualize {npy_path}: {e}")
    if args.make_video and img_paths:
        images = [imageio.imread(p) for p in img_paths]
        height, width, _ = images[0].shape
        video_path = os.path.join(args.output_dir, args.video_name)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))
        for img in images:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved to {video_path}")

if __name__ == '__main__':
    main()
