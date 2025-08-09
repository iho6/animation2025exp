import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import os

# Define skeleton connections (MediaPipe 33-point, simplified for demo)
# You may want to adjust this for your use case
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Right arm
    (0, 5), (5, 6), (6, 7), (7, 8),      # Left arm
    (9, 10),                            # Shoulders
    (11, 12), (12, 14), (14, 16),       # Right leg
    (11, 13), (13, 15),                 # Left leg
]


def draw_2d_skeleton(img, keypoints, connections, color=(0,255,0)):
    for i, pt in enumerate(keypoints):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 3, color, -1)
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = tuple(map(int, keypoints[i][:2]))
            pt2 = tuple(map(int, keypoints[j][:2]))
            cv2.line(img, pt1, pt2, color, 2)
    return img


def draw_3d_skeleton(ax, keypoints, connections, color='g'):
    xs, ys, zs = zip(*keypoints)
    ax.scatter(xs, ys, zs, c=color)
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            x = [keypoints[i][0], keypoints[j][0]]
            y = [keypoints[i][1], keypoints[j][1]]
            z = [keypoints[i][2], keypoints[j][2]]
            ax.plot(x, y, z, c=color)


def main(json_path, out_2d='skeleton2d.mp4', out_3d='skeleton3d.mp4', img_size=(640,480)):
    with open(json_path) as f:
        data = json.load(f)
    frames = data['frames']
    # 2D video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out2d = cv2.VideoWriter(out_2d, fourcc, 20, img_size)
    # 3D video (matplotlib animation)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ims = []
    for frame in frames:
        pose = frame['pose']
        # 2D
        keypoints2d = [(pt['x']*img_size[0], pt['y']*img_size[1]) for pt in pose]
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8)*255
        img = draw_2d_skeleton(img, keypoints2d, POSE_CONNECTIONS)
        out2d.write(img)
        # 3D
        keypoints3d = [(pt['x'], pt['y'], pt['z']) for pt in pose]
        ax.cla()
        draw_3d_skeleton(ax, keypoints3d, POSE_CONNECTIONS)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-1, 1)
        ax.set_axis_off()
        plt.tight_layout()
        # Capture the plot as an image
        fig.canvas.draw()
        im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ims.append(im)
    out2d.release()
    # Save 3D video
    h, w, _ = ims[0].shape
    out3d = cv2.VideoWriter(out_3d, fourcc, 20, (w, h))
    for im in ims:
        out3d.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    out3d.release()
    print(f"2D skeleton video saved as {out_2d}")
    print(f"3D skeleton video saved as {out_3d}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python skeleton_visualizer.py <skeleton.json>")
    else:
        main(sys.argv[1])
