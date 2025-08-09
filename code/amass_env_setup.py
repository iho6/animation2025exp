"""
AMASS Environment Setup Script
- This script sets up the environment for AMASS/SMPL development.
- Note: Jupyter/IPython magic commands (e.g., %autoreload, %matplotlib notebook) do not work in .py scripts.
- For interactive reloading, use importlib.reload in Python scripts.
- For matplotlib backend, set before importing pyplot.
"""


# Set matplotlib backend (optional, only works in some environments)
try:
    import matplotlib
    matplotlib.use('notebook')  # Or 'inline', depending on your environment
except ImportError:
    pass

import argparse

import torch
import numpy as np
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp


def main():
    parser = argparse.ArgumentParser(description='AMASS/SMPL Environment Setup and Visualization')
    parser.add_argument('--data', type=str, required=True, help='Path to AMASS/DMPL .npz data file')
    parser.add_argument('--body_models_dir', type=str, default='../amass/body_models', help='Path to body_models directory (default: ../amass/body_models)')
    parser.add_argument('--gender', type=str, default=None, help='Gender (neutral, male, female). If not set, will use gender from data file.')
    parser.add_argument('--num_betas', type=int, default=16, help='Number of body shape betas')
    parser.add_argument('--num_dmpls', type=int, default=8, help='Number of DMPL parameters')
    args = parser.parse_args()

    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bdata = np.load(args.data)
    subject_gender = args.gender if args.gender is not None else bdata['gender'].item() if hasattr(bdata['gender'], 'item') else bdata['gender']
    print('Data keys available:%s' % list(bdata.keys()))
    print('The subject of the mocap sequence is  {}.'.format(subject_gender))

    from human_body_prior.body_model.body_model import BodyModel
    bm_fname = osp.join(args.body_models_dir, 'smplh', str(subject_gender), 'model.npz')
    dmpl_fname = osp.join(args.body_models_dir, 'dmpls', str(subject_gender), 'model.npz')
    num_betas = args.num_betas
    num_dmpls = args.num_dmpls
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
    faces = c2c(bm.f)
    time_length = len(bdata['trans'])
    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device),
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device),
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device),
        'trans': torch.Tensor(bdata['trans']).to(comp_device),
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device),
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device)
    }
    print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    print('time_length = {}'.format(time_length))

    import trimesh
    from body_visualizer.tools.vis_tools import colors
    from body_visualizer.mesh.mesh_viewer import MeshViewer
    from body_visualizer.mesh.sphere import points_to_spheres
    from body_visualizer.tools.vis_tools import show_image

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    body_pose_beta = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas']})
    def vis_body_pose_beta(fId=0):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)
    vis_body_pose_beta(fId=0)

    body_pose_hand = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand']})
    def vis_body_pose_hand(fId=0):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)
    vis_body_pose_hand(fId=0)

    def vis_body_joints(fId=0):
        joints = c2c(body_pose_hand.Jtr[fId])
        joints_mesh = points_to_spheres(joints, point_color=colors['red'], radius=0.005)
        mv.set_static_meshes([joints_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)
    vis_body_joints(fId=0)

    body_dmpls = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls']})
    def vis_body_dmpls(fId=0):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_dmpls.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)
    vis_body_dmpls(fId=0)

    body_trans_root = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})
    def vis_body_trans_root(fId=0):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)
    vis_body_trans_root(fId=0)

    def vis_body_transformed(fId=0):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 1)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (1, 0, 0)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)
    vis_body_transformed(fId=0)

if __name__ == "__main__":
    main()

# If you want autoreload-like behavior in scripts, use importlib.reload(module)
# Example:
# import importlib
# import mymodule
# importlib.reload(mymodule)

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amass_npz_fname = osp.join(support_dir, 'github_data/dmpl_sample.npz') # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']

print('Data keys available:%s'%list(bdata.keys()))

print('The subject of the mocap sequence is  {}.'.format(subject_gender))

from human_body_prior.body_model.body_model import BodyModel

bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))

num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}

print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
print('time_length = {}'.format(time_length))


import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)


body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas']})

def vis_body_pose_beta(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_pose_beta(fId=0)

body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand']})

def vis_body_pose_hand(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_pose_hand(fId=0)

def vis_body_joints(fId = 0):
    joints = c2c(body_pose_hand.Jtr[fId])
    joints_mesh = points_to_spheres(joints, point_color = colors['red'], radius=0.005)

    mv.set_static_meshes([joints_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_joints(fId=0)

body_dmpls = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls']})

def vis_body_dmpls(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_dmpls.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_dmpls(fId=0)

body_trans_root = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls',
                                                                   'trans', 'root_orient']})

def vis_body_trans_root(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_trans_root(fId=0)

def vis_body_transformed(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (1, 0, 0)))

    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_transformed(fId=0)