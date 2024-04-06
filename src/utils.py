import numpy as np
import tqdm
import torch

import torch.nn as nn
import torch.nn.functional as F
import pymeshlab
import os
import glob

from constants import PAD_ID

def remove_pad(tensor, pad_id=PAD_ID):
    non_pad_idx = torch.nonzero(tensor != pad_id)[:, 0].unique()
    tensor = tensor[non_pad_idx]
    return tensor

def save_to_obj(vertices_tensor, faces_tensor, save_path='test.obj'):
    v = remove_pad(vertices_tensor).cpu().numpy()
    f = np.array(remove_pad(faces_tensor).cpu(), dtype=np.int32)
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=v, face_matrix=f)
    ms.add_mesh(mesh)
    ms.save_current_mesh(save_path)