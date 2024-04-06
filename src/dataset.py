import numpy as np
import tqdm
import torch

import torch.nn as nn
import torch.nn.functional as F
import pymeshlab
import os
import glob

from constants import PAD_ID

def _sort_vertices_by_zyx(vertices: np.array):
    vertices = np.array([tuple(x) for x in vertices], dtype = [('x', float), ('z', float), ('y', float)])
    new_idx = np.argsort(vertices, axis=0, order=['z', 'y', 'x'])
    old2new_vert_idx = dict(zip(new_idx, np.arange(vertices.shape[0])))
    vertices = np.sort(vertices, axis=0, order=['z','y','x'])
    vertices = np.array([list(x) for x in vertices])
    return old2new_vert_idx, vertices

def _update_face_vert_idx(faces: np.array, old2new_vert_idx: dict):
    for i_face in range(faces.shape[0]):
        for i_vert in range(3):
            faces[i_face][i_vert] = old2new_vert_idx[faces[i_face][i_vert]]
    return faces

def _sort_faces_by_lowest_vert_idx(faces):
    faces = np.sort(faces, axis=1)
    faces = np.array(
        [tuple(x) for x in faces],
        dtype= [('lowest', int), ('2nd_lowest', int), ('3rd_lowest', int)]
    )
    faces = np.sort(faces, axis=0, order = ['lowest', '2nd_lowest', '3rd_lowest'])
    faces = np.array([list(x) for x in faces])
    return faces

def sort_vert_and_faces(faces, vertices):
    old2new_vert_idx, vertices = _sort_vertices_by_zyx(vertices)
    faces = _update_face_vert_idx(faces, old2new_vert_idx=old2new_vert_idx)
    faces = _sort_faces_by_lowest_vert_idx(faces)
    return faces, vertices

def load_mesh(path= "../dolphin.obj"):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    vertices = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix()
    faces, vertices = sort_vert_and_faces(faces=faces, vertices=vertices)
    return torch.tensor(vertices), torch.tensor(faces)

def load_all_mesh(data_dir='../bench'):
    obj_files = glob.glob(os.path.join(data_dir, '/**/*.obj'), recursive=True)
    vertices, faces = [], []
    for obj_file in tqdm.tqdm(obj_files):
        obj_vertices, obj_faces = load_mesh(obj_file)
        vertices.append(obj_vertices)
        faces.append(obj_faces)
    return vertices, faces

def pad_faces_list(faces_list, vertices_list, pad_id=PAD_ID, percentile=80):
    face_lengths = [tensor.size(0) for tensor in faces_list]
    print([(p, int(np.percentile(face_lengths, p))) for p in [50, 75, 95]])

    max_face_length = int(np.percentile(face_lengths, percentile))
    valid_idx = [i for i, face_length in enumerate(face_lengths) if face_length <= max_face_length]
    max_vertice_length = max(vertices_list[i].size(0) for i in valid_idx)

    print(f'max_face_length = {max_face_length} | max_vert_length = {max_vertice_length}| {pad_id}')
    new_faces_list, new_vertices_list = [], []
    for i in valid_idx:
        new_faces_list.append(F.pad(faces_list[i], (0, 0, 0, (max_face_length-faces_list[i].size(0))), value=pad_id))
        new_vertices_list.append(F.pad(vertices_list[i], (0, 0, 0, (max_vertice_length-vertices_list[i].size(0))), value=pad_id))
        
    print(len(new_faces_list), len(faces_list))
    return new_faces_list, new_vertices_list

class MeshDataset(torch.utils.data.Dataset):
    def __init__(self, faces, vertices, device='cuda'):
        self.faces = faces
        self.vertices = vertices
        self.device = device
    
    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        return self.faces[idx].to(self.device), self.vertices[idx].to(self.device)

# def build_and_save_mesh_dataset():
if __name__ == '__main__':
    vertices, faces = load_all_mesh(data_dir='../../bench/')
    faces, vertices = pad_faces_list(faces, vertices, percentile=50)
    vertices = torch.stack(vertices)
    faces = torch.stack(faces).to(torch.int64)
    print(vertices.shape, faces.shape)
    mesh_dataset = MeshDataset(faces=faces, vertices=vertices)
    torch.save(mesh_dataset, 'mesh_dataset.pt')