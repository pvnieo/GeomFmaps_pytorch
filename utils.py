# stdlib
from pathlib import Path
# 3p
import numpy as np
import torch


def read_off(file):
    file = open(file, "r")
    if file.readline().strip() != "OFF":
        raise "Not a valid OFF header"

    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(" ")])
    verts = [[float(s) for s in file.readline().strip().split(" ")] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(" ")][1:] for i_face in range(n_faces)]

    return np.array(verts), np.array(faces)


def write_off(file, verts, faces):
    file = open(file, "w")
    file.write("OFF\n")
    file.write(f"{verts.shape[0]} {faces.shape[0]} {0}\n")
    for x in verts:
        file.write(f"{' '.join(map(str, x))}\n")
    for x in faces:
        file.write(f"{len(x)} {' '.join(map(str, x))}\n")


def read_mesh(file):
    file = Path(file)
    if file.suffix == ".off":
        return read_off(file)
    else:
        raise "File extention not implemented yet!"


def write_mesh(file, verts, faces):
    file = Path(file)
    if file.suffix == ".off":
        write_off(file, verts, faces)
    else:
        raise "File extention not implemented yet!"


def frobenius_loss(a, b):
    """Compute the Frobenius loss between a and b."""

    loss = torch.sum((a - b) ** 2, axis=(1, 2))
    return torch.mean(loss)
