import numpy as np
import os

import torch
import pyrenderer

"""
Format of .npz files:
density_%06d.npz -> field 'data' of shape (Z,Y,X,C=1)
velocity_%06d.npz -> field 'data' of shape (Z,Y,X,C=3)
"""

def convert(in_file_density: str, in_file_velocity:str, out_file: str):
    print("convert", in_file_density, "and", in_file_velocity, "to", out_file)
    data_file = np.load(in_file_density)
    data = data_file['data']
    density = data[...].astype(np.float32)
    density = np.moveaxis(density, (0,1,2,3), (3,2,1,0))
    print(data.shape, data.dtype, np.min(data), np.max(data), np.mean(data))
    MAX_DENSITY = 5
    density = np.clip(density, 0, MAX_DENSITY)/MAX_DENSITY

    data_file = np.load(in_file_velocity)
    data = data_file['data']
    velocity = data[...].astype(np.float32)
    velocity = np.moveaxis(velocity, (0,1,2,3), (3,2,1,0))
    
    assert density.shape[0] == 1
    assert velocity.shape[0] == 3
    assert density.shape[1:] == velocity.shape[1:]
    assert len(density.shape) == 4

    #make cubic
    max_dim = max(density.shape[1], density.shape[2], density.shape[3])
    if density.shape[1] < max_dim:
        a = (max_dim-density.shape[1]) // 2
        b = max_dim - density.shape[1] - a
        density = np.concatenate((
            np.zeros((1, a, density.shape[2], density.shape[3]), dtype=np.float32),
            density,
            np.zeros((1, b, density.shape[2], density.shape[3]), dtype=np.float32),
            ), axis=1)
        velocity = np.concatenate((
            np.zeros((3, a, density.shape[2], density.shape[3]), dtype=np.float32),
            velocity,
            np.zeros((3, b, density.shape[2], density.shape[3]), dtype=np.float32),
            ), axis=1)
    if density.shape[2] < max_dim:
        a = (max_dim-density.shape[2]) // 2
        b = max_dim - density.shape[2] - a
        density = np.concatenate((
            np.zeros((1, density.shape[1], a, density.shape[3]), dtype=np.float32),
            density,
            np.zeros((1, density.shape[1], b, density.shape[3]), dtype=np.float32),
            ), axis=2)
        velocity = np.concatenate((
            np.zeros((3, density.shape[1], a, density.shape[3]), dtype=np.float32),
            velocity,
            np.zeros((3, density.shape[1], b, density.shape[3]), dtype=np.float32),
            ), axis=2)
    if density.shape[3] < max_dim:
        a = (max_dim-density.shape[3]) // 2
        b = max_dim - density.shape[3] - a
        density = np.concatenate((
            np.zeros((1, density.shape[1], density.shape[2], a), dtype=np.float32),
            density,
            np.zeros((1, density.shape[1], density.shape[2], b), dtype=np.float32),
            ), axis=3)
        velocity = np.concatenate((
            np.zeros((3, density.shape[1], density.shape[2], a), dtype=np.float32),
            velocity,
            np.zeros((3, density.shape[1], density.shape[2], b), dtype=np.float32),
            ), axis=3)
    print("new shape:", density.shape)

    # join them
    full_data = np.concatenate((velocity, density), axis=0)
    
    vol = pyrenderer.Volume()
    vol.worldX = 1
    vol.worldY = 1
    vol.worldZ = 1
    vol.add_feature_from_tensor("vel+density", torch.from_numpy(full_data))
    vol.save(out_file, compression=0)

def convert_all_densities(folder: str):
    i = 1
    while True:
        in_file_density = os.path.join(folder, "density_%06d.npz"%i)
        in_file_velocity = os.path.join(folder, "velocity_%06d.npz"%i)
        if not os.path.exists(in_file_density):
            break
        out_file = os.path.join(folder, "volume_%06d.cvol"%i)
        convert(in_file_density, in_file_velocity, out_file)
        i += 1

if __name__ == '__main__':
    #convert_all_densities("sim_000000")
    #convert_all_densities("sim_000001")
    #convert_all_densities("sim_000002")
    #convert_all_densities("sim_000003")
    #convert_all_densities("sim_000004")
    #convert_all_densities("sim_000005")
    convert_all_densities("sim_000006")
    convert_all_densities("sim_000007")
    convert_all_densities("sim_000008")
    convert_all_densities("sim_000009")
    convert_all_densities("sim_000010")
    print("Done")
