import os
import numpy as np


root = r'.\MIL_new'

density_path = os.path.join(root, 'density')
embedding_path = os.path.join(root, 'embedding')
save_path = os.path.join(root, 'combined')

for file in os.listdir(density_path):
    tmp_name = file.split('_')[-1]
    tmp_embedding = np.load(os.path.join(embedding_path, 'embedding_'+tmp_name))
    tmp_density = np.expand_dims(np.load(os.path.join(density_path, file)), axis=1)

    tmp = np.concatenate((tmp_embedding, tmp_density), axis=1)

    np.save(os.path.join(save_path, tmp_name), tmp)