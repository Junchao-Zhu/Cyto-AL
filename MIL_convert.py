import os
import numpy as np
import pandas as pd
import shutil
import random

num = [50, 100, 150, 200, 250, 300]

for n in num:
    """ MIL selection for random choice """
    file_dir = r'.\MIL_new\MIL_classification'
    save_path = r'.\MIL_new\selected_results\{}_selected\Random'.format(n)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for d in os.listdir(file_dir):
        for file in os.listdir(os.path.join(file_dir, d)):
            tmp_path = os.path.join(file_dir, d, file)
            tmp_file = np.load(tmp_path)

            random_idxs = random.sample(range(0, tmp_file.shape[0]), n)
            sorted_numbers = sorted(random_idxs)

            tmp_random = tmp_file[sorted_numbers]

            save_dir = os.path.join(save_path, '{}'.format(d))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            tmp_save = os.path.join(save_dir, file)
            np.save(tmp_save, tmp_random)

    """ MIL selection for Ours """
    score_path = r'.\MIL_new\MIL_combined_information\score'
    file_path = r'.\MIL_new\MIL_classification'
    save_path = r'.\MIL_new\selected_results\{}_selected\Ours'.format(n)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for d in os.listdir(file_path):
        for file in os.listdir(os.path.join(file_path, d)):
            tmp_path = os.path.join(file_path, d, file)
            tmp_file = np.load(tmp_path)

            tmp_score = np.load(os.path.join(score_path, 'wsi_score_{}'.format(file)))
            indices = np.argsort(tmp_score)[-n:]
            largest_elements = np.sort(tmp_score)[-n:]
            select_file = tmp_file[indices]

            save_dir = os.path.join(save_path, '{}'.format(d))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_name = os.path.join(save_dir, file)
            np.save(save_name, select_file)
