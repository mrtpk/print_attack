from glob import glob
import os
import cv2
import pathlib
import numpy as np
from embedding import FaceEmbedder
import pickle

def pickle_write(path, _list):
    with open(path, 'wb') as fp:
        pickle.dump(_list, fp)

def convert(embedder, _path):
    for path in glob(_path):
        if os.path.isdir(path):
            convert(embedder, path + "/*")
        for img_path in glob(path + "/*.jpg"):
            print(img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feature = embedder.get_embedding(img)
            res_path = img_path.replace("data/", "data/res/")
            pathlib.Path("/".join(res_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
            res_path = res_path.replace(".jpg", ".npy")
            np.save(res_path, feature)

def generate_dir_meta(root_dirs, ext="jpg"):
    """
    Generate inputs and label meta from folder
    """
    dict_label = {}

    for root_dir in root_dirs:
        dirs = list(glob(root_dir))
        # print(dirs)
        tmp = []
        for _dir in dirs:
            _dir_content = list(glob(_dir + "/*"))
            _dir_name = _dir.split("/")[-1]
            # print(_dir, _dir_content)   
            for i, _label_dir in enumerate(_dir_content):
                # print(i, _label_dir)
                _label_dir_name = _label_dir.split("/")[-1]
                _label_index = i
                # print(_label_dir_name, _label_dir_name in ['real', 'fake'])
                if _label_dir_name in ['real', 'fake']: # very bad idea
                    _keys = {"real": 0, "fake": 1}
                    _label_index = _keys[_label_dir_name] 
                dict_label[_label_index] = _label_dir_name
                print("Reading from {}".format(_label_dir + "/*.{}".format(ext)))
                paths = list(glob(_label_dir + "/*.{}".format(ext)))
                print("paths", paths)
                for path in paths:
                    print(path)
                    tmp.append([path, _label_index])
            pickle_write(path="{}/{}.pkl".format(root_dir.strip("*"), _dir_name), _list=tmp)
    return dict_label
    
if __name__ == "__main__":
    # embedder = FaceEmbedder()
    # _path = "./data/collected_data/*"
    # convert(embedder, _path)
    # print(generate_dir_meta(root_dirs=["./data/image_recognition/processed/*"], ext="npy"))
    # print(generate_dir_meta(root_dirs=["./data/print_attack/processed/*"], ext="npy"))