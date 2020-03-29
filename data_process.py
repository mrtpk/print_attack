import pathlib
import numpy as np
import cv2
from glob import glob
from embedding import FaceEmbedder
import pickle
from sklearn.model_selection import train_test_split

def pickle_write(path, _list):
    with open(path, 'wb') as fp:
        pickle.dump(_list, fp)

def pickle_read(path):
    with open (path, 'rb') as fp:
        _list = pickle.load(fp)
    return _list

def create_directory(_path):
    """
    Creates directory with :param _path
    """
    pathlib.Path(_path).mkdir(parents=True, exist_ok=True)
    
def cvt_face_embeddings(embedder, root_dirs):
    """
    Converts images in the root directory to corresponding
    feature embedding preserving the folder structure.
    """
    for root_dir in root_dirs:
        dirs = list(glob(root_dir))
        for _dir in dirs:
            create_directory(_path = _dir.replace("raw", "processed"))
            paths_img = list(glob(_dir + "/*.jpg"))
            for path_img in paths_img:
                path_npy = path_img.replace("raw", "processed").replace(".jpg", ".npy")
                img = cv2.imread(path_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                feature = embedder.get_embedding(img)
                np.save(path_npy, feature)
                print(path_npy)
                # break
            # break
        # break

def generate_dir_meta(root_dirs, ext="jpg"):
    """
    Generate inputs and label meta from folder
    """
    dict_label = {}
    for root_dir in root_dirs:
        dirs = list(glob(root_dir))
        tmp = []
        for _dir in dirs:
            _dir_content = list(glob(_dir + "/*"))
            _dir_name = _dir.split("/")[-1]
            # print(_dir, _dir_content)
            for i, _label_dir in enumerate(_dir_content):
                # print(i, _label_dir)
                dict_label[i] = _label_dir.split("/")[-1]
                paths = list(glob(_label_dir + "/*.{}".format(ext)))
                # print("paths", paths)
                for path in paths:
                    tmp.append([path, i])
            pickle_write(path="{}/{}.pkl".format(root_dir.strip("*"), _dir_name), _list=tmp)
    return dict_label

def get_xy(label, x):
    _label = {"ClientFace": 0,
             "ImposterFace": 1}
    prefix = "./data/print_attack/processed/{}/".format(label)
    x = x.split(" ")[0]
    x = x.replace(".jpg", ".npy").replace("\\", "/")
    return [prefix + x, _label[label]]

def get_names_labels(label, operation):
    dataset = {"ClientFace": './data/print_attack/raw/client_{}_face.txt',
               "ImposterFace": './data/print_attack/raw/imposter_{}_face.txt'}
    with open(dataset[label].format(operation)) as fp:
        lines = fp.read().splitlines()
        lines = list(map(lambda x : get_xy(label, x), lines))
    return lines


def prepare_train_list():
    _train_lines = get_names_labels(label="ClientFace", operation="train")
    print("Training set contains {} fake samples".format(len(_train_lines)))
    train_lines = get_names_labels(label="ImposterFace", operation="train")
    print("Training set contains {} real samples".format(len(train_lines)))
    _lines = train_lines + _train_lines
    pickle_write("./data/print_attack/processed/train.pkl", _lines)
    print("Training set contains {} samples".format(len(_lines)))

def prepare_test_val_list():
    real_lines = get_names_labels(label="ClientFace", operation="test")
    real_test, real_val = train_test_split(real_lines, test_size=0.40, random_state=42)
    fake_lines = get_names_labels(label="ImposterFace", operation="test")
    fake_test, fake_val = train_test_split(fake_lines, test_size=0.40, random_state=42)
    pickle_write("./data/print_attack/processed/valid.pkl", real_val + fake_val)
    print("Validation set has {} real and {} fake samples".format(len(real_val), len(fake_val)))
    pickle_write("./data/print_attack/processed/test.pkl", real_test + fake_test)
    print("Test set has {} real and {} fake samples".format(len(real_test), len(fake_test)))    

if __name__ == "__main__":
    # embedder = FaceEmbedder()
    # # Convert faces to embeddings for 2d attack
    # cvt_face_embeddings(embedder, root_dirs = ["./data/print_attack/raw/ImposterFace/*", "./data/print_attack/raw/ClientFace/*"])
    # # Prepare train, validation and test dataset for attack classfiers
    # prepare_train_list()
    # prepare_test_val_list()
    # Convert faces to embeddings for face recognition
    # cvt_face_embeddings(embedder, root_dirs = ["./data/image_recognition/raw/test/*", "./data/image_recognition/raw/train/*", "./data/image_recognition/raw/valid/*"])
    # print(generate_dir_meta(root_dirs=["./data/image_recognition/processed/*"], ext="npy"))
    embedder = FaceEmbedder()
    # Convert faces to embeddings for 2d attack
    cvt_face_embeddings(embedder, root_dirs = ["./presentation_attack/data/collected_data/fake/crop/*"])
