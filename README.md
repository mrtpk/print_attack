### 2D print attack detection

#### Data preparation

Face crops are needed to train. Cropped faces has to be converted to face embeddings and should be placed in `data/image_recognition/processed` and `data/print_attack/processed`. A sample structure is in the `data` directory. Cropped faces can be converted to face embeddings using `convert()` in `convert_imgs.py`. After placing the embeddings in the required folder structure, run `generate_dir_meta()` in `convert_imgs.py` for `data/image_recognition/processed` and `data/print_attack/processed`.

#### Training

The DL models are trainined using `train_dnn.py`. The ML models- SVM, GMM, and logistic regresssion can be trained using `svm.py`, `gmm.py` and `log_regression.py`.

The face recognizer is trained using `face_recognition.py`.

#### Inference

Real time inference can be done using `app.py`.

#### Setup:
* Install Anaconda.
* `conda env create -f environment.yml` to create an environment.
* `conda activate print2d` to activate the environment.