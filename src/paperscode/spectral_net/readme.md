# Spectral Net Paper Implementation

- Paper [SpectralNet: Spectral Clustering using Deep Neural Networks
](https://arxiv.org/pdf/1801.01587)
- Check out blog for more deep dive on paper - [SpectralNet: Teaching a Neural Network to Cluster Like an Eigenvector Solver
](https://ursaj123.github.io/posts/Spectral-Net/)


## Simple Experiments
- I will be using MNIST for this paper (thus there is no data.py files for data preprocessing and all, I've been using it directly through torchvision datasets via downloading it.)
- Start a kaggle notebbok, and run on 2T4 GPU or a simple A100.
- First, clone the repo
    ```python
    import os

    REPO = "https://github.com/ursaj123/PapersCode.git"

    if not os.path.exists("paperscode"):
        !git clone {REPO}

    %cd PapersCode
    !pip install -e .
    ```
- Then, we first train the siamese network for good latent embeddings (optional though, see [siamese](siamese/) for more details),
    ```python
    exec_file = "src/paperscode/spectral_net/siamese/train.py"
    siamese_run_name = "siamese_twin_training_for_spectral_clustering_on_mnist"
    siamese_output_dir = "src/paperscode/spectral_net/siamese/runs"
    siamese_max_epochs = 5


    !python {exec_file} --run-name {siamese_run_name} --output-dir {siamese_output_dir} --max-epochs {siamese_max_epochs}

    ```

- You can see, the training logs, in the runs folder
    ```python
    siamese_path = siamese_output_dir + "/" + siamese_run_name
    
    # the training logs
    !cat $siamese_path/trainer.log


    # train losses
    import json

    with open(siamese_path + '/history.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    print(data)


    ```
- Then, we train the spectral net with $W$ being made from latent embeddings of siamese network (weight frozen),
    ```python
    exec_file_spectral = "src/paperscode/spectral_net/train.py"
    spectral_run_name = "spectral_clustering_on_mnist"
    spectral_output_dir = "src/paperscode/spectral_net/runs"
    spectral_max_epochs = 5
    siamese_ckpt_path = siamese_path + "/best.pt"
    affinity_matrix_clusters = 10
    output_clusters = 10
    precision = "float32"

    !python {exec_file_spectral} --run-name {spectral_run_name} --output-dir {spectral_output_dir} --max-epochs {spectral_max_epochs} --siamese-path {siamese_ckpt_path} --affinity-matrix-clusters {affinity_matrix_clusters} --precision {precision} --output-clusters {output_clusters}
    ```

- The resulting model, logs and training metrics plot are stored in corresponding directories of the model.




