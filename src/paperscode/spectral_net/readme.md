# Spectral Net Paper Implementation

- Paper [SpectralNet: Spectral Clustering using Deep Neural Networks
](https://arxiv.org/pdf/1801.01587)
- Check out blog for more deep dive on paper - [Research Roadmap](https://ursaj123.github.io/posts/Machine-Learning-Research-Roadmap/)


## Simple Experiments
- I will be using MNIST for this paper (thus there is no data.py files for data preprocessing and all, I've been using it directly through torchvision datasets via downloading it.)
- First we train the siamese network (optional though, see [siamese](siamese/) for more details),
    ```
    python src/paperscode/spectral_net/siamese/train.py \
    --run-name siamese_twin_training_for_spectral_clustering_on_mnist \
    --output-dir src/paperscode/spectral_net/siamese/runs
    ```
- 



