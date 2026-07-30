
## How to use this

To replicate any paper and experiments, open kaggle or colab, and paste the following
```python
import os

REPO = "https://github.com/ursaj123/PapersCode.git"

if not os.path.exists("paperscode"):
    !git clone {REPO}

%cd PapersCode
!pip install -e .
```

and then go through readme of specifc paper for further instructions, like for example, see [spectral_net](src/paperscode/spectral_net/), 
- I've been using most simplest of the datasets for all the training and experimenting purposes.
- I will be using MNIST for this paper (thus there is no data.py files for data preprocessing and all, I've been using it directly through torchvision datasets via downloading it.)
- First we train the siamese network (optional though, see [siamese](src/paperscode/spectral_net/siamese/) for more details),
    ```shell
    !python src/paperscode/spectral_net/siamese/train.py
    ```
- The results are stored in 
    


Most of these papers are 


