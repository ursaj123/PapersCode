
## Setup

To replicate any paper and experiments, open kaggle or colab, and paste the following
```python
import os

REPO = "https://github.com/ursaj123/PapersCode.git"

if not os.path.exists("paperscode"):
    !git clone {REPO}

%cd PapersCode
!pip install -e .
```


There are central files which I would be used for almost every paper, check [common](src/paperscode/common) for that, 
- Like [trainer.py](src/paperscode/common/trainer.py), the training args look like
    ```
    python train.py \
    --run-name spectral_v2 \
    --output-dir runs \
    --max-epochs 300 \
    --batch-size 1024 \
    --lr 5e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --precision bfloat16 \
    --ema \
    --no-early-stopping \
    --save-best \
    --log-level INFO \
    --device cuda
    ```
    see [cli.py](src/paperscode/common/cli.py) for more details and default parameters.

- I'll be adding common CustomDataset and CustomLosses , online batch processing patterns.

As of now, I've been using most simplest of the datasets for all the training and experimenting purposes.
    

Go through readme of specifc paper for further instructions, like for example, see [spectral_net](src/paperscode/spectral_net/), 




