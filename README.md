### How to Use
> IMPORTANT: Windows is not supported. Use Linux or MacOS.  
> It might work on windows but is not tested.  
> Use WSL if you want to run it on Windows.

#### Data Preprocessing and Caching

1. Prerequisites
- Ensure you have uv installed.
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

2. Clone the repository
```bash
git clone https://github.com/riverfog7/ai_experiment_project
```

3. Navigate to the project directory and install dependencies
```bash
cd ai_experiment_project
uv sync --dev --extra <cu126 | cu128 | cu130 | cpu> # choose the proper extra based on your system
```

4. Download the dataset
```bash
./scripts/download_dataset.py \ 
  --dataset-id 71362 \
  --output-dir ./datasets \
  --log-file download_progress.txt
```

5. Convert the dataset to huggingface format
```bash
./scripts/convert_hf.py \
  ./datasets \
  ./hf_datasets/recyclables_image_classification \
  --convert_to image_classification \
  --cache_dir ./.temp
```

6. Or download the pre-converted huggingface dataset
```bash
./scripts/preload_dataset.sh
```

7. Preprocess and cache the dataset by running `easy_load()` in python
```python
from aie_project.training.dataset_utils import easy_load

easy_load(
    data_path="./datasets/recyclables_image_classification",
)
```

8. Or download the preprocessed and cached dataset
```bash
./scripts/preload_cache.sh
```

9. Optionally log in to wandb for logging
```bash
uv tool run wandb login
```

#### Hyperparameter Optimization

1. Set up a PostgreSQL database, S3 bucket and configure environment variables
- POSTGRES_PASSWORD
- POSTGRES_HOST
- POSTGRES_PORT
- POSTGRES_USER
- POSTGRES_DB
- OPTUNA_STUDY_NAME
- OPTUNA_S3_BUCKET
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY

2. Run hyperparameter optimization
```bash
./scripts/tune.sh
```

#### Model Training
1. Configure the proper hyperparameters in `./aie_project/training/train.py`
2. Run the training script
```bash
./scripts/train.sh
```
