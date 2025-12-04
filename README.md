### How to Use
> IMPORTANT: Windows is not supported. Use Linux or MacOS.  
> It might work on windows but is not tested.  
> Use WSL if you want to run it on Windows.

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
