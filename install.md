To install create a virtual environment and install the dependencies:

first test 
    ```nvidia-smi``` 


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy sentencepiece huggingface-hub datasets tqdm
```

should be ready to run

```bash
RUN_ID=wsl_gpu_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt.py
```