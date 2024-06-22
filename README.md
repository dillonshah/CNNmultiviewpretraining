# Multi-view Image Synthesis for Breast Cancer Detection
For use with the EMory BrEast Imaging Dataset (EMBED).
This repository contains the code required to train a model in multi-view synthesis (MVS), where a view of the breast is generated from the opposing view. This serves as a pretraining task for downstream classification problems.

src/classification provides code required for downstream or supervised classification for breast density prediction and breast cancer detection.
src/pretraining provides code for the MVS pretraining.

## Example usage
Set up and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install requirements.txt
```

For pretraining:

```bash
python pretraining/pretrain.py --num_devices 2 # For 2 GPUs
```

For downstream breast density classification:

```bash
python classification/classification.py --density --pretrained --model_path <model path>
```
For downstream breast cancer classification:

```bash
python classification/classification.py --pretrained --model_path <model path>
```
**Note:** --help will list usage for each file.
