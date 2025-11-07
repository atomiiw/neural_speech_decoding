# Setup Instructions

## Requirements
- **Python 3.10** (NOT 3.11 or 3.12 - PyTorch 1.12.1 incompatible)
- CUDA 11.6 compatible GPU
- ~100GB disk space for data + outputs

## Installation

### Option 1: Conda (Recommended)
```bash
conda create -n neural_speech python=3.10
conda activate neural_speech
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Option 2: SLURM Cluster
```bash
#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

module load python/3.10
module load cuda/11.6

python3 -m venv venv
source venv/bin/activate
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## ⚠️ Google Colab NOT Supported
Colab uses Python 3.12 which is incompatible with PyTorch 1.12.1.
Use conda or SLURM instead.

## Data Setup
1. Download HB02 data from: https://data.mendeley.com/datasets/fp4bv9gtwk/2
2. Place in `example_data/` directory
3. Update `configs/AllSubjectInfo.json` RootPath to `./example_data/`

## Training

### Stage 1: Audio-to-Audio (a2a)
```bash
python train_a2a.py \
  --OUTPUT_DIR output/a2a/HB02 \
  --trainsubject HB02 \
  --testsubject HB02 \
  --param_file configs/a2a_production.yaml \
  --batch_size 16 \
  --reshape 1 \
  --DENSITY "HB" \
  --wavebased 1 \
  --n_filter_samples 80 \
  --n_fft 256 \
  --formant_supervision 1 \
  --intensity_thres -1 \
  --epoch_num 60
```

### Stage 2: ECoG-to-Audio (e2a)
```bash
python train_e2a.py \
  --OUTPUT_DIR output/e2a/resnet_HB02 \
  --trainsubject HB02 \
  --testsubject HB02 \
  --param_file configs/e2a_production.yaml \
  --batch_size 16 \
  --MAPPING_FROM_ECOG ECoGMapping_ResNet \
  --reshape 1 \
  --DENSITY "HB" \
  --wavebased 1 \
  --dynamicfiltershape 0 \
  --n_filter_samples 80 \
  --n_fft 256 \
  --formant_supervision 1 \
  --intensity_thres -1 \
  --epoch_num 60 \
  --pretrained_model_dir output/a2a/HB02 \
  --causal 0
```

## Fixes Applied
- ✓ `metrics/torch_stoi.py`: Disabled unused Resample transform (CUDA device mismatch)
- ✓ `requirements.txt`: Locked to PyTorch 1.12.1 + Python 3.10

## Output
Final weights: `output/e2a/resnet_HB02/model_epoch59.pth`
