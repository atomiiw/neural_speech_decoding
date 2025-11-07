#!/bin/bash
#SBATCH --job-name=neural_speech_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH -p gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mw582@duke.edu

# Initialize conda (DCC uses Miniconda, not system modules)
source ~/.bashrc
eval "$(conda shell.bash hook)"

# Create logs directory
mkdir -p logs

# Setup conda environment (first time only)
if ! conda env list | grep -q "neural_speech"; then
    echo "Creating conda environment..."
    conda create -n neural_speech python=3.10 -y
    conda activate neural_speech
    pip install --upgrade pip
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install -r requirements.txt
else
    conda activate neural_speech
fi

# Verify GPU
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Stage 1: Audio-to-Audio Training (a2a)
echo "========================================="
echo "Starting Stage 1: Audio-to-Audio (a2a)"
echo "========================================="
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

# Check if Stage 1 completed
if [ $? -eq 0 ]; then
    echo "✓ Stage 1 completed successfully"
    
    # Stage 2: ECoG-to-Audio Training (e2a)
    echo "========================================="
    echo "Starting Stage 2: ECoG-to-Audio (e2a)"
    echo "========================================="
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
    
    if [ $? -eq 0 ]; then
        echo "✓✓✓ Training complete! ✓✓✓"
        echo "Final weights: output/e2a/resnet_HB02/model_epoch59.pth"
    else
        echo "✗ Stage 2 failed"
        exit 1
    fi
else
    echo "✗ Stage 1 failed"
    exit 1
fi
