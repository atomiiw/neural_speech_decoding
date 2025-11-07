# Duke Compute Cluster (DCC) Guide

## Prerequisites
1. DCC account with NetID
2. Miniconda installed on DCC (if not, see below)

## Initial Setup (One Time)

### 1. Install Miniconda (if not already installed)
```bash
# SSH into DCC
ssh mw582@dcc-login.oit.duke.edu

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, accept default location: /hpc/dctrl/YOUR_NETID/miniconda3
```

### 2. Upload Data and Code
```bash
# On your local machine
scp -r neural_speech_decoding mw582@dcc-login.oit.duke.edu:~/

# Upload HB02 data to DCC
scp HB02.h5 mw582@dcc-login.oit.duke.edu:~/neural_speech_decoding/example_data/
```

### 3. Configure Paths
```bash
# SSH into DCC
cd ~/neural_speech_decoding

# Update config file
python3 << 'PYTHON'
import json
with open('configs/AllSubjectInfo.json', 'r') as f:
    config = json.load(f)
config['Shared']['RootPath'] = './example_data/'
with open('configs/AllSubjectInfo.json', 'w') as f:
    json.dump(config, f, indent=4)
PYTHON
```

## Running Training

### 1. Edit SLURM Script
```bash
nano run_training.slurm
```
- Change `YOUR_NETID@duke.edu` to your actual Duke email

### 2. Check GPU Availability
```bash
# See available GPUs in real-time
gpuavail

# Check queue
squeue -p gpu-common
```

### 3. Submit Job
```bash
sbatch run_training.slurm
```

### 4. Monitor Job
```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f logs/train_39333116.out

# Check error log if needed
tail -f logs/train_39333116.err
```

### 5. Cancel Job (if needed)
```bash
scancel JOBID
```

## DCC GPU Partitions

**gpu-common** (public access):
- a5000 (24GB) - 76 nodes
- 2080 (11GB) - 74 nodes  
- 5000_ada (32GB) - 28 nodes
- 6000_ada (48GB) - 24 nodes
- p100 (16GB) - 10 nodes

**Researcher-owned** (may need permission):
- a100 (80GB)
- h100 (80GB)

To request specific GPU: `#SBATCH --gres=gpu:a5000:1`

## Troubleshooting

### "QOSMaxSubmitJobPerUserLimit"
Too many jobs submitted. Wait for current jobs to finish or cancel some.

### "Job pending for a long time"
Try `scavenger-gpu` partition (lower priority but faster access):
```bash
#SBATCH -p scavenger-gpu
```

### Check disk space
```bash
du -sh ~/neural_speech_decoding
quota
```

### Module errors
DCC uses Miniconda, not system modules. Don't use `module load python`.

## Expected Timeline
- Stage 1 (a2a): ~6 hours on A5000/5000_ada
- Stage 2 (e2a): ~10 hours on A5000/5000_ada
- **Total: ~16 hours**

## Output
Final weights: `~/neural_speech_decoding/output/e2a/resnet_HB02/model_epoch59.pth`

Download to local machine:
```bash
scp YOUR_NETID@dcc-slogin.oit.duke.edu:~/neural_speech_decoding/output/e2a/resnet_HB02/model_epoch59.pth .
```

## Resources
- DCC Documentation: https://oit-rc.pages.oit.duke.edu/rcsupportdocs/
- GPU Scheduling: https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/slurm/
- Support: rc-help@duke.edu
