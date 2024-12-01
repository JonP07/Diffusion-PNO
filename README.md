# Diffusion-PNO
This is the official implementation for the paper "Safeguarding Text-to-Image Generation via Inference-Time Prompt-Noise Optimization" Peng et al.

## Introduction
![demo](images/system_fig3.png)

## Running the code
### Setup
```bash
conda env create -f environment.yaml
conda activate Diffusion-PNO
```

### Safegaurding SD1.5 outputs
```bash
python pno_main.py --dataset i2p_benchmark_harassment_hardest --objective Q16 --output_path ./output_folder
```
