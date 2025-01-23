# Quantizing SDXL for MLPerf

The purpose of this document is to allow other users to quantize SDXL in various different ways, corresponding to AMD's 2024-07 MLPerf submission.

## Environmental Setup

The environment can be set up as follows:

```bash
conda env create -n quant_sdxl -f env/brv_sdxl_mlperf_minimal.yml # Use env/brv_sdxl_mlperf_rocm_minimal.yml is using a ROCm compatible GPU
conda activate quant_sdxl
```

## Fetching and Pre-Processing Datasets

Information on fetching and pre-processing the data can be found [here](https://github.com/mlcommons/inference/tree/master/text_to_image).
Note, the latents generated here will be used in the next step.

## Quantize Int8 / FP16 Model

```bash
python quant_sdxl.py --model stabilityai/stable-diffusion-xl-base-1.0 --device <device> --calibration-prompt-path ./captions.tsv --checkpoint-name unet.ckpt  --path-to-latents <path/to/latents/latents.pt> --guidance-scale 7.5 --exclude-blacklist-act-eq [--path-to-coco <path/to/coco> --validation-prompts 5000]
```

Note, you can optionally validate on the MLPerf validation set, but be aware this will take ~48hrs.

## Quantize Int8 / FP8 Model

The Int8 / FP8 model can be quantized as follows:

```bash
python quant_sdxl.py --model stabilityai/stable-diffusion-xl-base-1.0 --device <device> --calibration-prompt-path ./captions.tsv --checkpoint-name unet.ckpt  --path-to-latents <path/to/latents/latents.pt> --guidance-scale 7.5 --quantize-sdp --exclude-blacklist-act-eq [--path-to-coco <path/to/coco> --validation-prompts 5000]
```
