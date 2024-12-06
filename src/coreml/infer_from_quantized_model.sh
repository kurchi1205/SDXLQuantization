python infer.py --prompt "A dog is running on highway." -i ml_package_sdxl_swift -o results --model-version stabilityai/stable-diffusion-xl-base-1.0 \
--controlnet r3gm_controlnet-recolor-sdxl-fp16 \
--controlnet-inputs "test_cases/img2img-sdxl-init.png" \
--image2image --model-sources "packages"