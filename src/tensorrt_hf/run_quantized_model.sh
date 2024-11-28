cd TensorRT/demo/Diffusion
# python3 demo_txt2img_xl.py \
#   "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
#   --build-static-batch \
#   --use-cuda-graph \
#   --num-warmup-runs 1 \
#   --width 1024 \
#   --height 1024 \
#   --denoising-steps 30 \
#   --onnx-base-dir ../stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base \
#   --onnx-refiner-dir ../stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-refiner

python demo_txt2img_xl.py "enchanted winter forest with soft diffuse light on a snow-filled day" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl --int8 --quantization-level 3