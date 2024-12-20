# swift run StableDiffusionSample "A dog running on the highway" --resource-path ml_package_sdxl_swift/Resources --output-path swift_results --compute-units all --xl
# swift run StableDiffusionSample "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" --resource-path ml_package_sdxl_swift/Resources \
# --output-path swift_results --compute-units all --xl --image "test_cases/img2img-sdxl-init.png" --seed 93 --strength 0.8 --step-count 30


swift run StableDiffusionSample "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
--resource-path ml_package_sdxl_swift/Resources \
--output-path swift_results \
--compute-units all \
--xl \
--image "test_cases/img2img-sdxl-init.png" \
--seed 93 \
--strength 0.8 \
--step-count 30 \
--controlnet R3GmControlnetRecolorSdxlFp16 \
--controlnet-inputs "test_cases/img2img-sdxl-init.png"
