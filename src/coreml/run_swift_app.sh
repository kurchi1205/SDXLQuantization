# swift run StableDiffusionSample "A bob h÷air girl" --resource-path ../../../SDXL/Resources_IOS --output-path swift_results --compute-units cpuAndNeuralEngine --xl --step-count 30
swift run StableDiffusionSample "A bob hair girl" --resource-path ml_package_sdxl_swift/Resources --output-path swift_results --compute-units cpuAndNeuralEngine --xl --step-count 30 --image "swift_results/A_bob_hair_girl.1050622739.final.png" 
# swift run StableDiffusionSample "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" --resource-path ml_package_sdxl_swift/Resources \
# --output-path swift_results --compute-units all --xl --image "test_cases/img2img-sdxl-init.png" --seed 93 --strength 0.8 --step-count 2


# swift run StableDiffusionSample "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
# --resource-path ml_package_sdxl_swift/Resources \
# --output-path swift_results \
# --compute-units all \
# --xl \
# --image "test_cases/img2img-sdxl-init.png" \
# --seed 93 \
# --strength 0.8 \
# --step-count 30 \
# --controlnet R3GmControlnetRecolorSdxlFp16 \
# --controlnet-inputs "test_cases/img2img-sdxl-init.png"
