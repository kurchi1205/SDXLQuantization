export DIFFUSERS_ROOT=$PWD
git clone https://github.com/mlcommons/inference.git
cd inference
git checkout 87ba8cb8a6a4f6525f26255fa513d902b17ab060
cd ./text_to_image/tools/
sh ./download-coco-2014.sh --num-workers 5
sh ./download-coco-2014-calibration.sh -n 5
cd ${DIFFUSERS_ROOT}
export PYTHONPATH="${DIFFUSERS_ROOT}/inference/text_to_image/:$PYTHONPATH"
