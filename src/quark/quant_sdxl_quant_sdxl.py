"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
import copy
from datetime import datetime
import json
import os
import time

from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas_examples.common.generative.nn import LoRACompatibleQuantConv2d, LoRACompatibleQuantLinear
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_processor import AttnProcessor
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import brevitas.nn as qnn

from brevitas.graph.base import ModuleToModuleByClass
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.inject.enum import StatsOp
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.utils.torch_utils import KwargsForwardHook
import brevitas.config as config

from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.stable_diffusion.sd_quant.export import export_quant_params
from brevitas_examples.stable_diffusion.sd_quant.nn import QuantAttention
from brevitas_examples.stable_diffusion.mlperf_evaluation.accuracy import compute_mlperf_fid

TEST_SEED = 123456
torch.manual_seed(TEST_SEED)

class WeightQuant(ShiftedUint8WeightPerChannelFloat):
    narrow_range = False
    scaling_min_val = 1e-4
    quantize_zero_point = True
    scaling_impl_type = 'parameter_from_stats'
    zero_point_impl = ParameterFromStatsFromParameterZeroPoint

class InputQuant(Int8ActPerTensorFloat):
    scaling_stats_op = StatsOp.MAX

class OutputQuant(Fp8e4m3FNUZActPerTensorFloat):
    scaling_stats_op = StatsOp.MAX

NEGATIVE_PROMPTS = ["normal quality, low quality, worst quality, low res, blurry, nsfw, nude"]

def load_calib_prompts(calib_data_path, sep="\t"):
    df = pd.read_csv(calib_data_path, sep=sep)
    lst = df["caption"].tolist()
    return lst

def run_val_inference(
        pipe,
        prompts,
        guidance_scale,
        total_steps,
        test_latents=None):
    with torch.no_grad():
        for prompt in tqdm(prompts):
            # We don't want to generate any image, so we return only the latent encoding pre VAE
            pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPTS[0],
                latents=test_latents,
                output_type='latent',
                guidance_scale=guidance_scale,
                num_inference_steps=total_steps)


def main(args):

    dtype = getattr(torch, args.dtype)

    calibration_prompts = load_calib_prompts(args.calibration_prompt_path)
    assert args.calibration_prompts <= len(calibration_prompts) , f"--calibration-prompts must be <= {len(calibration_prompts)}"
    calibration_prompts = calibration_prompts[:args.calibration_prompts]
    latents = torch.load(args.path_to_latents).to(torch.float16)

    # Create output dir. Move to tmp if None
    ts = datetime.fromtimestamp(time.time())
    str_ts = ts.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f'{str_ts}')
    os.mkdir(output_dir)
    print(f"Saving results in {output_dir}")

    # Dump args to json
    with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp)

    # Load model from float checkpoint
    print(f"Loading model from {args.model}...")
    variant = 'fp16' if dtype == torch.float16 else None
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, variant=variant, use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.vae.config.force_upcast=True
    print(f"Model loaded from {args.model}.")

    # Move model to target device
    print(f"Moving model to {args.device}...")
    pipe = pipe.to(args.device)

    # Enable attention slicing
    if args.attention_slicing:
        pipe.enable_attention_slicing()

    # Extract list of layers to avoid
    blacklist = []
    for name, _ in pipe.unet.named_modules():
        if 'time_emb' in name:
            blacklist.append(name.split('.')[-1])
    print(f"Blacklisted layers: {blacklist}")

    # Make sure there all LoRA layers are fused first, otherwise raise an error
    for m in pipe.unet.modules():
        if hasattr(m, 'lora_layer') and m.lora_layer is not None:
            raise RuntimeError("LoRA layers should be fused in before calling into quantization.")

    pipe.set_progress_bar_config(disable=True)

    if args.load_checkpoint is not None:
        with load_quant_model_mode(pipe.unet):
            pipe = pipe.to('cpu')
            print(f"Loading checkpoint: {args.load_checkpoint}... ", end="")
            pipe.unet.load_state_dict(torch.load(args.load_checkpoint, map_location='cpu'))
            print(f"Checkpoint loaded!")
        pipe = pipe.to(args.device)

    if args.load_checkpoint is not None:
        # Don't run full activation equalization if we're loading a quantized checkpoint
        num_ae_prompts = 2
    else:
        num_ae_prompts = len(calibration_prompts)
    with activation_equalization_mode(
            pipe.unet,
            alpha=args.act_eq_alpha,
            layerwise=True,
            blacklist_layers=blacklist if args.exclude_blacklist_act_eq else None,
            add_mul_node=True):
        # Workaround to expose `in_features` attribute from the Hook Wrapper
        for m in pipe.unet.modules():
            if isinstance(m, KwargsForwardHook) and hasattr(m.module, 'in_features'):
                m.in_features = m.module.in_features
        total_steps = args.calibration_steps
        run_val_inference(
            pipe,
            calibration_prompts[:num_ae_prompts],
            total_steps=total_steps,
            test_latents=latents,
            guidance_scale=args.guidance_scale)

    # Workaround to expose `in_features` attribute from the EqualizedModule Wrapper
    for m in pipe.unet.modules():
        if isinstance(m, EqualizedModule) and hasattr(m.layer, 'in_features'):
            m.in_features = m.layer.in_features


    quant_layer_kwargs = {
    'input_quant': InputQuant, 'weight_quant': WeightQuant, 'dtype': dtype, 'device': args.device, 'input_dtype': dtype, 'input_device': args.device}
    quant_linear_kwargs = copy.deepcopy(quant_layer_kwargs)
    if args.quantize_sdp:
        output_quant = OutputQuant
        rewriter = ModuleToModuleByClass(
            Attention,
            QuantAttention,
            softmax_output_quant=output_quant,
            query_dim=lambda module: module.to_q.in_features,
            dim_head=lambda module: int(1 / (module.scale ** 2)),
            processor=AttnProcessor(),
        is_equalized=True)
        config.IGNORE_MISSING_KEYS = True
        pipe.unet = rewriter.apply(pipe.unet)
        config.IGNORE_MISSING_KEYS = False
        pipe.unet = pipe.unet.to(args.device)
        pipe.unet = pipe.unet.to(dtype)
        # quant_kwargs = layer_map[torch.nn.Linear][1]
        what_to_quantize = ['to_q', 'to_k', 'to_v']
        quant_linear_kwargs['output_quant'] = lambda module, name: output_quant if any(ending in name for ending in what_to_quantize) else None
        quant_linear_kwargs['output_dtype'] = dtype
        quant_linear_kwargs['output_device'] = args.device

    layer_map = {
    nn.Linear: (qnn.QuantLinear, quant_linear_kwargs),
    nn.Conv2d: (qnn.QuantConv2d, quant_layer_kwargs),
    'diffusers.models.lora.LoRACompatibleLinear':
        (LoRACompatibleQuantLinear, quant_linear_kwargs),
    'diffusers.models.lora.LoRACompatibleConv': (LoRACompatibleQuantConv2d, quant_layer_kwargs)}

    pipe.unet = layerwise_quantize(
        model=pipe.unet, compute_layer_map=layer_map, name_blacklist=blacklist)
    print("Model quantization applied.")

    pipe.set_progress_bar_config(disable=True)

    if args.load_checkpoint is None:
        print("Applying activation calibration")
        with torch.no_grad(), calibration_mode(pipe.unet):
            run_val_inference(
                pipe,
                calibration_prompts,
                total_steps=args.calibration_steps,
                test_latents=latents,
                guidance_scale=args.guidance_scale)

        print("Applying bias correction")
        with torch.no_grad(), bias_correction_mode(pipe.unet):
            run_val_inference(
                pipe,
                calibration_prompts,
                total_steps=args.calibration_steps,
                test_latents=latents,
                guidance_scale=args.guidance_scale)

        if args.checkpoint_name is not None:
            torch.save(pipe.unet.state_dict(), os.path.join(output_dir, args.checkpoint_name))

    # Perform inference
    if args.validation_prompts > 0:
        print(f"Computing validation accuracy")
        compute_mlperf_fid(args.model, args.path_to_coco, pipe, args.validation_prompts, output_dir)

    if args.export_target:
        pipe.unet.to('cpu').to(dtype)
        export_quant_params(pipe, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stable Diffusion quantization')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default=None,
        help='Path or name of the model.')
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:0', help='Target device for quantized model.')
    parser.add_argument(
        '--calibration-prompt-path', type=str, default=None, help='Path to calibration prompt')
    parser.add_argument(
        '--calibration-prompts',
        type=int,
        default=500,
        help='Number of prompts to use for calibration. Default: %(default)s')
    parser.add_argument(
        '--validation-prompts',
        type=int,
        default=0,
        help='Number of prompt to use for validation. Default: %(default)s')
    parser.add_argument(
        '--path-to-coco',
        type=str,
        default=None,
        help=
        'Path to MLPerf compliant Coco dataset. Required when the --validation-prompts > 0 flag is set. Default: None'
    )
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default=None,
        help=
        'Name to use to store the checkpoint in the output dir. If not provided, no checkpoint is saved.'
    )
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to load. If provided, PTQ techniques are skipped.')
    parser.add_argument(
        '--path-to-latents',
        type=str,
        required=True,
        help=
        'Path to pre-defined latents.')
    parser.add_argument('--guidance-scale', type=float, default=8., help='Guidance scale.')
    parser.add_argument(
        '--calibration-steps', type=float, default=8, help='Steps used during calibration')
    add_bool_arg(
        parser,
        'output-path',
        str_true=True,
        default='.',
        help='Path where to generate output folder.')
    parser.add_argument(
        '--dtype',
        default='float16',
        choices=['float32', 'float16', 'bfloat16'],
        help='Model Dtype, choices are float32, float16, bfloat16. Default: float16')
    add_bool_arg(
        parser,
        'attention-slicing',
        default=False,
        help='Enable attention slicing. Default: Disabled')
    add_bool_arg(
        parser,
        'export-target',
        default=True,
        help='Export flow.')
    parser.add_argument(
        '--act-eq-alpha',
        type=float,
        default=0.9,
        help='Alpha for activation equalization. Default: 0.9')
    add_bool_arg(parser, 'quantize-sdp', default=False, help='Quantize SDP. Default: Disabled')
    add_bool_arg(
        parser,
        'exclude-blacklist-act-eq',
        default=False,
        help='Exclude unquantized layers from activation equalization. Default: Disabled')
    args = parser.parse_args()
    print("Args: " + str(vars(args)))
    main(args)
