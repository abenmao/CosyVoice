"""Minimal CosyVoice3 inference example targeting Intel Arc B60 (torch.xpu).

This script expects the CosyVoice3 model weights to reside under
./Fun-CosyVoice3-0.5B-2512 and requires intel-extension-for-pytorch.

Example:
    python cosyvoice3.py --text "你好，欢迎使用 CosyVoice3" --output demo.wav

Supported output files: .wav, .mp3 (requires torchaudio built with FFmpeg).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.profiler import ProfilerActivity, profile, schedule

try:
    sys.path.append('third_party/Matcha-TTS')
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError as ex:
    raise ImportError(
        "Please install the cosyvoice package (pip install cosyvoice)."
    ) from ex

try:
    import torchaudio
except ImportError as ex:
    raise ImportError(
        "torchaudio is required for saving the generated waveform."
    ) from ex


def save_audio(output_path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Persist the waveform, supporting wav and mp3 containers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels_first = waveform.detach().to("cpu")
    if channels_first.dim() != 2:
        raise ValueError("Waveform tensor must have shape (channels, frames).")

    suffix = output_path.suffix.lower()
    print("Saving audio to ", output_path)
    torchaudio.save(str(output_path), channels_first, sample_rate)
    return


def resolve_device(strict: bool = True) -> torch.device:
    """Return the Intel Arc XPU device, optionally falling back to CPU."""
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(0)
        return torch.device("xpu")
    if strict:
        raise RuntimeError(
            "torch.xpu device not found. Ensure an Intel Arc GPU and IPEX are available."
        )
    return torch.device("cpu")


def run_inference_zero_shot(
    model,
    text: str,
    prompt: str,
    prompt_speech: Path,
    stream: bool = True,
):
    """Execute CosyVoice3 inference and return the waveform. Record the metrics."""
    import time
    first_token_time = None
    last_token_time = None
    token_times = []
    token_count = 0
    start_time = time.perf_counter()
    with torch.inference_mode():
        result = model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt,
            prompt_wav=str(prompt_speech),
            stream=stream,
        )
        # Assuming result is an iterable, streaming each token
        audio_chunks = []
        for i, item in enumerate(result):
            now = time.perf_counter()
            if i == 0:
                first_token_time = now
            token_times.append(now)
            token_count += 1
            last_token_time = now
            chunk = item["tts_speech"]
            if chunk is None:
                raise RuntimeError("Model output did not contain 'tts_speech'.")
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
            audio_chunks.append(chunk)
        end_time = time.perf_counter()

        if audio_chunks:
            waveform = torch.cat(audio_chunks, dim=1)
        else:
            waveform = torch.zeros(1, 0)

    total_response_time = (end_time - start_time)
    if stream:
        first_byte_latency = (first_token_time - start_time) if first_token_time else None
        token_throughput = token_count / total_response_time if total_response_time > 0 else None
        inter_token_latencies = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
        avg_inter_token_latency = sum(inter_token_latencies) / len(inter_token_latencies) if inter_token_latencies else None
        last_token_latency = (last_token_time - start_time) if last_token_time else None

    print(f"[Metrics] Total response time: {total_response_time:.4f} seconds")
    if stream:
        print(f"[Metrics] First byte latency: {first_byte_latency:.4f} seconds")
        print(f"[Metrics] Token throughput: {token_throughput:.2f} token/s")
        print(f"[Metrics] inter-token latencies: {inter_token_latencies}")
        print(f"[Metrics] Average inter-token latency: {avg_inter_token_latency:.4f} seconds")
        print(f"[Metrics] Last token latency: {last_token_latency:.4f} seconds")
    return waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CosyVoice3 TTS inference on Intel Arc B60 via torch.xpu.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("../Fun-CosyVoice3-0.5B-2512"),
        help="Path to the CosyVoice3 model directory.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text for speech synthesis.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        help="Optional speaker prompt identifier or file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../cosyvoice3_output.wav"),
        help="Destination audio file (.wav or .mp3).",
    )
    parser.add_argument(
        "--prompt-speech",
        type=Path,
        default=Path("asset/zero_shot_prompt.wav"),
        help="Optional prompt speech audio file path.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="Language code understood by CosyVoice3 (default: auto).",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Permit CPU execution if torch.xpu is unavailable.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming inference mode when supported.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional Chrome trace path to store a torch profiler capture.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of inference passes to run (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = resolve_device(strict=not args.allow_fallback)
    torch.set_default_device(device)

    # Load model once
    print(f"Loading model from {args.model_dir}...")
    model = AutoModel(model_dir=str(args.model_dir), load_vllm=True, load_trt=True, fp16=True)
    model.model.flow.eval()
    model.model.llm.eval()
    model.model.hift.eval()
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        backend = "ipex"
        #torch.xpu.memory.set_pool_size(device='xpu:0', size=16**30)  # 预分配 1GB
        model.model.flow = ipex.optimize(model.model.flow, dtype=torch.bfloat16, graph_mode=True, auto_kernel_selection=True, weights_prepack=True, level="O1")
        model.model.llm = ipex.optimize(model.model.llm, dtype=torch.bfloat16, graph_mode=True, auto_kernel_selection=True, weights_prepack=True, level="O1")
        model.model.hift = ipex.optimize(model.model.hift, graph_mode=True, auto_kernel_selection=True, weights_prepack=True, level="O1")
    except ImportError:
        backend = "inductor"
        pass

    mode = "max-autotune"
    if mode is not None:
        options = None
    else:
        options={
            "triton.cudagraphs": True,
            "epilogue_fusion": True,
            "max_autotune": True,
        }
    #import pdb
    #pdb.set_trace()
    model.model.flow = torch.compile(model.model.flow, dynamic=False, fullgraph=True, mode=mode, options=options, backend=backend)
    model.model.llm = torch.compile(model.model.llm, dynamic=False, fullgraph=True, mode=mode, options=options, backend=backend)
    model.model.hift = torch.compile(model.model.hift, dynamic=False, fullgraph=True, mode=mode, options=options, backend=backend)

    if args.prompt is None:
        raise RuntimeError(
            "No TTS prompts bundled with the model; provide --prompt explicitly."
        )

    iterations = max(1, args.iterations)
    warmup_steps = 1 if iterations > 1 else 0
    active_steps = max(1, iterations - warmup_steps)

    def invoke_inference(step_index: int):
        start_time = time.perf_counter()
        waveform = run_inference_zero_shot(
            model=model,
            text=args.text,
            prompt=args.prompt,
            prompt_speech=args.prompt_speech,
            stream=True if args.stream else False,
        )
        elapsed = time.perf_counter() - start_time
        if step_index >= warmup_steps:
            active_index = step_index - warmup_steps + 1
            print(f"Inference step {active_index}/{active_steps} took {elapsed:.3f} s")
        return waveform

    waveform = None

    if args.profile:
        profile_path = args.profile
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        cuda_activity = getattr(ProfilerActivity, "CUDA", None)
        xpu_activity = getattr(ProfilerActivity, "XPU", None)
        if device.type == "cuda" and cuda_activity is not None:
            activities.append(cuda_activity)
        if device.type == "xpu" and xpu_activity is not None:
            activities.append(xpu_activity)

        prof_schedule = schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1)

        # Capture repeated inference passes with a bounded profiler schedule.
        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            schedule=prof_schedule,
        ) as prof:
            for step_index in range(iterations):
                waveform = invoke_inference(step_index)
                prof.step()
        prof.export_chrome_trace(str(profile_path))
        print(f"Profiler trace saved to {profile_path.resolve()}")
    else:
        for step_index in range(iterations):
            waveform = invoke_inference(step_index)

    # Save the final waveform
    save_audio(args.output, waveform, model.sample_rate)
    print(f"Generated waveform saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
