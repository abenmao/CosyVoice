# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HIFI-GAN"""

from typing import Dict, Optional, List
import numpy as np
from scipy.signal import get_window
import torch
from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

# -----------------------------------------------------------------------------
# ONNX Export Symbolics (Custom Registration)
# -----------------------------------------------------------------------------

def view_as_complex_static(g, input):
    # Map to Identity (ONNX uses float[..., 2] for complex)
    return g.op("Identity", input)

def view_as_real_static(g, input):
    # Map to Identity (ONNX uses float[..., 2] for complex)
    return g.op("Identity", input)

def fft_rfft_symbolic(g, input, n=None, dim=None, norm=None):
    # Map aten::fft_rfft to ONNX DFT(inverse=0, onesided=1)
    # Result is float[..., 2] in ONNX
    if dim is None or sym_help._is_none(dim):
        dim_val = -1
    else:
        dim_val = sym_help._get_const(dim, 'i', 'dim')

    inputs = [input]
    if n is not None and not sym_help._is_none(n):
        # Ensure n is 1D tensor if it's scalar
        inputs.append(n)

    return g.op("DFT", *inputs, axis_i=dim_val, inverse_i=0, onesided_i=1)

def fft_ifft_symbolic(g, input, n=None, dim=None, norm=None):
    # Map aten::fft_ifft to ONNX DFT(inverse=1, onesided=0)
    axis = -1
    if dim is not None and not sym_help._is_none(dim):
        dim_val = sym_help._maybe_get_const(dim, 'i')
        if dim_val is not None:
            axis = dim_val
    return g.op("DFT", input, axis_i=axis, inverse_i=1, onesided_i=0)

def fft_irfft_symbolic(g, input, n=None, dim=None, norm=None):
    # Map aten::fft_irfft to ONNX DFT(inverse=1, onesided=1)
    axis = -1
    if dim is not None and not sym_help._is_none(dim):
        dim_val = sym_help._maybe_get_const(dim, 'i')
        if dim_val is not None:
            axis = dim_val

    inputs = [input]
    # n is optional length of output signal. DFT op doesn't take it directly as input usually?
    # DFT op doesn't support 'n' (signal length) for resizing behavior directly in all opsets,
    # but for irfft with onesided=1, it implies standard reconstruction.
    # We ignore n (assuming it matches) or if it's needed for padding/trimming, that's complex.
    # Usually irfft result length is derived from input or n.
    # ONNX DFT doesn't seem to take output length.

    return g.op("DFT", *inputs, axis_i=axis, inverse_i=1, onesided_i=1)

def real_symbolic(g, input):
    # Map aten::real. Input is [..., 2] (complex). Output is [..., ] (real part).
    # Slice index 0 on last axis
    sliced = g.op("Slice", input,
                  g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
                  g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)),
                  g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
    return g.op("Squeeze", sliced, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))

def imag_symbolic(g, input):
    # Map aten::imag. Input is [..., 2] (complex). Output is [..., ] (imag part).
    # Slice index 1 on last axis
    sliced = g.op("Slice", input,
                  g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)),
                  g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64)),
                  g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
    return g.op("Squeeze", sliced, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))

def conj_symbolic(g, input):
    # Map aten::conj. Input is [..., 2]. Output is [..., 2] with imag part constant-multiplied by -1.
    # Multiply by [1, -1]
    const_val = torch.tensor([1, -1], dtype=torch.float32)
    const = g.op("Constant", value_t=const_val)
    return g.op("Mul", input, const)


# Register all symbolics in one place
try:
    register_custom_op_symbolic("aten::view_as_complex", view_as_complex_static, 17)
    register_custom_op_symbolic("aten::view_as_real", view_as_real_static, 17)
    register_custom_op_symbolic("aten::fft_rfft", fft_rfft_symbolic, 17)
    register_custom_op_symbolic("aten::fft_ifft", fft_ifft_symbolic, 17)
    register_custom_op_symbolic("aten::fft_irfft", fft_irfft_symbolic, 17)
    register_custom_op_symbolic("aten::real", real_symbolic, 17)
    register_custom_op_symbolic("aten::imag", imag_symbolic, 17)
    register_custom_op_symbolic("aten::_conj", conj_symbolic, 17)
except Exception as e:
    print(f"Failed to register custom symbolic: {e}")

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn.utils import remove_weight_norm
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm
from torch.distributions.uniform import Uniform
from cosyvoice.transformer.convolution import CausalConv1d, CausalConv1dDownSample, CausalConv1dUpsample
from cosyvoice.transformer.activation import Snake
from cosyvoice.utils.common import get_padding
from cosyvoice.utils.common import init_weights
import types


"""hifigan based generator implementation.

This code is modified from https://github.com/jik876/hifi-gan
 ,https://github.com/kan-bayashi/ParallelWaveGAN and
 https://github.com/NVIDIA/BigVGAN

"""


class ResBlock(torch.nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""
    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
        causal: bool = False,
    ):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation)) if causal is False else
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        causal_type='left'
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1)) if causal is False else
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        causal_type='left'
                    )
                )
            )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList([
            Snake(channels, alpha_logscale=False)
            for _ in range(len(self.convs1))
        ])
        self.activations2 = nn.ModuleList([
            Snake(channels, alpha_logscale=False)
            for _ in range(len(self.convs2))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            remove_weight_norm(self.convs2[idx])


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    @torch.no_grad()
    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, dim=1, length)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.transpose(1, 2)
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1))).to(f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i: i + 1, :] = f0 * (i + 1) / self.sampling_rate

        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1)).to(F_mat.device)
        phase_vec[:, 0, :] = 0

        # generate sine waveforms
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves.transpose(1, 2), uv.transpose(1, 2), noise


class SineGen2(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False,
                 causal=False):
        super(SineGen2, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale
        self.causal = causal
        if causal is True:
            self.rand_ini = torch.rand(1, 9)
            self.rand_ini[:, 0] = 0
            self.sine_waves = torch.rand(1, 300 * 24000, 9)

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        if self.training is False and self.causal is True:
            rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)
        else:
            rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2),
                                                         scale_factor=1 / self.upsample_scale,
                                                         mode="linear").transpose(1, 2)

            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale,
                                                    scale_factor=self.upsample_scale, mode="nearest" if self.causal is True else 'linear').transpose(1, 2)
            sines = torch.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        if self.training is False and self.causal is True:
            noise = noise_amp * self.sine_waves[:, :sine_waves.shape[1]].to(sine_waves.device)
        else:
            noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0, sinegen_type='1', causal=False):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        if sinegen_type == '1':
            self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        else:
            self.l_sin_gen = SineGen2(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod, causal=causal)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()
        self.causal = causal
        if causal is True:
            self.uv = torch.rand(1, 300 * 24000, 1)

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        if self.training is False and self.causal is True:
            noise = self.uv[:, :uv.shape[1]] * self.sine_amp / 3
        else:
            noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv

    def remove_weight_norm(self):
        pass

class HiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            nb_harmonics: int = 8,
            sampling_rate: int = 22050,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: List[int] = [8, 8],
            upsample_kernel_sizes: List[int] = [16, 16],
            istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes: List[int] = [3, 7, 11],
            resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes: List[int] = [7, 11],
            source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
            lrelu_slope: float = 0.1,
            audio_limit: float = 0.99,
            f0_predictor: torch.nn.Module = None,
    ):
        super(HiFTGenerator, self).__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # NOTE in CosyVoice2, we use the original SineGen implementation
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
            sinegen_type='1' if self.sampling_rate == 22050 else '2',
            causal=False)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(
            Conv1d(in_channels, base_channels, 7, 1, padding=3)
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(
                    Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1)
                )
            else:
                self.source_downs.append(
                    Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u, padding=(u // 2))
                )

            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d)
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, istft_params["n_fft"] + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.f0_predictor = f0_predictor

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        self.m_source.remove_weight_norm()
        for l in self.source_downs:
            remove_weight_norm(l)
        for l in self.source_resblocks:
            l.remove_weight_norm()

    def _onnx_stft(self, x):
        # x: [B, T]
        # Manual STFT implementation using Conv1d + FFT to avoid ONNX STFT op dynamic rank issues.

        if x.dim() == 3:
            x = x.squeeze(1)

        n_fft = self.istft_params["n_fft"]
        hop_length = self.istft_params["hop_len"]

        # 1. Pad signal (reflect)
        # center=True behavior: pad n_fft // 2 on both sides
        pad = n_fft // 2
        x = x.unsqueeze(1) # [B, 1, T]
        x = torch.nn.functional.pad(x, (pad, pad), mode='reflect')

        # 2. Frame extraction using Conv1d
        # We use a Conv1d to extract frames of size n_fft with stride hop_length.
        # Input: [B, 1, T_padded]
        # Weight: Identity matrix viewed as [n_fft, 1, n_fft].
        # This maps each position in the window to an output channel.
        # frame[b, k, t] = x[b, 0, t*stride + k]
        eye = torch.eye(n_fft, device=x.device, dtype=x.dtype)
        weights = eye.view(n_fft, 1, n_fft)

        # [B, n_fft, n_frames]
        frames = torch.nn.functional.conv1d(x, weights, stride=hop_length)

        # 3. Apply Window
        # window shape [n_fft] -> [1, n_fft, 1]
        w = self.stft_window.to(x.device).view(1, n_fft, 1)
        frames = frames * w

        # 4. FFT
        # Perform FFT using Matrix Multiplication to ensure ONNX compatibility and shape correctness.
        # This avoids using the atomic DFT op which can have shape inference issues.

        n_freqs = n_fft // 2 + 1

        # specific implementation of rfft using matmul
        # W[k, n] = exp(-2j * pi * k * n / N)
        # Real part: cos(...), Imag part: sin(...)

        # Create DFT matrix on the fly (or cache it)
        # We rely on ONNX constant folding for the meshgrid/cos/sin parts.

        k = torch.arange(n_freqs, device=x.device).unsqueeze(1) # [F, 1]
        n = torch.arange(n_fft, device=x.device).unsqueeze(0)   # [1, N]
        theta = -2 * np.pi * k * n / n_fft

        mat_real = torch.cos(theta).to(dtype=x.dtype) # [F, N]
        mat_imag = torch.sin(theta).to(dtype=x.dtype) # [F, N]

        # frames: [B, N, T]
        # output: [B, F, T]
        spec_real = torch.matmul(mat_real, frames)
        spec_imag = torch.matmul(mat_imag, frames)

        # Return Real and Imag parts [B, F, T]
        return spec_real, spec_imag

    def _onnx_istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)

        # Manual decomposition of ISTFT to ensure clean ONNX export:
        # Replaced with Manual IDFT using MatMul

        n_fft = self.istft_params["n_fft"]
        hop_length = self.istft_params["hop_len"]
        device = magnitude.device

        # 2. Inverse FFT
        # x[n] = 1/N * sum(X[k] * exp(j*2*pi*k*n/N))

        n_freqs = n_fft // 2 + 1
        k = torch.arange(n_freqs, device=device).unsqueeze(1) # [F, 1]
        n = torch.arange(n_fft, device=device).unsqueeze(0)   # [1, N]
        theta = 2 * np.pi * k * n / n_fft

        # Scaling
        scale = torch.ones(n_freqs, 1, device=device) / n_fft
        if n_fft % 2 == 0:
            scale[1:-1] *= 2
        else:
            scale[1:] *= 2

        # IDFT Matrix [N, F]
        # Transpose theta to [N, F]
        mat_real = (torch.cos(theta).T * scale.T).to(dtype=magnitude.dtype)
        mat_imag = (torch.sin(theta).T * scale.T).to(dtype=magnitude.dtype)

        # MatMul: (N, F) @ (B, F, T) -> (B, N, T)
        x_real = torch.matmul(mat_real, real)
        x_imag = torch.matmul(mat_imag, img)
        x = x_real - x_imag


        # 3. Apply Window
        window = self.stft_window.to(device)

        window = window.view(1, n_fft, 1)
        x = x * window

        # 4. Overlap-Add (Signal) using ConvTranspose1d
        # We view the frames as independent channels that need to be summed at offsets.
        # Weight [In, Out, K] = [n_fft, 1, n_fft]. Identity matrix.
        # This maps each sample in the frame to its correct position in the output window.
        ola_weights = torch.eye(n_fft, device=device).view(n_fft, 1, n_fft)

        # stride=hop_length places the next frame at +hop_length
        signal = F.conv_transpose1d(x, ola_weights, stride=hop_length, padding=0)

        # 5. Overlap-Add (Envelope) for NOLA normalization
        # We need to divide by the sum of squared windows.
        window_sq = window.pow(2)
        # Create a "ones" signal with the window shape to sum up window weights
        # Expanding window_sq to [B, n_fft, T] efficiently
        batch_size, _, n_frames = x.shape
        window_sq_frames = window_sq.expand(batch_size, -1, n_frames)

        envelope = F.conv_transpose1d(window_sq_frames, ola_weights, stride=hop_length, padding=0)

        # 6. Normalize
        signal = signal / (envelope + 1e-11)

        # 7. Trim padding (center=True behavior)
        # center=True pads n_fft // 2 at both ends during STFT.
        # We must remove this padding.
        pad_amt = n_fft // 2
        signal = signal[:, :, pad_amt:-pad_amt]

        return signal.squeeze(1)

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window.to(x.device),
            return_complex=True)
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(torch.complex(real, img), self.istft_params["n_fft"], self.istft_params["hop_len"],
                                        self.istft_params["n_fft"], window=self.stft_window.to(magnitude.device))
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        speech_feat = batch['speech_feat'].transpose(1, 2).to(device)
        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        # mel+source->speech
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, f0

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, cache_source: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        # use cache_source to avoid glitch
        if cache_source.shape[2] != 0:
            s[:, :, :cache_source.shape[2]] = cache_source
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s


class CausalHiFTGenerator(HiFTGenerator):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            nb_harmonics: int = 8,
            sampling_rate: int = 22050,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: List[int] = [8, 8],
            upsample_kernel_sizes: List[int] = [16, 16],
            istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes: List[int] = [3, 7, 11],
            resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes: List[int] = [7, 11],
            source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
            lrelu_slope: float = 0.1,
            audio_limit: float = 0.99,
            conv_pre_look_right: int = 4,
            f0_predictor: torch.nn.Module = None,
    ):
        torch.nn.Module.__init__(self)

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
            sinegen_type='1' if self.sampling_rate == 22050 else '2',
            causal=True)
        self.upsample_rates = upsample_rates
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(
            CausalConv1d(in_channels, base_channels, conv_pre_look_right + 1, 1, causal_type='right')
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    CausalConv1dUpsample(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u,
                    )
                )
            )

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(
                    CausalConv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1, causal_type='left')
                )
            else:
                self.source_downs.append(
                    CausalConv1dDownSample(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u)
                )

            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d, causal=True)
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, causal=True))

        self.conv_post = weight_norm(CausalConv1d(ch, istft_params["n_fft"] + 2, 7, 1, causal_type='left'))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.conv_pre_look_right = conv_pre_look_right
        self.f0_predictor = f0_predictor

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0), finalize: bool = True) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        if finalize is True:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(x[:, :, :-self.conv_pre_look_right], x[:, :, -self.conv_pre_look_right:])
            s_stft_real = s_stft_real[:, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
            s_stft_imag = s_stft_imag[:, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)

            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        if finalize is False:
            x = x[:, :-int(np.prod(self.upsample_rates) * self.istft_params['hop_len'])]
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        # mel->f0 NOTE f0_predictor precision is crucial for causal inference, move self.f0_predictor to cpu if necessary
        self.f0_predictor.to('cpu')
        f0 = self.f0_predictor(speech_feat.cpu(), finalize=finalize).to(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        if hasattr(self, 'stream_model') and hasattr(self, 'final_model'):
            if finalize is True:
                ort_inputs = {'speech_feat': speech_feat.cpu().numpy(), 'cache_source': s.cpu().numpy()}
                ort_outputs = self.final_model.output_names
                [estimator, stream], _ = self.final_model.acquire_estimator()
                outputs = estimator.run(ort_outputs, ort_inputs)
                self.final_model.release_estimator(estimator, stream)
            else:
                ort_inputs = {'speech_feat': speech_feat[:, :, :-self.f0_predictor.condnet[0].causal_padding].cpu().numpy(), 'cache_source': s.cpu().numpy()}
                ort_outputs = self.stream_model.output_names
                [estimator, stream], _ = self.stream_model.acquire_estimator()
                outputs = estimator.run(ort_outputs, ort_inputs)
                self.stream_model.release_estimator(estimator, stream)
            generated_speech = torch.from_numpy(outputs[0]).to(speech_feat.device)
            return generated_speech, s

        if finalize is True:
            generated_speech = self.decode(x=speech_feat, s=s, finalize=finalize)
        else:
            generated_speech = self.decode(x=speech_feat[:, :, :-self.f0_predictor.condnet[0].causal_padding], s=s, finalize=finalize)
        return generated_speech, s

    def export_onnx(self, onnx_model_path: str, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('xpu'), finalize: bool = True) -> None:
        """ Export the generator to ONNX format

        Args:
            onnx_model_path (str): Path to save the ONNX model.
            device (torch.device): Device to run the export on. Default to cpu to avoid potential segfaults with STFT on cuda.
            finalize (bool): Whether to export finalize=True (full inference) or False (streaming). Default True.
        """
        self = self.to(device)
        self.eval()

        ## reset _stft and _istft with ONNX compatible versions
        self._stft = self._onnx_stft
        self._istft = self._onnx_istft

        # Create a wrapper that fixes the finalize parameter and implements forward logic
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model, finalize_value):
                super().__init__()
                self.model = model
                self.finalize_value = finalize_value

            def forward(self, speech_feat, s):
                s = s.to(speech_feat.dtype)
                x = self.model.decode(x=speech_feat, s=s, finalize=self.finalize_value)
                return x

        wrapper = ONNXWrapper(self, finalize)
        speech_feat = torch.randn(1, 80, 300, device=device, dtype=torch.float32)
        s = torch.randn(1, 1, speech_feat.shape[2] * 480, device=device, dtype=torch.float32)

        torch.onnx.export(
            wrapper,
            (speech_feat, s),
            onnx_model_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
            input_names=['speech_feat', 'cache_source'],
            output_names=['waveform'],
            dynamic_axes={
                'speech_feat': {0: 'batch', 2: 'time_feat'},
                'cache_source': {0: 'batch', 2: 'time_source'},
                'waveform': {1: 'time_source'},
            },
            dynamo=False,
        )
        print("Export complete.")

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from hyperpyyaml import load_hyperpyyaml
    with open('./pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})
    model = configs['hift']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    max_len, chunk_size, context_size = 300, 30, 8
    mel = torch.rand(1, 80, max_len).to(device)
    pred_gt, _ = model.inference(mel)
    for i in range(0, max_len, chunk_size):
        finalize = True if i + chunk_size + context_size >= max_len else False
        pred_chunk, _ = model.inference(mel[:, :, : i + chunk_size + context_size], finalize=finalize)
        pred_chunk = pred_chunk[:, i * 480:]
        print((pred_gt[:, i * 480:i * 480 + pred_chunk.shape[1]] - pred_chunk).abs().max().item())
