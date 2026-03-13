"""
Simple PyTorch Yamnet inference example

This script demonstrates how to:
- load the PyTorch Yamnet model and `yamnet.pth` state dict
- load a WAV file, compute log-mel spectrogram patches compatible
  with the Yamnet model, and
- run the model to get predictions and embeddings.

Usage:
  pip install torch librosa numpy soundfile
  python yamnet/example_pytorch_inference.py /path/to/audio.wav

Notes:
- This is a lightweight example intended for feature extraction.
- It uses Yamnet-style parameters: sr=16000, n_mels=64, window=25ms, hop=10ms,
  patch_frames=96 (approximately 0.96s per patch).
"""
import sys
import os
from typing import Tuple
import time

import numpy as np
import torch
import librosa

from torch_audioset.yamnet.model import yamnet as torch_yamnet


def waveform_to_log_mel_patches(waveform: np.ndarray,
                                sample_rate: int = 16000,
                                n_mels: int = 64,
                                window_seconds: float = 0.025,
                                hop_seconds: float = 0.01,
                                patch_frames: int = 96,
                                fmin: float = 0.0,
                                fmax: float = 8000.0) -> np.ndarray:
    """
    Convert 1D or 2D waveform to a stack of log-mel patches.
    Input: (length,) for single audio or (batch_size, length) for batch.
    Output: (N, patch_frames, n_mels) for single, or (batch_size, N, patch_frames, n_mels) for batch.
    This follows Yamnet typical settings: 25 ms window, 10 ms hop, 64 mel bins, 96 frames per patch.
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
        return_single = True
    else:
        return_single = False

    batch_size = waveform.shape[0]
    all_patches = []

    for b in range(batch_size):
        wf = waveform[b]
        # librosa parameters
        n_fft = int(round(window_seconds * sample_rate))
        hop_length = int(round(hop_seconds * sample_rate))

        # power spectrogram -> mel
        mel_spec = librosa.feature.melspectrogram(
            y=wf,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0,
        )

        # convert to log mel (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        # transpose to shape (frames, n_mels)
        frames = log_mel.T

        # build patches (non-overlapping here for simplicity)
        n_frames = frames.shape[0]
        if n_frames < patch_frames:
            # pad with minimal value (same as librosa.power_to_db floor)
            pad_amount = patch_frames - n_frames
            pad = np.full((pad_amount, n_mels), fill_value=frames.min())
            frames = np.vstack([frames, pad])
            n_frames = frames.shape[0]

        # Number of non-overlapping patches
        n_patches = n_frames // patch_frames
        patches = []
        for i in range(n_patches):
            s = i * patch_frames
            patch = frames[s:s + patch_frames]
            patches.append(patch)

        if patches:  # only stack if there are patches
            patches = np.stack(patches, axis=0).astype(np.float32)
        else:
            # if no patches, create a zero patch (edge case)
            patches = np.zeros((1, patch_frames, n_mels), dtype=np.float32)
        all_patches.append(patches)

    result = np.stack(all_patches, axis=0)
    if return_single:
        result = result[0]
    return result


def load_model(pth_path: str):
    model = torch_yamnet(pretrained=False)
    state = torch.load(pth_path)
    model.load_state_dict(state)
    model.eval()
    return model


def wav_path_run_inference(wav_path: str, pth_path: str, out_prefix: str = 'yamnet_out') -> Tuple[np.ndarray, np.ndarray]:
    # load audio at 16k
    waveform, sr = librosa.load(wav_path, sr=16000, mono=True)
    print('Loaded waveform shape:', waveform.shape, 'Sample rate:', sr)
    patches = waveform_to_log_mel_patches(waveform, sample_rate=sr)

    # model expects shape [N, 1, patch_frames, n_mels]
    x = torch.from_numpy(patches).unsqueeze(1)  # [N, 1, 96, 64]
    print('log_mel_patches shape:', x.shape)

    model = load_model(pth_path)

    with torch.no_grad():
        # `to_prob=True` returns probabilities in convert_yamnet usage; if not available,
        # model may return logits—adjust accordingly.
        try:
            preds = model(x, to_prob=True)
        except TypeError:
            preds = model(x)

        preds = preds.cpu().numpy()

    # Save outputs
    np.save(f'{out_prefix}_predictions.npy', preds)
    print('Predictions saved to', f'{out_prefix}_predictions.npy')
    print('Predictions shape:', preds.shape)
    return preds


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python yamnet/example_pytorch_inference.py /path/to/audio.wav')
        sys.exit(1)

    wav_path = sys.argv[1]
    # default pth in repo root 'yamnet.pth'
    pth_path = os.path.join(os.path.dirname(__file__), 'yamnet.pth')
    pth_path = os.path.normpath(pth_path)
    if not os.path.exists(pth_path):
        print('Could not find yamnet.pth at', pth_path)
        print('Place the converted `yamnet.pth` next to the repo root or change `pth_path`.')
        sys.exit(1)

    start_time = time.time()
    wav_path_run_inference(wav_path, pth_path)
    end_time = time.time()
    print(f"processing time: {end_time - start_time:.2f} seconds")

