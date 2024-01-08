from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from scipy.io.wavfile import read


def get_file_under_current_path(path, extension):
    return sorted(list(Path(path).rglob("*." + extension)))


def load_wav(path):
    sr, wav = read(path)

    if sr != 16000:
        raise ValueError(
            f"{sr} SR doesn't match target 16000 SR, please resample the audio files"
        )

    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    if np.max(wav) > 1.0 or np.min(wav) < -1.0:
        raise ValueError("audio is not normalize from -1 to 1")
    return torch.FloatTensor(wav).unsqueeze(1)
