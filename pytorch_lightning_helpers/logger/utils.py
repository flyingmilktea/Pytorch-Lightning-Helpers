from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from matplotlib import pyplot as plt
from scipy.io.wavfile import read

matplotlib.use("Agg")
mel_basis = {}
hann_window = {}

def get_file_under_current_path(path, extension):
    return sorted(list(Path(path).rglob("*." + extension)))


def load_wav(path):
    sr, wav = read(path)

    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    if sr != 16000:
        raise ValueError(
            "{} SR doesn't match target {} SR, plz resample the audio files".format(
                sr, 16000
            )
        )
    if np.max(wav) > 1.0 or np.min(wav) < -1.0:
        raise ValueError("audio is not normalize from -1 to 1")
    return torch.FloatTensor(wav).unsqueeze(1)


def save_figure_to_numpy(fig, spectrogram=False):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if spectrogram:
        return data
    data = np.transpose(data, (2, 0, 1))
    return data


def wandb_media_log(info_dict):
    loggerdict = {}

    for k, v in info_dict.items():
        if "_audio" in k:
            loggerdict[f"train_audio/{k}"] = wandb.Audio(
                np.float32(v.detach().cpu().numpy().flatten()), sample_rate=16000
            )

        elif "_graph" in k:
            concated = torch.cat([i for i in v], dim=1)
            loggerdict[f"train_graph/{k}"] = wandb.Image(
                plot_spectrogram_to_numpy(np.float32(concated.detach().cpu().numpy()))
            )

        elif "_distribution" in k:
            loggerdict[f"train_distribution/{k}"] = wandb.Image(
                real_fake_distribution(*v)
            )

    return loggerdict
