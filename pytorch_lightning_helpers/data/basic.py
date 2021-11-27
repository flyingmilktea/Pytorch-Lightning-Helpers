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


def get_speaker_id_path(wav_path):
    return Path(*list(Path(wav_path).parts[:-2]))


def get_file_under_current_path(path, extension):
    return sorted(list(Path(path).rglob("*." + extension)))


def get_files_under_speaker_id(path, extension):
    id_path = get_speaker_id_path(path)
    return get_file_under_current_path(id_path, extension)


def load_mel(path):
    return torch.Tensor(np.load(path)).transpose(0, 1)


def load_concat_embs(ref_list, tgt_mel=None):
    embs = [load_mel(i).squeeze().cpu() for i in ref_list]
    if tgt_mel != None:
        embs.append(tgt_mel)
    return torch.cat(embs)


def load_w2v(path):
    vec = torch.load(path, "cpu")
    vec.requires_grad = False
    return vec


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


def mel_spectrogram(
    y,
    n_fft=1024,
    num_mels=256,
    sampling_rate=16000,
    hop_size=320,
    win_size=1024,
    fmin=0,
    fmax=8000,
    center=False,
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def plot_spectrogram_to_numpy(spectrogram):
    # spectrogram.shape = (256, x)
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig, True)
    plt.close()
    return data


def save_figure_to_numpy(fig, spectrogram=False):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if spectrogram:
        return data
    data = np.transpose(data, (2, 0, 1))
    return data


def real_fake_distribution(real_miscs=None, fake_miscs=None, log=False):
    fig, ax = plt.subplots(1, len(real_miscs), figsize=(16, 4), dpi=75)
    for i, (real_misc, fake_misc) in enumerate(zip(real_miscs, fake_miscs)):
        ax[i].hist(
            real_misc.detach().cpu().numpy().flatten(),
            bins=20,
            density=True,
            log=log,
            alpha=0.3,
            color="skyblue",
            label="real",
        )
        ax[i].hist(
            fake_misc.detach().cpu().numpy().flatten(),
            bins=20,
            density=True,
            log=log,
            alpha=0.3,
            color="red",
            label="fake",
        )
    plt.legend()
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig, True)
    plt.close()
    return data
