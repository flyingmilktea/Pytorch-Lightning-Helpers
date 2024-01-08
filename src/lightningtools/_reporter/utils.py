from pathlib import Path

import numpy as np


def get_file_under_current_path(path, extension):
    return sorted(list(Path(path).rglob("*." + extension)))


def save_figure_to_numpy(fig, spectrogram=False):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if spectrogram:
        return data
    data = np.transpose(data, (2, 0, 1))
    return data
