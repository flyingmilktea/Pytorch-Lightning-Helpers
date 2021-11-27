#!/usr/bin/env python3


def scale_loss(loss_func, scale):
    return lambda **kwargs: {k: v * scale for (k, v) in loss_func(**kwargs).items()}
