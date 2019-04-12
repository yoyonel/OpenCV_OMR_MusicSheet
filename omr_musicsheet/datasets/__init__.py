"""
"""
from pathlib import Path

import pkg_resources


def get_datasets_path() -> Path:
    return Path(
        pkg_resources.resource_filename(
            'omr_musicsheet',
            'datasets/data/')
    )


def get_image_path(img_fn) -> Path:
    img_path = get_datasets_path() / img_fn
    if not img_path.exists():
        msg_err = f"{img_path} does'nt exist !"
        raise IOError(msg_err)
    return img_path


__all__ = ['get_datasets_path', 'get_image_path']
