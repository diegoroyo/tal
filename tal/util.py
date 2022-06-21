import os
import re
from textwrap import dedent, indent

SPEED_OF_LIGHT = 300_000_000


def local_file_path(path):
    root = os.path.split(__file__)[0]
    return os.path.join(root, path)


def fdent(text, **kwargs):
    text = dedent(text)
    vars_text = list(re.compile(r'\{(.*)\}').findall(text))
    vars_args = list(kwargs.keys())
    assert sorted(vars_text) == sorted(vars_args), \
        f'fdent error: keys do not match\n  a) {vars_text}\n  b) {vars_args}'

    if len(kwargs) == 0:
        return text

    for key in vars_text:
        indentation = re.compile(f'{{{key}}}').split(text)[0]
        try:
            indentation = indentation[indentation.rindex('\n')+1:]
        except ValueError:
            # no newline found
            pass
        lines = kwargs[key].split('\n')
        if len(lines) > 1:
            kwargs[key] = '\n'.join(
                [lines[0], indent('\n'.join(lines[1:]), indentation)])

    return text.format(**kwargs)


def tonemap_ldr(image):
    import numpy as np
    if image.shape[-1] == 4:
        # assume it's rgba
        # assert alpha channel goes to 1
        assert np.isclose(np.max(image[..., 3]), 1.0)
        # scale RGB channels to maximum 255 value
        image[..., 0:3] *= 255.0 / np.max(image[..., 0:3])
        # do not scale alpha with RGB as they are in different ranges
        image[..., 3] *= 255
        return image.astype(np.uint8)
    else:
        # assume it's a RGB or streak image
        image *= 255.0 / np.max(image)
        return image.astype(np.uint8)


def write_img(path, img):
    import imageio
    imageio.imwrite(path, img)
