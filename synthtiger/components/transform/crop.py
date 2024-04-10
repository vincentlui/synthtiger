"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component
import albumentations as A


class Crop(Component):
    def __init__(self, pxs=None, percents=None, direction='right'):
        super().__init__()
        self.pxs = pxs
        self.percents = percents
        self.direction = direction

        shapes = [(1, 2), (2, 2)]
        if self.pxs is not None and np.array(self.pxs).shape not in shapes:
            raise TypeError("Shape of pxs must be (1,2) or (2,2)")
        if self.percents is not None and np.array(self.percents).shape not in shapes:
            raise TypeError("Shape of percents must be (1,2) or (2,2)")
        if not direction in ['left', 'right']:
            raise TypeError("Direction must be 'left' or 'right'")

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        pxs = meta.get(
            "pxs",
            tuple(np.random.randint(px[0], px[1] + 1) for px in self.pxs)
            if self.pxs is not None
            else None,
        )
        percents = meta.get(
            "percents",
            tuple(
                np.random.uniform(percent[0], percent[1]) for percent in self.percents
            )
            if self.percents is not None
            else None,
        )

        meta = {
            "pxs": pxs,
            "percents": percents,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        pxs = meta["pxs"]
        percents = meta["percents"]

        if pxs is not None:
            pxs = np.tile(pxs, 2)[:2]
        if percents is not None:
            percents = np.tile(percents, 2)[:2]

        for layer in layers:
            image = layer.output()
            height, width = image.shape[:2]

            crops = [0, 0, 0, 0]
            if pxs is not None:
                crops = pxs
            elif percents is not None:
                crops = percents * (width, height)

            if self.direction == 'right':
                crops = np.amin((crops, (width - 1, height - 1)), axis=0)
                crops = np.amax((crops, (0, 0)), axis=0)
                right, bottom = crops 
                right, bottom = round(right), round(bottom)

                image = utils.crop_image(
                    image, top=0, right=right, bottom=bottom, left=0
                )
                topleft = layer.topleft 
            else:
                crops = (width, height) - crops
                crops = np.amin((crops, (width - 1, height - 1)), axis=0)
                crops = np.amax((crops, (0, 0)), axis=0)
                left, top = crops 
                left, top = round(left), round(top)

                image = utils.crop_image(
                    image, top=top, right=width, bottom=height, left=left
                )
                topleft = [top, left]

            topleft = layer.topleft 
            height, width = image.shape[:2]
            layer.image = image
            layer.bbox = [*topleft, width, height]

        return meta
