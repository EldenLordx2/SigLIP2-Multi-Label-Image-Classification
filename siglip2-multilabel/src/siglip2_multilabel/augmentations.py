import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def cutout_pil(image: Image.Image, pad_size: int, replace=0) -> Image.Image:
    image_np = np.array(image)
    image_height, image_width = image_np.shape[:2]

    cutout_center_height = np.random.randint(0, image_height + 1)
    cutout_center_width = np.random.randint(0, image_width + 1)

    lower_pad = max(0, cutout_center_height - pad_size)
    upper_pad = max(0, image_height - cutout_center_height - pad_size)
    left_pad = max(0, cutout_center_width - pad_size)
    right_pad = max(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]

    mask = np.pad(np.zeros(cutout_shape, dtype=image_np.dtype), padding_dims, constant_values=1)
    mask = np.expand_dims(mask, axis=-1)
    if image_np.ndim == 3:
        mask = np.tile(mask, [1, 1, image_np.shape[2]])

    image_np = np.where(
        np.equal(mask, 0),
        np.full_like(image_np, fill_value=replace, dtype=image_np.dtype),
        image_np,
    )
    return Image.fromarray(image_np)


class SubPolicyV2:
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int32),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
            "cutout": np.round(np.linspace(0, 20, 10), 0).astype(np.int32),
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, fillcolor + (255,)), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0), Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0), Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0), fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])), fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, int(magnitude)),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, _: ImageOps.autocontrast(img),
            "equalize": lambda img, _: ImageOps.equalize(img),
            "invert": lambda img, _: ImageOps.invert(img),
            "cutout": lambda img, magnitude: cutout_pil(img, int(magnitude)),
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class RandAugmentV4:
    def __init__(self):
        self._policies = self.get_rand_policies()

    @classmethod
    def get_trans_list(cls):
        return [
            "shearX", "shearY", "translateX", "translateY", "rotate",
            "color", "posterize", "solarize", "contrast", "sharpness",
            "brightness", "autocontrast", "equalize", "invert", "cutout"
        ]

    @classmethod
    def get_rand_policies(cls):
        op_list = []
        for trans in cls.get_trans_list():
            for magnitude in range(1, 10):
                op_list.append((0.5, trans, magnitude))
        return [[op1, op2] for op1 in op_list for op2 in op_list]

    def __call__(self, img: Image.Image) -> Image.Image:
        chosen_policy = self._policies[random.randint(0, len(self._policies) - 1)]
        policy = SubPolicyV2(*chosen_policy[0], *chosen_policy[1])
        return policy(img)


class TrainImageTransform:
    def __init__(self, image_size=448, cutout_length=224, use_randaugment=True):
        self.image_size = int(image_size)
        self.cutout_length = int(cutout_length)
        self.randaugment = RandAugmentV4() if use_randaugment else None

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        if self.cutout_length > 0:
            image = cutout_pil(image, pad_size=self.cutout_length // 2, replace=0)
        if self.randaugment is not None:
            image = self.randaugment(image)
        return image


class ValImageTransform:
    def __init__(self, image_size=448):
        self.image_size = int(image_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        return image
