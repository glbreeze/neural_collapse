
import math
import random
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=1.0, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        sigma = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def get_moco_base_augmentation(min_scale=0.2, normalize=None, size=32):
    normalize = normalize if normalize is not None else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return [
        transforms.RandomResizedCrop(size, scale=(min_scale, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), # not strengthened
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(radius_min=0.1, radius_max=2.0)], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]