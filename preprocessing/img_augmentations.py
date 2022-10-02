import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(h: int = 224, w: int = 224) -> A.Compose:
    transforms = A.Compose(
        [
            A.Resize(int(h * 1.15), int(h * 1.15)),
            A.RandomCrop(h, w),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return transforms


def get_eval_transforms(h: int = 224, w: int = 224) -> A.Compose:
    transforms = A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return transforms
