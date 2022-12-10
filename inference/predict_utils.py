from typing import Tuple, Union, List, Dict
from datetime import datetime, timedelta
import requests
import numpy as np
import cv2
import albumentations as A
import torch
from preprocessing import TextPreprocessor
from networks import CLIK
from utils.data_utils import txt_input_to_device


class Predictor:
    def __init__(
        self,
        model: CLIK,
        img_transforms: A.Compose,
        txt_preprocessor: TextPreprocessor = None,
        device: torch.device = None,
    ):
        self.model = model
        self.img_transforms = img_transforms
        self.txt_preprocessor = txt_preprocessor
        self.device = device if device is not None else torch.device("cpu")

        self.model.to(self.device)
        if self.model.training:
            self.model.eval()
        assert (
            not self.model.training
        ), "model is on train mode. please activate '.eval()'"

    def __call__(self, plan_attrs, prod_paths, topk: int = 1):
        return self.infer(plan_attrs, prod_paths, topk)

    def infer(
        self,
        topic: Dict[str, str],
        images: Union[List[int], List[str], List[np.ndarray]],
        topk: int = 1,
    ) -> Tuple[int, str]:
        batch = self.collate(topic, images)
        logits = self.model.predict(batch)
        _, indices = torch.topk(logits.squeeze(), k=topk)
        output = [images[i] for i in indices.tolist()]
        return output

    def collate(self, topic: Dict[str, str], images: Union[List[str], List[np.ndarray]]):
        if not isinstance(images, list):
            images = [images]

        # collate batch
        if topic.get("date", None) is None:
            topic["date"] = self.today()

        topic_preprocessed = txt_input_to_device(self.txt_preprocessor(**topic, expand_dim=True), device=self.device)
        images_preprocessed = self.read_transform_images(images).to(self.device)
        batch = dict(topics=topic_preprocessed, images=images_preprocessed)
        return batch

    def read_transform_images(self, images: Union[List[str], List[np.ndarray]]) -> torch.Tensor:
        if isinstance(images[0], str):
            images = []
            for path in images:
                if path.startswith("http"):  # load by request
                    img_arr = np.array(
                        bytearray(requests.get(path).content), dtype=np.uint8
                    )
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                else:  # load from local
                    img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(self.img_transforms(image=img)["image"])

        elif isinstance(images[0], np.ndarray):
            images = [self.img_transforms(image=img)["image"] for img in images]

        images = torch.stack(images).to(self.device)
        return images

    @staticmethod
    def today() -> str:
        t = datetime.today() + timedelta(hours=9)  # korean time
        t = t.strftime("%Y-%m-%d")
        return t
