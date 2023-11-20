import random
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import CIFAR10

class TriggerHandler(object):
    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img

class CIFAR10Poison(CIFAR10):
    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.width, self.height, self.channels = self.__shape_info__()

        # poisoned image objects
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        # set poisoning rate
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        # print(f"poisoning_rate now is: {self.poisoning_rate}")
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        # convert a NumPy array to a PIL Image
        img = Image.fromarray(img)

        # change target and paste trigger for poisoning img
        for index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
