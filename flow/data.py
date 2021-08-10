from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as f
import PIL
from torch.utils.data import Dataset


class CelebA(Dataset):
    '''
    CelebA PyTorch dataset

    The built-in PyTorch dataset for CelebA is outdated.
    '''
    base_folder = 'celeba'

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None,):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        celeb_path = lambda x: self.root / self.base_folder / x

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
        }
        splits_df = pd.read_csv(celeb_path('list_eval_partition.csv'))
        self.filename = splits_df[splits_df['partition'] == split_map[split]]['image_id'].tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = (self.root / self.base_folder / 'img_align_celeba' /
                    'img_align_celeba' / self.filename[index])
        X = PIL.Image.open(img_path)

        target: Any = []
        if self.transform is not None:
            X = self.transform(X)

        return X, 0

    def __len__(self) -> int:
        return len(self.filename)
