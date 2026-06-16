from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (64, 64)


def preprocess_image(image: Image.Image, image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, *, flatten: bool = True) -> np.ndarray:
    """Resize and normalise a PIL image to a NumPy array."""

    resized = image.convert('RGB').resize(image_size)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    if flatten:
        arr = arr.reshape(-1)
    return arr


class SatelliteDataset:
    SPLIT_ALIASES = {
        'train': ['train', 'train data', 'train_data', 'training'],
        'val': ['val', 'valid', 'validation', 'val data', 'validation data', 'valid data', 'val_data'],
        'test': ['test', 'test data', 'test_data', 'testing'],
    }

    def __init__(
        self,
        root_dir: str,
        metadata_csv: str | None = None,
        transform: Callable[[Image.Image], np.ndarray] | None = None,
        split: str | None = None,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.root = Path(root_dir).expanduser().resolve()
        self.split = split.lower() if split else None
        self.image_size = image_size
        self.transform = transform or (lambda img: preprocess_image(img, image_size=self.image_size, flatten=True))

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.data_root = self._locate_data_root()

        if metadata_csv and Path(metadata_csv).exists():
            df = pd.read_csv(metadata_csv)
            base_dir = self.data_root
            self.items: List[Tuple[Path, str]] = []
            for _, row in df.iterrows():
                img_path = base_dir / row['filename']
                if not img_path.exists():
                    raise FileNotFoundError(f"Image '{row['filename']}' listed in metadata not found under {base_dir}")
                self.items.append((img_path, row['label']))
        else:
            self.items = self._gather_items(self.data_root)

        if not self.items:
            split_label = self.split or 'all'
            raise RuntimeError(f"No image files found under '{split_label}' directory {self.data_root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        path, label = self.items[idx]
        with Image.open(path) as img:
            data = self.transform(img)
        return data, label

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        features: List[np.ndarray] = []
        labels: List[str] = []
        for path, label in self.items:
            with Image.open(path) as img:
                features.append(self.transform(img))
            labels.append(label)
        return np.stack(features), np.asarray(labels)

    def iter_samples(self) -> Iterable[Tuple[np.ndarray, str]]:
        for path, label in self.items:
            with Image.open(path) as img:
                yield self.transform(img), label

    def _locate_data_root(self) -> Path:
        if not self.split:
            return self.root

        aliases = self._split_aliases(self.split)
        search_roots = [self.root]
        try:
            search_roots.extend([p for p in self.root.iterdir() if p.is_dir()])
        except PermissionError:
            pass

        for base in search_roots:
            match = self._match_split_dir(base, aliases)
            if match is not None:
                return match

        raise FileNotFoundError(
            f"Could not locate split '{self.split}' under {self.root}. Checked aliases: {aliases}"
        )

    def _match_split_dir(self, base: Path, aliases: List[str]) -> Path | None:
        try:
            candidates = [p for p in base.iterdir() if p.is_dir()]
        except PermissionError:
            return None

        alias_norms = {self._normalize_split_name(alias) for alias in aliases}
        for candidate in candidates:
            if self._normalize_split_name(candidate.name) in alias_norms:
                return candidate
        return None

    def _split_aliases(self, split: str) -> List[str]:
        norm = self._normalize_split_name(split)
        for canonical, aliases in self.SPLIT_ALIASES.items():
            options = [canonical, *aliases]
            normalized_options = {self._normalize_split_name(opt) for opt in options}
            if norm in normalized_options:
                expanded = OrderedDict()
                for opt in options:
                    expanded[opt] = None
                    expanded[opt.replace('_', ' ')] = None
                    expanded[opt.replace(' ', '_')] = None
                    expanded[opt.replace(' ', '-')] = None
                return list(expanded.keys())

        base = split.replace('-', ' ').replace('_', ' ')
        return list(OrderedDict.fromkeys([
            split,
            base,
            base.replace(' ', '_'),
            base.replace(' ', '-'),
        ]))

    @staticmethod
    def _normalize_split_name(name: str) -> str:
        cleaned = name.lower().strip()
        return ''.join(ch for ch in cleaned if ch.isalnum())

    @staticmethod
    def _gather_items(data_root: Path) -> List[tuple[Path, str]]:
        items: List[tuple[Path, str]] = []
        for cls_dir in sorted(data_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = cls_dir.name
            for img in cls_dir.iterdir():
                if img.is_file() and img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    items.append((img, label))
        return items
