import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image
import torchvision.transforms as T
from torch import Tensor

import numpy as np
from torch.utils.data import Dataset, ConcatDataset

logger = logging.getLogger(__name__)


class ScanControlSequenceDataset(Dataset):
    """
    A PyTorch Dataset for a single sequence of LiDAR scans and control commands.

    Loads synchronized .npy files (scans, steers, accelerations) from a specific
    directory. The LiDAR scans are normalized by the specified maximum range.

    Attributes:
        seq_dir (Path): Path to the sequence directory.
        max_range (float): Maximum range for LiDAR normalization.
        scans (np.ndarray): Normalized scan data array (N, num_points).
        steers (np.ndarray): Steering angle array (N,).
        accels (np.ndarray): Acceleration array (N,).
    """

    def __init__(self, seq_dir: Union[str, Path], max_range: float = 30.0):
        """
        Initializes the dataset from a sequence directory.

        Args:
            seq_dir: Path to the directory containing .npy files.
            max_range: Maximum range value to normalize LiDAR data (0.0 to 1.0).

        Raises:
            ValueError: If data lengths do not match or files are missing.
        """
        self.seq_dir = Path(seq_dir)
        self.max_range = max_range

        try:
            # Load raw data
            self.scans = np.load(self.seq_dir / "scans.npy")         # Shape: (N, num_points)
            self.steers = np.load(self.seq_dir / "steers.npy")       # Shape: (N,)
            self.accels = np.load(self.seq_dir / "accelerations.npy") # Shape: (N,)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required .npy files in {self.seq_dir}: {e}")

        # Validate data consistency
        n_samples = len(self.scans)
        if not (len(self.steers) == n_samples and len(self.accels) == n_samples):
            raise ValueError(
                f"Data length mismatch in {self.seq_dir}: "
                f"Scans={len(self.scans)}, Steers={len(self.steers)}, Accels={len(self.accels)}"
            )

        # Preprocessing: Clip and Normalize
        # Values are clipped to [0, max_range] and then scaled to [0, 1]
        self.scans = np.clip(self.scans, 0.0, self.max_range) / self.max_range


class ImageScanSequenceDataset(ScanControlSequenceDataset):
    """Sequence dataset that pairs images with LiDAR scans and control targets.

    Each image is used for a fixed number of consecutive scans (determined by
    `img_scan_ratio`). For example, with `img_scan_ratio=4`, one image covers
    4 scans. No downsampling is performed on LiDAR; instead, images are repeated
    to align with all scans.
    """

    def __init__(
        self,
        seq_dir: Union[str, Path],
        max_range: float = 30.0,
        image_dir: str = "images",
        img_scan_ratio: int = 4,
        img_size: int = 224,
        img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        super().__init__(seq_dir, max_range=max_range)

        img_dir = Path(seq_dir) / image_dir
        if not img_dir.exists() or not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in img_exts])
        if not img_files:
            raise FileNotFoundError(f"No images found in {img_dir}")

        # Keep all LiDAR scans (no downsampling)
        scans_orig = self.scans
        steers_orig = self.steers
        accels_orig = self.accels
        n_scan = len(scans_orig)
        n_img = len(img_files)

        # Each image covers img_scan_ratio consecutive scans
        # Repeat each image img_scan_ratio times to match all scans
        img_indices = []
        for img_idx in range(n_img):
            img_indices.extend([img_idx] * img_scan_ratio)

        # Handle case where scans don't divide evenly by img_scan_ratio
        if len(img_indices) < n_scan:
            # Extend by repeating last image for remaining scans
            remaining = n_scan - len(img_indices)
            img_indices.extend([n_img - 1] * remaining)
        elif len(img_indices) > n_scan:
            # Truncate if we have too many image references
            img_indices = img_indices[:n_scan]

        # Verify alignment
        if len(img_indices) != n_scan:
            raise RuntimeError(
                f"Failed to align scans ({n_scan}) with image indices ({len(img_indices)}) in {seq_dir}"
            )

        # Store the mapping
        self.scans = scans_orig.astype(np.float32)
        self.steers = steers_orig.astype(np.float32)
        self.accels = accels_orig.astype(np.float32)
        self.img_files = img_files
        self.img_indices = img_indices

        # Set up image transform
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ])

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tensor, np.ndarray]:
        scan = self.scans[idx].astype(np.float32)
        accel = np.float32(self.accels[idx])
        steer = np.float32(self.steers[idx])
        target = np.array([accel, steer], dtype=np.float32)

        # Get the corresponding image based on img_indices mapping
        img_idx = self.img_indices[idx]
        img_path = self.img_files[img_idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return scan, img, target


class MultiSeqConcatDataset(ConcatDataset):
    """
    A PyTorch ConcatDataset that aggregates multiple SequenceDatasets.

    Automatically discovers valid sequence directories within a root directory.
    Supports filtering sequences using inclusion and exclusion keywords.
    """

    def __init__(
        self, 
        dataset_root: Union[str, Path], 
        max_range: float = 30.0, 
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None,
        # Image related options
        image_dir: str = "images",
        img_scan_ratio: int = 4,
        img_size: int = 224,
        img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
    ):
        """
        Initializes the concatenated dataset.

        Args:
            dataset_root: Root directory containing sequence folders.
            max_range: Maximum range for LiDAR normalization.
            include: List of substrings; if provided, only directories containing
                     at least one of these substrings will be loaded.
            exclude: List of substrings; directories containing any of these
                     substrings will be skipped.

        Raises:
            RuntimeError: If no valid sequences are found after filtering.
        """
        dataset_root = Path(dataset_root)
        
        # If the dataset root itself contains the required files (flat layout),
        # treat it as a single sequence. This supports datasets where scans.npy
        # / steers.npy / accelerations.npy live directly under train/.
        required_files = ["scans.npy", "steers.npy", "accelerations.npy"]
        datasets = []
        if all((dataset_root / f).exists() for f in required_files):
            images_path = dataset_root / image_dir
            if images_path.exists() and images_path.is_dir():
                try:
                    ds = ImageScanSequenceDataset(
                        dataset_root,
                        max_range=max_range,
                        image_dir=image_dir,
                        img_scan_ratio=img_scan_ratio,
                        img_size=img_size,
                        img_mean=img_mean,
                        img_std=img_std,
                        img_exts=img_exts,
                    )
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to load root sequence with images {dataset_root}: {e}")
            else:
                try:
                    ds = ScanControlSequenceDataset(dataset_root, max_range=max_range)
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to load root sequence {dataset_root}: {e}")

        # Discover all subdirectories
        all_seq_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        target_seq_dirs = []

        # Apply filters
        for p in all_seq_dirs:
            name = p.name

            # Skip top-level image directories (e.g., a dataset root "images/")
            if name == image_dir:
                logger.debug(f"Skipping top-level image directory: {p}")
                continue

            # Check inclusion criteria (OR logic)
            if include and not any(inc in name for inc in include):
                continue

            # Check exclusion criteria (OR logic)
            if exclude and any(exc in name for exc in exclude):
                continue

            target_seq_dirs.append(p)

        # Instantiate datasets (keep any dataset(s) found at the root)
        for seq_dir in target_seq_dirs:
            # Quick check for file existence before initialization
            required_files = ["scans.npy", "steers.npy", "accelerations.npy"]

            # If an image folder exists, use ImageScanSequenceDataset which will
            # downsample LiDAR to synchronize with images.
            images_path = seq_dir / image_dir
            if images_path.exists() and images_path.is_dir():
                try:
                    ds = ImageScanSequenceDataset(
                        seq_dir,
                        max_range=max_range,
                        image_dir=image_dir,
                        img_scan_ratio=img_scan_ratio,
                        img_size=img_size,
                        img_mean=img_mean,
                        img_std=img_std,
                        img_exts=img_exts,
                    )
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to load sequence with images {seq_dir}: {e}")
            elif all((seq_dir / f).exists() for f in required_files):
                try:
                    ds = ScanControlSequenceDataset(seq_dir, max_range=max_range)
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to load sequence {seq_dir}: {e}")
            else:
                logger.warning(f"Skipping {seq_dir.name}: Missing .npy files.")

        if not datasets:
            # For debugging, list candidate directories and explain why they were skipped
            cand = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
            logger.debug(f"Candidate subdirectories: {cand}")
            raise RuntimeError(f"No valid sequences found in {dataset_root} with provided filters.")

        super().__init__(datasets)
        logger.info(f"Loaded {len(datasets)} sequences from {dataset_root}. Total samples: {len(self)}")
