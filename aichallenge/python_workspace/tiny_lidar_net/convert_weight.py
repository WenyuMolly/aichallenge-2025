import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import torch

from lib.model import TinyLidarNet, TinyLidarNetSmall


def extract_params_to_dict(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """Extracts the state dictionary from a PyTorch model and converts it to a NumPy dictionary.

    This function acts as a pure transformation layer, isolating the logic of
    parameter extraction and naming convention changes (dot to underscore) from
    file I/O operations. This design ensures high testability.

    Args:
        model: The PyTorch model instance to extract weights from.

    Returns:
        A dictionary mapping parameter names (with underscores replaced) to
        detached NumPy arrays on the CPU.
    """
    return {
        k.replace('.', '_'): v.detach().cpu().numpy()
        for k, v in model.state_dict().items()
    }


def save_numpy_dict(params: Dict[str, np.ndarray], output_path: Path) -> None:
    """Saves a NumPy dictionary to a file system path.

    Handles the creation of parent directories if they do not exist and
    persists the parameter dictionary as a .npy file.

    Args:
        params: The dictionary of model parameters.
        output_path: The filesystem path where the .npy file will be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, params)
    print(f"Saved NumPy weights to: {output_path}")


def load_model(
    model_name: str, input_dim: int, output_dim: int, ckpt_path: Path
) -> torch.nn.Module:
    """Initializes the model architecture and loads weights from a checkpoint.

    Args:
        model_name: The name of the architecture ('tinylidarnet' or 'tinylidarnet_small').
        input_dim: The size of the input dimension (e.g., LiDAR rays).
        output_dim: The size of the output dimension (e.g., control commands).
        ckpt_path: The path to the PyTorch checkpoint file (.pth).

    Returns:
        The PyTorch model instance with loaded weights.

    Raises:
        ValueError: If the provided model_name is not supported.
        FileNotFoundError: If the checkpoint file does not exist at ckpt_path.
    """
    if model_name == "tinylidarnet":
        model = TinyLidarNet(input_dim=input_dim, output_dim=output_dim)
    elif model_name == "tinylidarnet_small":
        model = TinyLidarNetSmall(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")

    # unwrap common checkpoint wrappers
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw

    # strip 'module.' prefix if present
    def _strip_module(k: str) -> str:
        return k[7:] if k.startswith("module.") else k

    sd = { _strip_module(k): v for k, v in sd.items() }

    # If checkpoint contains TinyLidarImageNet keys with 'lidar_net.' prefix,
    # adapt by extracting only the lidar sub-dict and removing the prefix so
    # it can be loaded into a TinyLidarNet instance (using strict=False so
    # missing/mismatched keys are reported but do not raise).
    if any(k.startswith("lidar_net.") for k in sd.keys()):
        lidar_sd = { k[len("lidar_net."):]: v for k, v in sd.items() if k.startswith("lidar_net.") }

        # Filter keys whose shapes don't match the target model to avoid
        # size-mismatch RuntimeError (e.g., fused fc1 in TinyLidarImageNet).
        model_sd = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in lidar_sd.items():
            if k in model_sd:
                if v.shape == model_sd[k].shape:
                    filtered[k] = v
                else:
                    skipped.append((k, v.shape, model_sd[k].shape))
            else:
                skipped.append((k, None, None))

        res = model.load_state_dict(filtered, strict=False)
        missing = getattr(res, 'missing_keys', None) or []
        unexpected = getattr(res, 'unexpected_keys', None) or []
        print(f"Loaded lidar weights from {ckpt_path} into TinyLidarNet (loaded_keys={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)})")
        if skipped:
            print("Some keys were skipped due to shape mismatch or absence in the target model (sample):")
            for s in skipped[:10]:
                print("  -", s)
    else:
        # Try direct load (may raise if incompatible)
        try:
            model.load_state_dict(sd)
            print(f"Loaded checkpoint: {ckpt_path}")
        except RuntimeError as e:
            # Provide helpful diagnostics
            print(f"Failed to load checkpoint directly into model: {e}")
            # If sd contains keys for a wrapped TinyLidarNet (prefixed), try stripping
            prefixed = any(k.startswith('module.') for k in sd.keys())
            if prefixed:
                sd2 = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
                try:
                    model.load_state_dict(sd2)
                    print(f"Loaded checkpoint after stripping 'module.' prefixes: {ckpt_path}")
                except Exception:
                    raise
            else:
                raise

    return model


def convert_checkpoint(
    model_name: str,
    input_dim: int,
    output_dim: int,
    ckpt: Path,
    output: Path,
    resnet_out: Path | None = None,
    include_resnet_in_npy: bool = False,
    skipped_out: Path | None = None,
) -> None:
    """Orchestrates the model conversion process.

    This function combines the loading of the model architecture, the extraction
    of parameters into a pure dictionary format, and the saving to disk.

    Args:
        model_name: The name of the architecture to load.
        input_dim: The input dimension size.
        output_dim: The output dimension size.
        ckpt: The source path to the PyTorch checkpoint.
        output: The destination path for the converted NumPy file.
        include_resnet_in_npy: If True, embed ResNet weights into the NumPy
            output (keys will have '.' -> '_' name mapping, e.g. 'resnet_conv1_weight').
        skipped_out: Optional path to save any skipped parameters (due to shape
            mismatch) as a .pth so no trained weights are lost.
    """
    # 1. Load Model (I/O & Logic)
    model = load_model(model_name, input_dim, output_dim, ckpt)

    # Reload raw checkpoint state_dict normalized for further extraction
    raw = torch.load(ckpt, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw

    # normalize module prefix
    def _strip_module(k: str) -> str:
        return k[7:] if k.startswith("module.") else k

    sd = { _strip_module(k): v for k, v in sd.items() }

    # Optionally extract ResNet backbone weights from the checkpoint and
    # save them separately as a torch state_dict (.pth)
    if resnet_out is not None:
        resnet_sd = { k[len("resnet."):]: v for k, v in sd.items() if k.startswith("resnet.") }
        if resnet_sd:
            resnet_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(resnet_sd, resnet_out)
            print(f"Saved ResNet backbone state_dict to: {resnet_out}")
        else:
            print(f"[WARN] No 'resnet.' keys found in checkpoint {ckpt}; nothing saved to {resnet_out}")

    # If requested, include ResNet weights directly into the NumPy output
    resnet_npy = {}
    if include_resnet_in_npy:
        resnet_sd = { k: v for k, v in sd.items() if k.startswith("resnet.") }
        for k, v in resnet_sd.items():
            mapped = k.replace('.', '_')
            # store as numpy
            resnet_npy[mapped] = v.detach().cpu().numpy()
        print(f"Included {len(resnet_npy)} ResNet parameters into NumPy output")

    # Also capture any lidar keys that were skipped due to shape mismatch so
    # they can be saved separately (preserve everything trained)
    skipped_dict = {}
    if any(k.startswith("lidar_net.") for k in sd.keys()):
        lidar_sd = { k[len("lidar_net."):]: v for k, v in sd.items() if k.startswith("lidar_net.") }
        model_sd = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in lidar_sd.items():
            if k in model_sd:
                if v.shape == model_sd[k].shape:
                    filtered[k] = v
                else:
                    skipped.append((k, v.shape, model_sd[k].shape))
                    skipped_dict[k] = v
            else:
                skipped.append((k, None, None))
                skipped_dict[k] = v

        # Informative print (load already performed in load_model) but report skipped
        if skipped:
            print(f"Some keys were skipped due to shape mismatch or absence in the target model (sample):")
            for s in skipped[:10]:
                print("  -", s)

    # 2. Extract Parameters (Pure Logic) -> Easy to Unit Test
    params = extract_params_to_dict(model)

    # Merge resnet params into the same numpy dict if requested
    if include_resnet_in_npy:
        # Avoid key collisions by using the dotted->underscore mapping
        params.update(resnet_npy)

    # If user asked to save skipped params, write them to a .pth so nothing is lost
    if skipped_out is not None and skipped_dict:
        skipped_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(skipped_dict, skipped_out)
        print(f"Saved skipped parameters to: {skipped_out}")

    # 3. Save to Disk (I/O)
    save_numpy_dict(params, output)


def main() -> None:
    """Main entry point for the command-line interface.

    Parses command-line arguments and triggers the checkpoint conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert PyTorch weights to NumPy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, choices=["tinylidarnet", "tinylidarnet_small"], default="tinylidarnet", help="Model architecture")
    parser.add_argument("--input-dim", type=int, default=1080, help="Input dimension size")
    parser.add_argument("--output-dim", type=int, default=2, help="Output dimension size")
    parser.add_argument("--ckpt", type=Path, required=True, help="Source .pth checkpoint")
    parser.add_argument("--output", type=Path, default=Path("./weights/converted_weights.npy"), help="Destination .npy path")
    parser.add_argument("--save-resnet", type=Path, default=None, help="Optional path to save extracted ResNet backbone state_dict (.pth)")
    parser.add_argument("--include-resnet-in-npy", action="store_true", help="Embed ResNet parameters into the NumPy output (keys will be dotted->underscored)")
    parser.add_argument("--save-skipped", type=Path, default=None, help="Optional path to save any skipped parameters (due to shape mismatch) as a .pth")

    args = parser.parse_args()

    convert_checkpoint(
        args.model,
        args.input_dim,
        args.output_dim,
        args.ckpt,
        args.output,
        resnet_out=args.save_resnet,
        include_resnet_in_npy=args.include_resnet_in_npy,
        skipped_out=args.save_skipped,
    )


if __name__ == "__main__":
    main()
