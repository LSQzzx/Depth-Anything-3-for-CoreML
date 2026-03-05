import argparse
import sys
from pathlib import Path
from typing import Tuple

import coremltools as ct
import numpy
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor, inference_mode
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F

sys.path.insert(0, str(Path.cwd() / "src"))
print(Path.cwd())
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.model.cam_enc import CameraEnc
from depth_anything_3.model.cam_dec import CameraDec

INPUT_SIZE = 504
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_WEIGHTS_PATH = Path(Path.cwd() / "DA3-SMALL" / "model.safetensors")
DEFAULT_COREML_OUTPUT = Path("da3.mlpackage")
EXAMPLE_INPUT_SHAPE = (1, 4, 3, INPUT_SIZE, 378)


def _build_backbone() -> DinoV2:
    return DinoV2(
        name="vits",
        out_layers=[5, 7, 9, 11],
        alt_start=4,
        qknorm_start=4,
        rope_start=4,
        cat_token=True,
    )

def _build_head() -> DualDPT:
    return DualDPT(
        dim_in=768,
        output_dim=2,
        features=64,
        out_channels=[48, 96, 192, 384],
        pos_embed=False,
    )

def _build_camera_encoder() -> CameraEnc:
    return CameraEnc(
        dim_out=384
    )

def _build_camera_decoder() -> CameraDec:
    return CameraDec(
        dim_in=768
    )

def build_depth_estimator() -> DepthAnything3Net:
    return DepthAnything3Net(net=_build_backbone(), head=_build_head(), cam_enc=_build_camera_encoder(), cam_dec=_build_camera_decoder())

def load_depth_estimator(weights_path: Path) -> DepthAnything3Net:
    model = build_depth_estimator()
    checkpoint = load_file(str(weights_path))
    sanitized = {k.replace("model.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(sanitized, strict=False)
    return model.eval()

MODEL = load_depth_estimator(DEFAULT_WEIGHTS_PATH)

def resize_and_pad(image: Image.Image, target_size: int) -> Tuple[Image.Image, Tuple[int, int]]:
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    resized = F.resize(image, [scaled_height, scaled_width]).convert("RGB")
    padding = (0, 0, target_size - scaled_width, target_size - scaled_height)
    padded = F.pad(resized, padding, fill=0)
    return padded, (scaled_width, scaled_height)

def tensor_from_image(image: Image.Image) -> Tensor:
    tensor = F.to_tensor(image)
    return F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

@inference_mode()
def estimate_depth(image: Image.Image, depth_model: DepthAnything3Net | None = None) -> numpy.ndarray:
    depth_model = depth_model or MODEL
    padded_image, (scaled_width, scaled_height) = resize_and_pad(image, INPUT_SIZE)
    img_tensor = tensor_from_image(padded_image)
    depth_output = depth_model(img_tensor[None, None])[0]
    depth_crop = depth_output[:, :, :scaled_height, :scaled_width]
    width, height = image.size
    depth_resized = interpolate(depth_crop, size=(height, width), mode="bilinear")
    return depth_resized[0, 0].cpu().numpy()

def visualize_depth(depth: numpy.ndarray) -> Image.Image:
    from cv2 import COLOR_BGR2RGB, COLORMAP_INFERNO, applyColorMap, cvtColor

    depth_range = depth.max() - depth.min()
    depth_normalized = (depth - depth.min()) / (depth_range + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(numpy.uint8)
    depth_colored = applyColorMap(depth_uint8, COLORMAP_INFERNO)
    depth_colored = cvtColor(depth_colored, COLOR_BGR2RGB)
    return Image.fromarray(depth_colored)

def export_coreml_model(depth_model: DepthAnything3Net, output_path: Path) -> None:
    example_input = torch.rand(EXAMPLE_INPUT_SHAPE)
    traced_model = torch.jit.trace(depth_model, example_input)
    model_from_trace = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=EXAMPLE_INPUT_SHAPE, name="inputs")],
        outputs=[
            ct.TensorType(name="depth"),
            ct.TensorType(name="depth_conf"),
            ct.TensorType(name="extrinsics"),
            ct.TensorType(name="intrinsics"),
        ],
        )
    model_from_trace.save(str(output_path))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Anything 3 CoreML converter")
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run a quick depth inference on assets/examples/SOH/000.png before exporting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_test:
        sample_image = Image.open("assets/examples/SOH/000.png")
        depth_map = estimate_depth(sample_image)
        depth_image = visualize_depth(depth_map)
        depth_image.show()

    export_coreml_model(MODEL, DEFAULT_COREML_OUTPUT)


if __name__ == "__main__":
    main()
