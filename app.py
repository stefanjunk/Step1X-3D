import gc
import os
import uuid
import torch
import trimesh
import argparse
import gradio as gr
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validate_device(device_name):
    device = torch.device(device_name)
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{device_name} wurde angefordert, aber PyTorch sieht keine CUDA-GPU. "
            "Pruefe zuerst `nvidia-smi` auf dem Host und im Container."
        )
    device_index = 0 if device.index is None else device.index
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{device_name} wurde angefordert, sichtbar sind aber nur "
            f"{torch.cuda.device_count()} CUDA-GPU(s)."
        )


def generate_func(
    input_image_path, guidance_scale, inference_steps, max_facenum, symmetry, edge_type
):
    if "Label" in args.geometry_model:
        with torch.cuda.device(args.geometry_device):
            out = geometry_model(
                input_image_path,
                label={"symmetry": symmetry, "edge_type": edge_type},
                guidance_scale=float(guidance_scale),
                octree_resolution=384,
                max_facenum=int(max_facenum),
                num_inference_steps=int(inference_steps),
            )
    else:
        with torch.cuda.device(args.geometry_device):
            out = geometry_model(
                input_image_path,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(inference_steps),
                max_facenum=int(max_facenum),
            )

    save_name = str(uuid.uuid4())
    print(save_name)
    geometry_save_path = f"{args.cache_dir}/{save_name}.glb"
    texture_image = out.image.copy()
    geometry_mesh = out.mesh[0]
    del out
    geometry_mesh.export(geometry_save_path)

    geometry_mesh = remove_degenerate_face(geometry_mesh)
    geometry_mesh = reduce_face(geometry_mesh)
    gc.collect()
    with torch.cuda.device(args.geometry_device):
        torch.cuda.empty_cache()
    # Some texture CUDA extensions still use the process-wide current device.
    # Keep them on the configured texture GPU and restore it afterwards.
    with torch.cuda.device(args.texture_device):
        # Geometry preprocessing already produced a cropped RGBA image. Reuse
        # it so the texture stage neither repeats segmentation nor keeps a
        # second background-removal model in host RAM.
        textured_mesh = texture_model(texture_image, geometry_mesh, remove_bg=False)
    textured_save_path = f"{args.cache_dir}/{save_name}-textured.glb"
    textured_mesh.export(textured_save_path)

    del textured_mesh, geometry_mesh, texture_image
    gc.collect()
    for device_name in {
        args.geometry_device,
        args.texture_device,
        args.texture_aux_device,
    }:
        device = torch.device(device_name)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    print("Generate finish")
    return geometry_save_path, textured_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry_model", type=str, default="Step1X-3D-Geometry-Label-1300m"
    )
    parser.add_argument(
        "--texture_model", type=str, default="Step1X-3D-Texture"
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--geometry-device",
        default=os.getenv("GEOMETRY_DEVICE", "cuda:0"),
        help="CUDA-Geraet fuer die Geometrie-Pipeline",
    )
    parser.add_argument(
        "--texture-device",
        default=os.getenv("TEXTURE_DEVICE", "cuda:1"),
        help="CUDA-Geraet fuer Texturgenerierung und Texture Baking",
    )
    parser.add_argument(
        "--texture-aux-device",
        default=os.getenv("TEXTURE_AUX_DEVICE", "cuda:0"),
        help="CUDA-Geraet fuer speicherintensive VAE-Schritte",
    )
    parser.add_argument(
        "--texture-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=env_flag("TEXTURE_CPU_OFFLOAD", False),
        help="SDXL-Komponenten zwischen CPU und Textur-GPU verschieben",
    )
    parser.add_argument(
        "--background-removal-device",
        default=os.getenv("BACKGROUND_REMOVAL_DEVICE", "cuda:1"),
        help="Geraet fuer den BiRefNet-Fallback der Standalone-Texturpipeline",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    validate_device(args.geometry_device)
    validate_device(args.texture_device)
    validate_device(args.texture_aux_device)
    validate_device(args.background_removal_device)
    print(
        "Device-Aufteilung: "
        f"Geometrie={args.geometry_device}, Textur={args.texture_device}, "
        f"Textur-VAE={args.texture_aux_device}, "
        f"BiRefNet-Fallback={args.background_removal_device}, "
        f"Texture-CPU-Offload={args.texture_cpu_offload}"
    )

    geometry_model = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder=args.geometry_model
    ).to(args.geometry_device)

    texture_model = Step1X3DTexturePipeline.from_pretrained(
        "stepfun-ai/Step1X-3D",
        subfolder=args.texture_model,
        device=args.texture_device,
        aux_device=args.texture_aux_device,
        cpu_offload=args.texture_cpu_offload,
        background_removal_device=args.background_removal_device,
    )

    with gr.Blocks(title="Step1X-3D demo") as demo:
        gr.Markdown("# Step1X-3D")
        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(label="Image", type="filepath")
                guidance_scale = gr.Number(label="Guidance Scale", value="7.5")
                inference_steps = gr.Slider(
                    label="Inferece Steps", minimum=1, maximum=100, value=50
                )
                max_facenum = gr.Number(label="Max Face Num", value="400000")
                symmetry = gr.Radio(
                    choices=["x", "asymmetry"],
                    label="Symmetry Type",
                    value="x",
                    type="value",
                )
                edge_type = gr.Radio(
                    choices=["sharp", "normal", "smooth"],
                    label="Edge Type",
                    value="sharp",
                    type="value",
                )
                btn = gr.Button("Start")
            with gr.Column(scale=4):
                textured_preview = gr.Model3D(label="Textured", height=380)
                geometry_preview = gr.Model3D(label="Geometry", height=380)
            with gr.Column(scale=1):
                gr.Examples(
                    examples=[
                        ["examples/images/000.png"],
                        ["examples/images/001.png"],
                        ["examples/images/004.png"],
                        ["examples/images/008.png"],
                        ["examples/images/028.png"],
                        ["examples/images/032.png"],
                        ["examples/images/061.png"],
                        ["examples/images/107.png"],
                    ],
                    inputs=[input_image],
                    cache_examples=False,
                )

        btn.click(
            generate_func,
            inputs=[
                input_image,
                guidance_scale,
                inference_steps,
                max_facenum,
                symmetry,
                edge_type,
            ],
            outputs=[geometry_preview, textured_preview],
        )

    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port
    )
