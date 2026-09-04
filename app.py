import gc
import os
import uuid
import torch
import trimesh
import argparse
import gradio as gr
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
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
    geometry_mesh = out.mesh[0]
    del out

    # Upstream exported the raw mesh here and only cleaned it to feed the
    # texture stage. With that stage gone the cleaned mesh is the product, so it
    # is exported instead, and the face budget comes from the request rather
    # than from reduce_face's default.
    geometry_mesh = remove_degenerate_face(geometry_mesh)
    geometry_mesh = reduce_face(geometry_mesh, int(max_facenum))
    geometry_mesh.export(geometry_save_path)

    del geometry_mesh
    gc.collect()
    device = torch.device(args.geometry_device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    print("Generate finish")
    return geometry_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry_model", type=str, default="Step1X-3D-Geometry-Label-1300m"
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--geometry-device",
        default=os.getenv("GEOMETRY_DEVICE", "cuda:0"),
        help="CUDA-Geraet fuer die Geometrie-Pipeline",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    validate_device(args.geometry_device)
    print(f"Device-Aufteilung: Geometrie={args.geometry_device}")

    geometry_model = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder=args.geometry_model
    ).to(args.geometry_device)

    with gr.Blocks(title="Step1X-3D geometry") as demo:
        gr.Markdown(
            "# Step1X-3D — geometry only\n"
            "This fork generates untextured geometry. Colour is assigned "
            "downstream in CAD or in the slicer."
        )
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
                geometry_preview = gr.Model3D(label="Geometry", height=760)
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
            outputs=[geometry_preview],
        )

    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port
    )
