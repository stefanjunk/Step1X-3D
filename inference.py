import warnings

warnings.filterwarnings("ignore")
import os

from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
import torch

def geometry_pipeline(input_image_path, save_glb_path):
    """
    The base geometry model, input image generate glb
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m'
    ).to(os.getenv("GEOMETRY_DEVICE", "cuda:0"))

    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)
    out = pipeline(input_image_path, guidance_scale=7.5, num_inference_steps=50, generator=generator)

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    out.mesh[0].export(save_glb_path)


def geometry_label_pipeline(input_image_path, save_glb_path):
    """
    The label geometry model, support using label to control generation, input image generate glb
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-Label-1300m'
    ).to(os.getenv("GEOMETRY_DEVICE", "cuda:0"))
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)

    out = pipeline(
        input_image_path,
        label={"symmetry": "x", "edge_type": "sharp"},
        guidance_scale=7.5,
        octree_resolution=384,
        max_facenum=400000,
        num_inference_steps=50,
        generator=generator
    )

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    out.mesh[0].export(save_glb_path)


if __name__ == "__main__":
    image_path = "examples/images/000.png"
    geometry_pipeline(image_path, "output/000.glb")
    geometry_label_pipeline(image_path, "output/000-label.glb")
