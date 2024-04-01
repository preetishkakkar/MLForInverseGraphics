import math

import torch
from jaxtyping import Float, install_import_hook
from torch import Tensor

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.geometry import (
        homogenize_points,
        homogenize_vectors,
        project,
        transform_cam2world,
        transform_rigid,
        transform_world2cam,
    )


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    batch_size = extrinsics.shape[0]
    height, width = resolution

    """create white canvas"""
    canvases = torch.ones(batch_size, width, height)

    homogenized_vertices = homogenize_points(vertices)

    for i in range(batch_size):
        current_extrinsic = extrinsics[i]
        current_intrinsic = intrinsics[i]
        current_canvas = canvases[i]

        for homogenized_vertex in homogenized_vertices:
            "transform the points into camera space by using transform_world2cam"
            transformed_pt = transform_world2cam(homogenized_vertex, current_extrinsic)

            "project the transformed points into image plane by using project func"
            projected_pt = project(transformed_pt, current_intrinsic)

            "make corresponding color on canvas black"
            ycoord = math.floor(height * projected_pt[0].item())
            xcoord = math.floor(height * projected_pt[1].item())
            current_canvas[xcoord][ycoord] = 0.0

    return canvases

    raise NotImplementedError("This is your homework.")
