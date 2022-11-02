from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name: str, sources: list[str]):
    name = f"pcdet.ops.{name}"
    module_dir = Path(name.replace(".", "/")).parent
    return CUDAExtension(
        name=name, sources=[str(module_dir / "src" / source) for source in sources]
    )


if __name__ == "__main__":
    setuptools.setup(
        cmdclass={"build_ext": BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name="iou3d_nms.iou3d_nms_cuda",
                sources=[
                    "iou3d_cpu.cpp",
                    "iou3d_nms_api.cpp",
                    "iou3d_nms.cpp",
                    "iou3d_nms_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d.roiaware_pool3d_cuda",
                sources=[
                    "roiaware_pool3d.cpp",
                    "roiaware_pool3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="roipoint_pool3d.roipoint_pool3d_cuda",
                sources=[
                    "roipoint_pool3d.cpp",
                    "roipoint_pool3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2.pointnet2_stack.pointnet2_stack_cuda",
                sources=[
                    "pointnet2_api.cpp",
                    "ball_query.cpp",
                    "ball_query_gpu.cu",
                    "group_points.cpp",
                    "group_points_gpu.cu",
                    "sampling.cpp",
                    "sampling_gpu.cu",
                    "interpolate.cpp",
                    "interpolate_gpu.cu",
                    "voxel_query.cpp",
                    "voxel_query_gpu.cu",
                    "vector_pool.cpp",
                    "vector_pool_gpu.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2.pointnet2_batch.pointnet2_batch_cuda",
                sources=[
                    "pointnet2_api.cpp",
                    "ball_query.cpp",
                    "ball_query_gpu.cu",
                    "group_points.cpp",
                    "group_points_gpu.cu",
                    "interpolate.cpp",
                    "interpolate_gpu.cu",
                    "sampling.cpp",
                    "sampling_gpu.cu",
                ],
            ),
        ],
    )
