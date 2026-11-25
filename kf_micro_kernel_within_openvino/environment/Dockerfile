# Image with a pre-built OpenVINO (GPU plugin + ov_gpu_unit_tests, debug caps and oneDNN for GPU enabled).
# The repository / branch below hold the GatedResidual oneDNN READ path whose UP_PROJ micro-GEMM kernel (gr_up_micro.cl) is optimized.
#
# Build from this folder:
#   docker build -f Dockerfile -t kf-openvino-gr-up-micro:<tag> .
# Optional: --build-arg OPENVINO_REF=<commit-sha> pins a commit of the branch.
FROM amr-registry.caas.intel.com/caisp-team/kernelfoundry/sycl-all

ARG OPENVINO_REPO_URL=https://github.com/riverlijunjie/openvino-1.git
ARG OPENVINO_BRANCH=river/qwen4_gr_impl_micro_gemm_v2
ARG OPENVINO_REF=

ENV DEBIAN_FRONTEND=noninteractive \
    OPENVINO_ROOT=/opt/openvino \
    OPENVINO_BUILD_DIR=/opt/openvino/build \
    CCACHE_DIR=/opt/ccache

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        cmake \
        git \
        libdrm-dev \
        libpugixml-dev \
        libtbb-dev \
        libva-dev \
        libwayland-dev \
        libx11-dev \
        libxext-dev \
        libxfixes-dev \
        libxrandr-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Without OPENVINO_REF the tip of OPENVINO_BRANCH is used.
RUN git clone --filter=blob:none --no-checkout "${OPENVINO_REPO_URL}" "${OPENVINO_ROOT}" && \
    git -C "${OPENVINO_ROOT}" checkout --detach "${OPENVINO_REF:-origin/${OPENVINO_BRANCH}}" && \
    git -C "${OPENVINO_ROOT}" submodule update --init --recursive --depth 1

# ENABLE_DEBUG_CAPS exposes GPU_GR_KERNEL_SELECTION, which forces the oneDNN READ path (and so gr_up_micro) on any GPU.
RUN cmake -S "${OPENVINO_ROOT}" -B "${OPENVINO_BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DENABLE_INTEL_GPU=ON \
        -DENABLE_ONEDNN_FOR_GPU=ON \
        -DENABLE_DEBUG_CAPS=ON \
        -DENABLE_TESTS=ON \
        -DENABLE_SAMPLES=OFF && \
    cmake --build "${OPENVINO_BUILD_DIR}" \
        --target ov_gpu_unit_tests \
        --parallel 12

WORKDIR /opt/openvino
