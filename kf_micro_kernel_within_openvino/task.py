import gzip
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kernelfoundry import TestBase
from kernelfoundry.eval_pipeline.utils.performance import detect_profiler, profiler_session


OPENVINO_ROOT = Path("/opt/openvino")
OPENVINO_BUILD = OPENVINO_ROOT / "build"
OPENVINO_BIN = OPENVINO_ROOT / "bin/intel64/Release"
GPU_UNIT_TESTS_TARGET = "ov_gpu_unit_tests"
GPU_UNIT_TESTS = OPENVINO_BIN / GPU_UNIT_TESTS_TARGET

KERNEL_RELATIVE_PATH = "src/plugins/intel_gpu/src/graph/impls/ocl_v2/gr_up_micro.cl"
KERNEL_SOURCE = OPENVINO_ROOT / KERNEL_RELATIVE_PATH
TEST_SOURCE_NAME = "gr_up_micro_kf_test.cpp"
TEST_DIR = OPENVINO_ROOT / "src/plugins/intel_gpu/tests/unit"
TEST_DESTINATION = TEST_DIR / "test_cases" / TEST_SOURCE_NAME
TEST_SOURCES_LIST = TEST_DIR / "sources.cmake"
TEST_SOURCES_ANCHOR = "${CMAKE_CURRENT_LIST_DIR}/test_cases/gated_residual_gpu_test.cpp"
TEST_SOURCES_ENTRY = f"    ${{CMAKE_CURRENT_LIST_DIR}}/test_cases/{TEST_SOURCE_NAME}\n"

GTEST_SUITE = "kf/gr_up_micro_kf_test"
# M (tokens) of the UP_PROJ micro-GEMM; K = L = 320, N = C * D = 2560 are fixed in the gtest.
TOKENS = [1024, 2048, 4096, 8192]
BENCHMARK_ITERATIONS = 100
EVOLVE_PATTERN = re.compile(r"// \[EVOLVE_START\]\s*(.*?)\s*// \[EVOLVE_END\]", re.DOTALL)


def _task_dir() -> Path:
    return Path(__file__).parent


def _artifact_dir() -> Path:
    path = _task_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _install_test_source() -> None:
    shutil.copyfile(_task_dir() / TEST_SOURCE_NAME, TEST_DESTINATION)
    sources = TEST_SOURCES_LIST.read_text()
    if TEST_SOURCE_NAME in sources:
        return
    lines = sources.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if TEST_SOURCES_ANCHOR in line:
            lines.insert(index + 1, TEST_SOURCES_ENTRY)
            TEST_SOURCES_LIST.write_text("".join(lines))
            return
    raise ValueError(f"{TEST_SOURCES_ANCHOR} not found in {TEST_SOURCES_LIST}")


def _write_kernel(patch_source: bool) -> None:
    if patch_source:
        candidate = (_task_dir() / "kernel.cl").read_text()
        match = EVOLVE_PATTERN.search(candidate)
        if match is None:
            raise ValueError("kernel.cl must contain a well-formed evolve region")
        source = match.group(1).strip()
        if "KERNEL(gr_up_micro)" not in source:
            raise ValueError("The evolve region must define the OpenVINO gr_up_micro kernel")
        KERNEL_SOURCE.write_text(source + "\n")
    else:
        # The reference is the kernel of the checked-out commit, whatever an earlier candidate build left in the tree.
        original = subprocess.run(
            ["git", "-C", str(OPENVINO_ROOT), "show", f"HEAD:{KERNEL_RELATIVE_PATH}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        KERNEL_SOURCE.write_text(original)


def _prepare_artifacts(variant: str, patch_source: bool) -> list[str]:
    _write_kernel(patch_source)
    _install_test_source()
    subprocess.run(
        ["cmake", "--build", str(OPENVINO_BUILD), "--target", GPU_UNIT_TESTS_TARGET, "--parallel", "4"],
        check=True,
        timeout=1700,
    )
    archive = _artifact_dir() / f"{GPU_UNIT_TESTS_TARGET}_{variant}.gz"
    with GPU_UNIT_TESTS.open("rb") as source_file, gzip.open(archive, "wb", compresslevel=6) as archive_file:
        shutil.copyfileobj(source_file, archive_file)
    return [str(archive.resolve())]


def _environment() -> dict:
    environment = os.environ.copy()
    library_paths = [str(OPENVINO_BIN)]
    if environment.get("LD_LIBRARY_PATH"):
        library_paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return environment


class TestOpenVINOGrUpMicro(TestBase):
    """Replace OpenVINO's gr_up_micro.cl (UP_PROJ micro-GEMM of GatedResidual), rebuild the GPU unit tests and run them."""

    def build(self, gpu_arch, **buildtime_params) -> list[str]:
        return _prepare_artifacts("candidate", patch_source=True)

    def build_reference(self, gpu_arch, **buildtime_params) -> list[str]:
        return _prepare_artifacts("reference", patch_source=False)

    @pytest.fixture
    def unit_tests(self, use_reference) -> Path:
        variant = "reference" if use_reference else "candidate"
        archive = _artifact_dir() / f"{GPU_UNIT_TESTS_TARGET}_{variant}.gz"
        executable = _artifact_dir() / f"{GPU_UNIT_TESTS_TARGET}_{variant}"
        with gzip.open(archive, "rb") as archive_file, executable.open("wb") as executable_file:
            shutil.copyfileobj(archive_file, executable_file)
        executable.chmod(0o755)
        return executable

    @pytest.mark.parametrize("tokens", TOKENS)
    def test_correctness(self, unit_tests, tokens):
        result = subprocess.run(
            [str(unit_tests), f"--gtest_filter={GTEST_SUITE}.correctness/M{tokens}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=OPENVINO_BUILD,
            env=_environment(),
        )
        assert result.returncode == 0, f"gr_up_micro correctness failed:\n{result.stdout}\n{result.stderr}"
        assert "[  PASSED  ] 1 test" in result.stdout, f"gr_up_micro correctness test did not run:\n{result.stdout}"
        assert "[  SKIPPED ]" not in result.stdout, f"gr_up_micro correctness test was skipped:\n{result.stdout}"

    @pytest.mark.parametrize("tokens", TOKENS)
    @pytest.mark.performance
    def test_benchmark(self, unit_tests, tokens, request, profile_store):
        environment = _environment()
        environment["KF_GR_ITERS"] = str(BENCHMARK_ITERATIONS)
        process = subprocess.Popen(
            [str(unit_tests), f"--gtest_filter={GTEST_SUITE}.benchmark/M{tokens}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=OPENVINO_BUILD,
            env=environment,
        )
        try:
            banner = []
            for line in process.stdout:
                if line.strip() == "KF_READY":
                    break
                banner.append(line)
            else:
                raise AssertionError("Benchmark runner did not initialize:\n" + "".join(banner))

            runtimes = []
            profiler_name = detect_profiler() or ""
            with profiler_session(profiler_name):
                process.stdin.write("RUN\n")
                process.stdin.flush()
                done = False
                trailer = []
                for line in process.stdout:
                    line = line.strip()
                    if line.startswith("KF_RUNTIME_MS "):
                        runtime_ms = float(line.removeprefix("KF_RUNTIME_MS "))
                        assert math.isfinite(runtime_ms) and runtime_ms > 0, f"Invalid runtime: {runtime_ms}"
                        runtimes.append(runtime_ms)
                    elif line == "KF_DONE":
                        done = True
                        break
                    else:
                        trailer.append(line)
                assert done, "Benchmark runner did not finish cleanly:\n" + "\n".join(trailer)

            return_code = process.wait(timeout=120)
            assert return_code == 0, f"Benchmark runner exited {return_code}"
            assert len(runtimes) == BENCHMARK_ITERATIONS
            profile_store[request.node.nodeid] = runtimes
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    task = TestOpenVINOGrUpMicro()
    for variant, build_function in (("reference", task.build_reference), ("candidate", task.build)):
        for artifact in build_function(gpu_arch="ptl"):
            print(f"ARTIFACT {variant} {Path(artifact).name}: {Path(artifact).stat().st_size} bytes", flush=True)
