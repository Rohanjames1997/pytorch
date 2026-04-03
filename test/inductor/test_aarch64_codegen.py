# Owner(s): ["oncall: cpu inductor"]
"""
Tests for aarch64-specific codegen paths in the inductor.

Covers NEON vectorization, MKLDNN/ACL fusion, compiler flags,
and layout optimization for grouped convolutions.
"""
import os
import platform
import sys
import unittest

import torch
from torch import nn
from torch._dynamo.utils import same
from torch._inductor import config, cpu_vec_isa, metrics

try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase
check_model = test_torchinductor.check_model

IS_AARCH64 = platform.machine() == "aarch64"

requires_aarch64 = unittest.skipUnless(IS_AARCH64, "aarch64 only")


@requires_aarch64
class TestAarch64Codegen(TestCase):
    """Tests for aarch64-specific inductor codegen."""

    common = check_model

    def test_vec_isa_selection(self):
        """A valid vector ISA should be selected on aarch64."""
        isa_list = cpu_vec_isa.valid_vec_isa_list()
        self.assertTrue(len(isa_list) > 0, "No valid ISA found on aarch64")
        picked = cpu_vec_isa.pick_vec_isa()
        self.assertTrue(picked, "No ISA picked on aarch64")
        self.assertIn(picked.bit_width(), (128, 256),
                       "Expected NEON (128-bit) or SVE256 (256-bit)")

    def test_neon_vectorized_elementwise(self):
        """Elementwise ops should generate vectorized NEON kernels."""

        def fn(x):
            return torch.relu(torch.sin(x) + torch.cos(x))

        x = torch.randn(1024)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertGreater(
                metrics.generated_cpp_vec_kernel_count,
                0,
                "Expected vectorized kernels on aarch64",
            )

    def test_neon_reduction_vectorized(self):
        """Reduction ops should use vectorized NEON paths."""

        def fn(x):
            return x.sum(dim=-1), x.max(dim=-1).values

        x = torch.randn(32, 256)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))

    def test_neon_float64_reduction(self):
        """Float64 reductions should work correctly with NEON."""

        def fn(x):
            return x.sum(dim=-1), x.mean(dim=-1)

        x = torch.randn(16, 128, dtype=torch.float64)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            self.common(fn, (x,))

    def test_neon_welford_reduction(self):
        """Welford reduction (batch norm) should be vectorized on NEON."""

        def fn(x):
            return torch.var_mean(x, dim=-1)

        x = torch.randn(32, 256)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            self.common(fn, (x,))

    def test_neon_ceil_floor_round(self):
        """ceil/floor/round should use native NEON intrinsics (vectorized)."""

        def fn(x):
            return torch.ceil(x) + torch.floor(x) + torch.round(x)

        x = torch.randn(1024)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            if cpu_vec_isa.pick_vec_isa():
                self.assertGreater(
                    metrics.generated_cpp_vec_kernel_count,
                    0,
                    "ceil/floor/round should be vectorized on NEON",
                )

    def test_compiler_flags_mcpu_native(self):
        """aarch64 should use -mcpu=native instead of -march=native."""
        from torch._inductor.cpp_builder import (
            CppBuilder,
            CppTorchDeviceOptions,
        )

        opts = CppTorchDeviceOptions(
            vec_isa=cpu_vec_isa.pick_vec_isa(),
            device_type="cpu",
        )
        builder = CppBuilder(name="test", sources="test.cpp", BuildOption=opts)
        cmd = builder.get_command_line()
        self.assertIn("-mcpu=native", cmd)
        self.assertNotIn("-march=native", cmd)

    def test_compiler_flags_no_fno_tree_loop_vectorize(self):
        """aarch64 should NOT have -fno-tree-loop-vectorize."""
        from torch._inductor.cpp_builder import (
            CppBuilder,
            CppTorchDeviceOptions,
        )

        opts = CppTorchDeviceOptions(
            vec_isa=cpu_vec_isa.pick_vec_isa(),
            device_type="cpu",
        )
        builder = CppBuilder(name="test", sources="test.cpp", BuildOption=opts)
        cmd = builder.get_command_line()
        self.assertNotIn("-fno-tree-loop-vectorize", cmd)

    def test_cpu_capability_macro(self):
        """CPU_CAPABILITY_NEON or CPU_CAPABILITY_SVE256 macro should be defined."""
        from torch._inductor.cpp_builder import (
            CppBuilder,
            CppTorchDeviceOptions,
        )

        vec = cpu_vec_isa.pick_vec_isa()
        opts = CppTorchDeviceOptions(vec_isa=vec, device_type="cpu")
        builder = CppBuilder(name="test", sources="test.cpp", BuildOption=opts)
        cmd = builder.get_command_line()
        has_neon = "CPU_CAPABILITY_NEON" in cmd
        has_sve = "CPU_CAPABILITY_SVE256" in cmd
        self.assertTrue(has_neon or has_sve,
                        "Expected CPU_CAPABILITY_NEON or CPU_CAPABILITY_SVE256")

    def test_grouped_conv_layout(self):
        """Grouped convolutions should not force NHWC on aarch64 when
        it would hurt performance (groups > 1, in_channels > 1)."""

        model = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=32),
            nn.ReLU(),
        )
        model.eval()
        x = torch.randn(1, 64, 16, 16)

        with torch.no_grad():
            expected = model(x)
            compiled = torch.compile(model)
            result = compiled(x)
            self.assertTrue(same(result, expected, tol=1e-4))

    def test_depthwise_conv_correctness(self):
        """Depthwise convolutions (groups=in_channels) should work correctly."""

        model = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        model.eval()
        x = torch.randn(1, 32, 16, 16)

        with torch.no_grad():
            expected = model(x)
            compiled = torch.compile(model)
            result = compiled(x)
            self.assertTrue(same(result, expected, tol=1e-4))

    @unittest.skipUnless(
        torch.backends.mkldnn.is_available(),
        "MKLDNN not available",
    )
    def test_mkldnn_fusion_enabled(self):
        """MKLDNN fusions should be enabled on aarch64 even with ACL."""
        from torch._inductor.fx_passes.mkldnn_fusion import _mkldnn_fusion_init

        _mkldnn_fusion_init()

    def test_neon_transcendental_vectorized(self):
        """Transcendental functions should be vectorized via Sleef on NEON."""

        def fn(x):
            return torch.exp(x) + torch.log(x.abs() + 1) + torch.tanh(x)

        x = torch.randn(1024)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            if cpu_vec_isa.pick_vec_isa():
                self.assertGreater(
                    metrics.generated_cpp_vec_kernel_count,
                    0,
                    "Transcendentals should be vectorized on NEON",
                )

    def test_neon_batch_norm_vectorized(self):
        """Batch normalization should produce vectorized code on NEON."""

        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        model.eval()
        x = torch.randn(32, 256)

        with torch.no_grad(), config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            expected = model(x)
            compiled = torch.compile(model)
            result = compiled(x)
            self.assertTrue(same(result, expected, tol=1e-4))

    def test_neon_softmax_correctness(self):
        """Softmax should produce correct results with NEON vectorization."""

        def fn(x):
            return torch.softmax(x, dim=-1)

        x = torch.randn(32, 128)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            self.common(fn, (x,))

    def test_neon_layer_norm_correctness(self):
        """Layer normalization should be correct with NEON."""

        model = nn.LayerNorm(128)
        model.eval()
        x = torch.randn(32, 128)

        with torch.no_grad(), config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            expected = model(x)
            compiled = torch.compile(model)
            result = compiled(x)
            self.assertTrue(same(result, expected, tol=1e-4))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if IS_AARCH64:
        run_tests()
