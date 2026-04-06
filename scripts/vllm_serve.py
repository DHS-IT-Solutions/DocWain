"""Launch vLLM with NVML driver mismatch workaround."""
import sys
import types

# Build pynvml shim module
shim = types.ModuleType("pynvml")

class _MemInfo:
    total = 80 * 1024**3
    free = 80 * 1024**3
    used = 0

shim.nvmlInit = lambda: None
shim.nvmlShutdown = lambda: None
shim.nvmlDeviceGetCount = lambda: 1
shim.nvmlDeviceGetHandleByIndex = lambda i: 0
shim.nvmlDeviceGetName = lambda h: "NVIDIA A100-SXM4-80GB"
shim.nvmlDeviceGetMemoryInfo = lambda h: _MemInfo()
shim.nvmlDeviceGetUUID = lambda h: "GPU-fake-uuid"
shim.nvmlDeviceGetCudaComputeCapability = lambda h: (8, 0)
shim.nvmlSystemGetDriverVersion = lambda: "550.144.03"
shim.nvmlSystemGetNVMLVersion = lambda: "550.144.03"
shim.NVML_SUCCESS = 0
sys.modules["pynvml"] = shim

import vllm.utils.import_utils as _iu
_iu.import_pynvml = lambda: shim

if __name__ == "__main__":
    import runpy
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
