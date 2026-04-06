"""Drop-in pynvml replacement for systems with NVML driver mismatch.

Import this module before vLLM to make it think NVML is working.
Sets up a site-packages override so vllm.utils.import_utils.import_pynvml
returns this shim instead of the real pynvml.
"""


class _MemInfo:
    total = 80 * 1024 * 1024 * 1024
    free = 80 * 1024 * 1024 * 1024
    used = 0


NVML_SUCCESS = 0


def nvmlInit():
    pass


def nvmlShutdown():
    pass


def nvmlDeviceGetCount():
    return 1


def nvmlDeviceGetHandleByIndex(index):
    return 0


def nvmlDeviceGetName(handle):
    return "NVIDIA A100-SXM4-80GB"


def nvmlDeviceGetMemoryInfo(handle):
    return _MemInfo()


def nvmlDeviceGetUUID(handle):
    return "GPU-fake-uuid-for-nvml-shim"


def nvmlDeviceGetCudaComputeCapability(handle):
    return (8, 0)


def nvmlSystemGetDriverVersion():
    return "550.144.03"


def nvmlSystemGetNVMLVersion():
    return "550.144.03"
