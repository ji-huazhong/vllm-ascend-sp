import vllm
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.worker.worker_base import WorkerBase

from vllm_sp.args import EngineArgsPatch, AsyncEngineArgsPatch
from vllm_sp.config import ParallelConfigPatch, VllmConfigPatch
from vllm_sp.ulysses import apply_ulysses_patches
from .utils import PatchHelper


class EngineCoreProcPatch(PatchHelper[EngineCoreProc]):

    _orig_run_engine_core = EngineCoreProc.run_engine_core

    @staticmethod
    def run_engine_core(*args, **kwargs):
        # When starting the API server, it will spawn a new process to run the
        # EngineCore. We need to load the plugins in the new process before it
        # initializes the Executor.
        vllm.plugins.load_general_plugins()
        return EngineCoreProcPatch._orig_run_engine_core(*args, **kwargs)



class WorkerBasePatch(PatchHelper[WorkerBase]):

    _orig_init = WorkerBase.__init__

    def __init__(self, *args, **kwargs):
        # Some patches like the GPUModelRunner will import CUDA libraries when
        # they are initialized, which will cause process forking to fail. For
        # these patches, we need to delay the initialization until after the
        # process has been forked (i.e., in the WorkerBase initializer).
        from vllm_sp.model_runner import NPUModelRunnerPatch

        NPUModelRunnerPatch.apply_patch()

        return self._orig_init(*args, **kwargs)



def ascend_plugin():
    # Patches that make later patches work properly.
    EngineCoreProcPatch.apply_patch()
    WorkerBasePatch.apply_patch()

    # Patches to vLLM arguments and configuration objects.
    EngineArgsPatch.apply_patch()
    AsyncEngineArgsPatch.apply_patch()
    ParallelConfigPatch.apply_patch()
    VllmConfigPatch.apply_patch()

    # Main optimization patches
    apply_ulysses_patches()
