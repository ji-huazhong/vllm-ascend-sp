from dataclasses import dataclass

from vllm.config import ParallelConfig, VllmConfig

from .args import get_current_ascend_args
from .utils import PatchHelper


@dataclass
class AscendParallelConfig(ParallelConfig):
    ulysses_sequence_parallel_size: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ascend_args = get_current_ascend_args()
        self.ulysses_sequence_parallel_size = ascend_args.ulysses_sequence_parallel_size
    
    @property
    def world_size(self) -> int:
        return (self.pipeline_parallel_size *
                self.tensor_parallel_size *
                self.ulysses_sequence_parallel_size)
    
    @world_size.setter
    def world_size(self, value: int) -> None:
        # ParallelConfig.__post_init__ will assign world_size to PP * TP, while
        # we want PP * TP * SP to be the world size. So we define world_size as
        # a property with a no-op setter to ignore the value later assigned by
        # ParallelConfig.__post_init__.
        pass


class ParallelConfigPatch(PatchHelper[ParallelConfig]):
    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticParallelConfig instead of a
        # ParallelConfig when creating a new instance of the class.
        if cls is ParallelConfig:
            return AscendParallelConfig.__new__(AscendParallelConfig,
                                                *args, **kwargs)
        return super(ParallelConfig, cls).__new__(cls)
    

class VllmConfigPatch(PatchHelper[VllmConfig]):
    _orig_str = VllmConfig.__str__

    def __str__(self, *args, **kwargs):
        string = self._orig_str(*args, **kwargs):
        string += f", ulysses_sequence_parallel_size={self.ulysses_sequence_parallel_size}"
        return string
