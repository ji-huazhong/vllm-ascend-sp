import argparse
from dataclasses import dataclass

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils import FlexibleArgumentParser

from .utils import PatchHelper


@dataclass
class AscendArgs:
    ulysses_sequence_parallel_size: int = 1


@dataclass
class AscendEngineArgs(EngineArgs, AscendArgs):
    pass


@dataclass
class AscendAsyncEngineArgs(AsyncEngineArgs, AscendArgs):
    pass


_current_ascend_args: AscendArgs = None


def get_current_ascend_args() -> AscendArgs:
    return _current_ascend_args


class EngineArgsPatch(PatchHelper[EngineArgs]):
    _orig_add_cli_args = EngineArgs.add_cli_args
    # Bypassing the Descriptor Protocol
    _orig_from_cli_args = EngineArgs.__dict__["from_cli_args"]
    _orig_create_engine_config = EngineArgs.create_engine_config

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = EngineArgsPatch._origin_add_cli_args(parser)
        parser.add_argument(
            "--ulysses-sequence-parallel-size",
            type=int,
            default=AscendEngineArgs.ulysses_sequence_parallel_size,
            help="Number of ulysses sequence parallel replicas",
        )
        return parser
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        if cls is EngineArgs:
            return EngineArgsPatch._orig_from_cli_args(AscendEngineArgs, args)
        if cls is AsyncEngineArgs:
            return EngineArgsPatch._orig_from_cli_args(AscendAsyncEngineArgs, args)
        return EngineArgsPatch._orig_from_cli_args(cls, args)

    def create_engine_config(self, *args, **kwargs):
        # Temporarily makes the engine args available as a global variable when
        # running this method so that the customized config classes can grab their
        # values during initialization.
        global _current_ascend_args
        try:
            _current_ascend_args = self
            return self._orig_create_engine_config(*args, **kwargs)
        finally:
            _current_ascend_args = None

class AsyncEngineArgsPatch(PatchHelper[AsyncEngineArgs]):

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticAsyncEngineArgs instead of an
        # AsyncEngineArgs when creating a new instance of the class.
        if cls is AsyncEngineArgs:
            return AscendAsyncEngineArgs.__new__(AscendAsyncEngineArgs,
                                                 *args, **kwargs)
        return super(AsyncEngineArgs, cls).__new__(cls)
