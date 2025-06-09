from collections.abc import Callable
from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch import nn

__all__ = ["ParallelDims"]


@dataclass
class ParallelDims:
    dp_shard: int
    tp: int
    world_size: int
    enable_loss_parallel: bool = False

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_shard, tp = (
            self.dp_shard,
            self.tp,
        )
        for d in (dp_shard, tp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard >= 1

        assert dp_shard * tp == self.world_size, (
            f"Invalid parallel dims: * dp_shard({dp_shard}) * "
            f"tp({tp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self) -> DeviceMesh:
        dims = []
        names = []
        for d, name in zip(
            [self.dp_shard, self.tp],
            ["dp_shard", "tp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        return self._build_mesh("cuda", dims, names, init_device_mesh)

    def _build_mesh(
        self,
        device_type: str,
        dims: list[int],
        names: list[str],
        init_device_mesh_fn: Callable,
    ) -> DeviceMesh:
        mesh = init_device_mesh_fn(device_type, dims, mesh_dim_names=names)

        # TODO: can init special groups here if needed
        return mesh

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    def get_tp_mesh(self, mesh: DeviceMesh) -> DeviceMesh | None:
        if not self.tp_enabled:
            return None
        return mesh[("tp",)] if "tp" in mesh.mesh_dim_names else None
