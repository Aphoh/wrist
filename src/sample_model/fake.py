# mypy: allow-untyped-defs

import random
import torch
import torch.distributed as dist

from torch._C._distributed_c10d import (
    FakeProcessGroup,
    FakeWork,
)

# Sets up fake collective impls
from torch.distributed._tools import fake_collectives


class FakeStore(dist.Store):
    """
    A fake store is a fake Key-Value store simply for initialization usage
    the of fake process group, one can either use FakeStore or HashStore.
    """


def _create_fake_pg(prefix_store, rank, world_size, timeout):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convinient tool when playing
    with distributed but don't care about the actual data.
    """
    return FakeProcessGroup(rank, world_size)


dist.Backend.register_backend("fake", _create_fake_pg, devices=['cpu', 'cuda'])

used_ids: set[int] = set()
def generate_unique_id() -> int:
    global used_ids
    while True:
        new_id = random.randint(1, 10**9)
        if new_id not in used_ids:
            used_ids.add(new_id)
            return new_id

# Function to create and return FakeWork object
def create_fakework(args, return_first_arg=True):  # type: ignore[no-untyped-def]
    work = FakeWork()
    work.seq_id = generate_unique_id()
    fakework_script_obj = work.boxed()
    return (args[0], fakework_script_obj) if return_first_arg else fakework_script_obj
