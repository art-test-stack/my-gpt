from gpt_lib.utils.schemas import ParallelismConfig

def init_process_groups(world_size, tp_size):
    assert world_size % tp_size == 0
    dp_size = world_size // tp_size

    rank = dist.get_rank()

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    tp_group = dist.new_group(
        ranks=[dp_rank * tp_size + i for i in range(tp_size)]
    )

    dp_group = dist.new_group(
        ranks=[i * tp_size + tp_rank for i in range(dp_size)]
    )

    return dp_group, tp_group, dp_rank, tp_rank, dp_size


def choose_parallelism(world_size: int, tp_size: int) -> ParallelismConfig:
    config = dict(
        world_size=world_size,
        tp_size=tp_size,
        dp_size=world_size // tp_size,
        
    )
    if tp_size > 1 and world_size // tp_size > 1:
        return "dp_tp"
    elif tp_size > 1:
        return "tp"
    elif world_size > 1:
        return "dp"
    else:
        return "single"