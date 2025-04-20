
def _repeated_subgroup(group_size, repeats, base=0):
    """ 
        Outputs the sequence
        [base, base+1, base+2, ..., base+group_size-1] 
        repeated `repeats` times
    """
    repeated = [base + i for i in range(group_size)] 
    return [a for _ in range(repeats) for a in repeated]

def strided_indices(world_size, stride, group_size):
    num_groups = world_size // group_size
    group_idxs = []
    strided_subgroup_size = group_size * stride
    n_repeats = strided_subgroup_size // stride
    for base in range(0, num_groups, stride):
        subgroup = _repeated_subgroup(stride, n_repeats, base)
        group_idxs.extend(subgroup)
    # Map group_idxs back to ranks
    groups = [[] for _ in range(num_groups)]
    for i, idx in enumerate(group_idxs):
        groups[idx].append(i)
    return groups


def _check_eq(a, b):
    assert a == b, f"{a} != {b}"


def main():
    _check_eq(strided_indices(8, 1, 4), [[0, 1, 2, 3], [4, 5, 6, 7]])
    _check_eq(strided_indices(8, 2, 4), [[0, 2, 4, 6], [1, 3, 5, 7]])
    _check_eq(strided_indices(8, 2, 2), [[0, 2], [1, 3], [4, 6], [5, 7]])
    _check_eq(strided_indices(16, 2, 4), [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]])
    print("Passed")


if __name__ == "__main__":
    main()