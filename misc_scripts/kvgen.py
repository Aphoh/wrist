def generate_values(max_f, max_q, max_kv):
    """
    Generate all possible values of (q + 2*kv)(max_f/max_q) where:
    - q <= max_q
    - kv <= max_kv
    - kv <= q
    - q, kv are powers of 2
    
    Args:
        max_f (int): Maximum f value (must be power of 2)
        max_q (int): Maximum q value (must be power of 2)
        max_kv (int): Maximum kv value (must be power of 2)
    
    Returns:
        list: Sorted list of unique values
    """
    # Verify inputs are powers of 2
    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0
    
    if not all(is_power_of_2(x) for x in [max_f, max_q, max_kv]):
        raise ValueError("All inputs must be powers of 2")
    
    # Generate powers of 2 up to max values
    q_values = [2**i for i in range(int.bit_length(max_q)) if 2**i <= max_q]
    kv_values = [2**i for i in range(int.bit_length(max_kv)) if 2**i <= max_kv]
    
    results = set()
    multiplier = max_f // max_q
    
    for q in q_values:
        # Only consider kv values that are <= q and <= max_kv
        valid_kv = [kv for kv in kv_values if kv <= q and kv <= max_kv]
        for kv in valid_kv:
            value = (q + 2 * kv) * multiplier
            results.add(value)
    
    return sorted(list(results))

# Example usage:
def print_results(max_f, max_q, max_kv):
    print(f"Results for max_f={max_f}, max_q={max_q}, max_kv={max_kv}:")
    results = generate_values(max_f, max_q, max_kv)
    print(f"Number of values: {len(results)}")
    print("Values:", results)
    print()

# Test cases
print_results(32, 16, 8)
print_results(64, 32, 16)