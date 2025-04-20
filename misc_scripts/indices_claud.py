from itertools import combinations_with_replacement, permutations
from collections import Counter

def sorted_combination_permutations(iterable, r):
    """
    Generate r-length combinations with replacement from iterable,
    including all sorting permutations of each combination.
    
    Elements in each output tuple maintain relative position order
    based on first appearance in the input iterable.
    
    Example:
        sorted_combination_permutations('AB', 3) ->
        AAA AAB ABB BBB BBA BAA
    """
    # First get all combinations with replacement
    combinations = combinations_with_replacement(iterable, r)
    sort_orders = list(permutations(set(iterable)))
    
    # For each combination, generate unique permutations
    for combo in combinations:
        # Generate unique permutations considering repeats
        # We use set() to avoid duplicates in case the input has repeats
        res = set()
        for sort_order in sort_orders:
            res.add(tuple(sorted(combo, key=lambda x: sort_order.index(x))))
        yield from res

# Example usage
if __name__ == "__main__":
    # Test with the example case
    test_input = 'ABC'
    r = 3
    result = list(sorted_combination_permutations(test_input, r))
    print(f"Input: {test_input}, r={r}")
    print("Output:", ' '.join(''.join(x) for x in result))
    
    # Additional test case with repeated elements
    test_input2 = 'AA'
    result2 = list(sorted_combination_permutations(test_input2, 2))
    print(f"\nInput: {test_input2}, r=2")
    print("Output:", ' '.join(''.join(x) for x in result2))