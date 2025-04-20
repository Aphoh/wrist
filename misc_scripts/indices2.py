import unittest

def custom_method(iterable, r):
    # Convert iterable to a sorted tuple to ensure consistent order
    pool = tuple(sorted(iterable, key=lambda x: str(x)))
    n = len(pool)
    if not n and r:
        return []
    if r == 0:
        return ['']
    indices = [0] * r
    perm_asc_set = set()
    perm_desc_set = set()
    
    while True:
        # Generate the current combination based on indices
        comb = tuple(pool[i] for i in indices)
        
        # Permutation sorted in ascending order based on string representation
        perm_asc = tuple(sorted(comb, key=lambda x: str(x)))
        perm_asc_set.add(perm_asc)
        
        # Permutation sorted in descending order based on string representation
        perm_desc = tuple(sorted(comb, key=lambda x: str(x), reverse=True))
        perm_desc_set.add(perm_desc)
        
        # Move to the next combination
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            break
        indices[i] += 1
        indices[i+1:] = [indices[i]] * (r - i - 1)
    
    # Output permutations sorted ascending in lex order based on string representation
    result = []
    for p in sorted(perm_asc_set, key=lambda x: ''.join(str(i) for i in x)):
        result.append(''.join(str(x) for x in p))
    
    # Output permutations sorted descending in reverse lex order based on string representation
    for p in sorted(perm_desc_set, key=lambda x: ''.join(str(i) for i in x), reverse=True):
        if p not in perm_asc_set:
            result.append(''.join(str(x) for x in p))
    
    return result

class TestCustomMethod(unittest.TestCase):
    def test_example_input(self):
        # Test the example provided: custom_method('AB', 3)
        expected_output = ['AAA', 'AAB', 'ABB', 'BAA', 'BBA', 'BBB']
        result = custom_method('AB', 3)
        self.assertEqual(sorted(result), sorted(expected_output))
    
    def test_single_character(self):
        # Test with a single character iterable
        expected_output = ['AAA']
        result = custom_method('A', 3)
        self.assertEqual(result, expected_output)
    
    def test_empty_iterable(self):
        # Test with an empty iterable
        expected_output = []
        result = custom_method('', 3)
        self.assertEqual(result, expected_output)
    
    def test_zero_length(self):
        # Test with r = 0
        expected_output = ['']
        result = custom_method('AB', 0)
        self.assertEqual(result, expected_output)
    
    def test_longer_iterable(self):
        # Test with a longer iterable
        expected_output = [
            'AA', 'AB', 'AC', 'BA', 'BB', 'BC', 'CA', 'CB', 'CC'
        ]
        result = custom_method('ABC', 2)
        self.assertEqual(sorted(result), sorted(expected_output))
    
    def test_numeric_iterable(self):
        # Test with numeric characters
        expected_output = ['11', '12', '21', '22']
        result = custom_method('12', 2)
        self.assertEqual(sorted(result), sorted(expected_output))
    
    def test_special_characters(self):
        # Test with special characters
        expected_output = ['!!', '!@', '@!', '@@']
        result = custom_method('!@', 2)
        self.assertEqual(sorted(result), sorted(expected_output))
    
    def test_large_r(self):
        # Test with a larger r value
        expected_output = [
            'AAAA', 'AAAB', 'AABB', 'ABBB', 'BAAA', 'BBAA', 'BBBA', 'BBBB'
        ]
        result = custom_method('AB', 4)
        self.assertEqual(sorted(result), sorted(expected_output))
    
    def test_non_string_iterable(self):
        # Test with an iterable of integers
        expected_output = ['111', '112', '122', '211', '221', '222']
        result = custom_method([1, 2], 3)
        self.assertEqual(sorted(result), sorted(expected_output))
    
    def test_duplicate_elements(self):
        # Test with duplicate elements in the iterable
        expected_output = ['AAA', 'AAB', 'ABB', 'BAA', 'BBA', 'BBB']
        result = custom_method('AAB', 3)
        self.assertEqual(sorted(result), sorted(expected_output))
    
    def test_r_greater_than_iterable(self):
        # Test where r is greater than the number of unique elements
        expected_output = ['AAAA', 'AAAB', 'AABB', 'ABBB', 'BAAA', 'BBAA', 'BBBA', 'BBBB']
        result = custom_method('AB', 4)
        self.assertEqual(sorted(result), sorted(expected_output))
    
    def test_r_equals_zero(self):
        # Test with r = 0
        expected_output = ['']
        result = custom_method('ABC', 0)
        self.assertEqual(result, expected_output)
    
    def test_iterable_with_none(self):
        # Test with None in the iterable
        expected_output = ['NoneNone']
        result = custom_method([None], 2)
        self.assertEqual(result, expected_output)
    
    def test_mixed_data_types(self):
        # Test with mixed data types
        expected_output = ['11', '1A', 'A1', 'AA']
        result = custom_method(['A', 1], 2)
        self.assertEqual(sorted(result), sorted(expected_output))


    def test_abcd_length_3(self):
       # Test with 'ABCD' and r = 3
       expected_output = [
           'AAA', 'AAB', 'AAC', 'AAD', 'ABB', 'ABC', 'ABD', 'ACC', 'ACD', 'ADD',
           'BBB', 'BBC', 'BBD', 'BCC', 'BCD', 'BDD',
           'CCC', 'CCD', 'CDD',
           'DDD',
           'DDC', 'DDB', 'DDA',
           'DCC', 'DCB', 'DCA',
           'DBB', 'DBA',
           'DAA',
           'CCB', 'CCA',
           'CBB', 'CBA',
           'CAA',
           'BBA', 'BAA'
       ]
       assert len(set(expected_output)) == len(expected_output)
       result = custom_method('ABCD', 3)
       self.assertEqual(sorted(result), sorted(expected_output))

if __name__ == '__main__':
    unittest.main()
