use std::marker::PhantomData;

pub struct CombinationsWithReplacement<'a, T> {
    pool: &'a [T],
    n: usize,
    r: usize,
    indices: Option<Vec<usize>>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: Clone> Iterator for CombinationsWithReplacement<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // If indices are None, iteration is over
        let indices = match &mut self.indices {
            Some(indices) => indices,
            None => return None,
        };

        // Generate the current combination
        let result = indices.iter().map(|&i| self.pool[i].clone()).collect();

        // Now advance indices
        let mut i = self.r;
        let mut found = false;
        while i > 0 {
            i -= 1;
            if indices[i] != self.n - 1 {
                // Found an index to increment
                indices[i] += 1;
                let val = indices[i];
                for j in (i + 1)..self.r {
                    indices[j] = val;
                }
                found = true;
                break;
            }
        }
        if !found {
            // All indices are at maximum, iteration is over
            self.indices = None;
        }

        Some(result)
    }
}

pub fn combinations_with_replacement<'a, T>(
    pool: &'a [T],
    r: usize,
) -> CombinationsWithReplacement<'a, T> {
    let n = pool.len();
    let indices = if n == 0 && r > 0 {
        None
    } else {
        Some(vec![0; r])
    };
    CombinationsWithReplacement {
        pool,
        n,
        r,
        indices,
        _marker: PhantomData,
    }
}

pub fn n_combinations(n: usize, r: usize) -> u128 {
    let mut num = 1u128;
    let mut denom = 1u128;
    for i in 1..r {
        num *= (n + i - 1) as u128;
        denom *= i as u128;
    }
    num / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_case() {
        let numbers = [0, 1, 2];
        let k = 4;
        let iter = combinations_with_replacement(&numbers, k);

        let expected = vec![
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 1],
            vec![0, 0, 0, 2],
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 2],
            vec![0, 0, 2, 2],
            vec![0, 1, 1, 1],
            vec![0, 1, 1, 2],
            vec![0, 1, 2, 2],
            vec![0, 2, 2, 2],
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 2],
            vec![1, 1, 2, 2],
            vec![1, 2, 2, 2],
            vec![2, 2, 2, 2],
        ];

        let result: Vec<Vec<i32>> = iter.collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_pool() {
        let numbers: [i32; 0] = [];
        let k = 3;
        let iter = combinations_with_replacement(&numbers, k);

        let result: Vec<Vec<i32>> = iter.collect();
        let expected: Vec<Vec<i32>> = vec![];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_zero_length_combination() {
        let numbers = [1, 2, 3];
        let k = 0;
        let iter = combinations_with_replacement(&numbers, k);

        let result: Vec<Vec<i32>> = iter.collect();
        let expected = vec![vec![]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_single_element_pool() {
        let numbers = [42];
        let k = 3;
        let iter = combinations_with_replacement(&numbers, k);

        let expected = vec![vec![42, 42, 42]];
        let result: Vec<Vec<i32>> = iter.collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_general_case() {
        let numbers = ['A', 'B', 'C'];
        let k = 2;
        let iter = combinations_with_replacement(&numbers, k);

        let expected = vec![
            vec!['A', 'A'],
            vec!['A', 'B'],
            vec!['A', 'C'],
            vec!['B', 'B'],
            vec!['B', 'C'],
            vec!['C', 'C'],
        ];

        let result: Vec<Vec<char>> = iter.collect();
        assert_eq!(result, expected);
    }
}