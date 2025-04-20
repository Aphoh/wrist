use std::collections::BTreeSet;

use itertools::Itertools;

pub struct CombinationsWithReplacement<T> {
    pool: Vec<T>,
    n: usize,
    r: usize,
    indices: Option<Vec<usize>>,
}

impl<'a, T: Clone> Iterator for CombinationsWithReplacement<T> {
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

pub struct PermutedIndCombinationsWithReplacement<'a, T> {
    pool: &'a [T],
    sort_orders: Vec<Vec<usize>>,
    comb_iter: CombinationsWithReplacement<usize>,
    curr_elems: Vec<Vec<usize>>,
}

impl<'a, T: Clone> Iterator for PermutedIndCombinationsWithReplacement<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_elems.is_empty() {
            let combination = self.comb_iter.next()?;
            let mut result = BTreeSet::new();
            for sort_order in &self.sort_orders {
                let permuted_combo = combination
                    .clone()
                    .into_iter()
                    .sorted_by_key(|&i| sort_order[i])
                    .collect_vec();
                result.insert(permuted_combo);
            }
            self.curr_elems = result.into_iter().collect();
        }
        self.curr_elems
            .pop()
            .map(|inds| inds.into_iter().map(|i| self.pool[i].clone()).collect())
    }
}

pub fn permuted_combinations_with_replacement<'a, T>(
    pool: &'a [T],
    r: usize,
) -> PermutedIndCombinationsWithReplacement<'a, T>
where
    T: Clone,
{
    let inds = (0..pool.len()).collect_vec();
    let sort_orders = (0..inds.len())
        .into_iter()
        .permutations(inds.len())
        .collect_vec();
    let comb_iter = combinations_with_replacement(inds, r);
    PermutedIndCombinationsWithReplacement {
        pool,
        sort_orders,
        comb_iter,
        curr_elems: Default::default(),
    }
}

pub fn combinations_with_replacement<T>(pool: Vec<T>, r: usize) -> CombinationsWithReplacement<T> {
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
    }
}

pub fn n_repeated_combinations(n: usize, k: usize) -> u128 {
    let mut res = 0u128;
    for t in 1..(k + 1) {
        res += factorial(t) * binomial_coefficient(n - 1, t - 1) * binomial_coefficient(k, t);
    }
    res
}

pub fn factorial(n: usize) -> u128 {
    (1..=n).fold(1, |acc, x| acc * x as u128)
}

pub fn binomial_coefficient(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) as u128 / (i + 1) as u128;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_case() {
        let numbers = vec![0, 1, 2];
        let k = 4;
        let iter = combinations_with_replacement(numbers, k);

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
        let numbers: Vec<i32> = vec![];
        let k = 3;
        let iter = combinations_with_replacement(numbers, k);

        let result: Vec<Vec<i32>> = iter.collect();
        let expected: Vec<Vec<i32>> = vec![];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_zero_length_combination() {
        let numbers = vec![1, 2, 3];
        let k = 0;
        let iter = combinations_with_replacement(numbers, k);

        let result: Vec<Vec<i32>> = iter.collect();
        let expected : Vec<Vec<i32>> = vec![vec![]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_single_element_pool() {
        let numbers = vec![42];
        let k = 3;
        let iter = combinations_with_replacement(numbers, k);

        let expected = vec![vec![42, 42, 42]];
        let result: Vec<Vec<i32>> = iter.collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_general_case() {
        let numbers = vec!['A', 'B', 'C'];
        let k = 2;
        let iter = combinations_with_replacement(numbers, k);

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

    #[test]
    fn test_permuted_combinations_with_replacement() {
        let inds = [0, 1, 2];
        let k = 3;
        let iter = permuted_combinations_with_replacement(&inds, k);

        macro_rules! c {
            ($($x:expr)*) => {
                vec![$($x),*]
            };
        }

        // We have 0 1 2, 0 2 1, 1 0 2, 1 2 0, 2 0 1, 2 1 0

        let expected: BTreeSet<Vec<usize>> = [
            // First sorting order 0 1 2:
            c!(0 0 0),
            c!(0 0 1),
            c!(0 0 2),
            c!(0 1 1),
            c!(0 1 2),
            c!(0 2 2),
            c!(1 1 1),
            c!(1 1 2),
            c!(1 2 2),
            c!(2 2 2),
            // Second sorting order 0 2 1:
            c!(0 2 1),
            c!(2 1 1),
            c!(2 2 1),
            // Third sorting order 1 0 2:
            c!(1 0 2),
            c!(1 1 0),
            c!(1 0 0),
            // Fourth sorting order 1 2 0:
            c!(1 2 0),
            c!(2 0 0),
            c!(2 2 0),
            // Fifth sorting order 2 0 1:
            c!(2 0 1),
            // Sixth sorting order 2 1 0:
            c!(2 1 0),
        ]
        .into_iter()
        .collect();

        let result: BTreeSet<Vec<usize>> = iter.collect();
        let expected_not_result = expected.difference(&result).collect_vec();
        let result_not_expected = result.difference(&expected).collect_vec();
        assert_eq!(
            result, expected,
            "Expected not in result: {:?}, Result not in expected: {:?}",
            expected_not_result, result_not_expected
        );

        assert_eq!(
            result.len(),
            n_repeated_combinations(inds.len(), k) as usize
        )
    }
}
