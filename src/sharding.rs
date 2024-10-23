use crate::network::{Collective, CollectiveType};

#[derive(Clone)]
pub struct SeqModelSpec {
    pub batch: u64,
    pub sequence: u64,
    pub feature: u64,
    pub layers: u64,
}

impl SeqModelSpec {
    pub fn expand_by(&mut self, sharding: ShardingType) -> &mut Self {
        match sharding {
            ShardingType::Data => self.batch *= 2,
            ShardingType::Pipeline => self.layers *= 2,
            ShardingType::Sequence => self.sequence *= 2,
            ShardingType::Tensor => self.feature *= 2,
        }
        self
    }

    pub fn split_by(&self, other: &Self) -> Self {
        Self {
            batch: self.batch / other.batch,
            sequence: self.sequence / other.sequence,
            feature: self.feature / other.feature,
            layers: self.layers / other.layers,
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub enum ShardingType {
    Data = 10,
    Pipeline = 20,
    Sequence = 30,
    Tensor = 40,
}

impl ShardingType {
    pub fn short_name(&self) -> &'static str {
        match self {
            ShardingType::Data => "D",
            ShardingType::Pipeline => "P",
            ShardingType::Sequence => "S",
            ShardingType::Tensor => "T",
        }
    }
}

pub trait ShardSpec: AsMut<[ShardingType]> + AsRef<[ShardingType]> {}
impl<M: AsMut<[ShardingType]> + AsRef<[ShardingType]>> ShardSpec for M {}

#[derive(Debug, Clone)]
pub struct ShardStrategy<M: ShardSpec> {
    // This is a sorted sequence of sharding types
    // The order of the types is the order of the tiers
    // So the 0 element represents the leaf tier
    pub pieces: M,
}

impl<M: ShardSpec> std::fmt::Display for ShardStrategy<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for piece in self.pieces.as_ref() {
            write!(f, "{}", piece.short_name())?;
        }
        Ok(())
    }
}

impl<M: ShardSpec> ShardStrategy<M> {
    pub fn new(types: M) -> Option<Self> {
        types
            .as_ref()
            .windows(2)
            .all(|w| w[0] >= w[1])
            .then(|| Self::new_unchecked(types))
    }

    pub fn new_unchecked(types: M) -> Self {
        Self { pieces: types }
    }

    pub fn axis_splits(&self) -> SeqModelSpec {
        let mut spec = SeqModelSpec {
            batch: 1,
            sequence: 1,
            feature: 1,
            layers: 1,
        };
        for piece in self.pieces.as_ref() {
            spec.expand_by(*piece);
        }
        spec
    }

    
    pub fn num_gpus_in_sharding_groups(&self, sharding: ShardingType) -> Option<u32> {
        // This is just 2^the number of times it appears in the sharding group
        let count: u32 = self.pieces.as_ref().iter().map(|x| (*x == sharding) as u32).sum();
        (count > 0).then(|| 1 << count)
    }

    pub fn stride_of_sharding_groups(&self, sharding: ShardingType) -> u32 {
        // This is 2^the number of elements that come before the first instance of the sharding type
        let mut stride = 1;
        for piece in self.pieces.as_ref() {
            if piece != &sharding {
                stride *= 2;
            } else {
                break;
            }
        }
        stride
    }

    pub fn collective(&self, stype: ShardingType, ctype: CollectiveType, piece_bytes: u64) -> Option<Collective> {
        let n_gpus = self.num_gpus_in_sharding_groups(stype)?;
        let stride = self.stride_of_sharding_groups(stype);
        Some(Collective {
            ctype,
            stride,
            piece_bytes,
            n_gpus,
        })
    }
}
