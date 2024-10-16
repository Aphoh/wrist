#[derive(Clone)]
pub struct SeqModelSpec {
    pub batch: u32,
    pub sequence: u32,
    pub feature: u32,
    pub layers: u32,
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

    pub fn top_tier_for(&self, sharding: ShardingType) -> Option<u32> {
        self.pieces
            .as_ref()
            .iter()
            .rposition(|&x| x == sharding)
            .map(|x| x as u32)
    }

    pub fn num_accelerators_for_type(&self, sharding: ShardingType) -> Option<u32> {
        self.top_tier_for(sharding).map(|x| 2u32.pow(x + 1))
    }
}
