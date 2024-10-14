/// Symmetric network tree
/// 0-index tier is considered the leaf tier
pub struct Network<CollectiveMeasurer> {
    n_tiers: u32,
    measurer: CollectiveMeasurer,
}

impl<M: CollectiveMeasurer> Network<M> {
    pub fn new(n_tiers: u32, measurer: M) -> Self {
        Network { n_tiers, measurer }
    }

    pub fn n_tiers(&self) -> u32 {
        self.n_tiers
    }

    pub fn accelerators_by_tier(&self) -> Vec<u32> {
        (0..self.n_tiers).map(|i| 2u32.pow(i as u32)).collect()
    }

    pub fn num_accelerators(&self, tier: u32) -> u32 {
        return 2u32.pow(tier);
    }

    pub fn duration_ms<C: AsRef<[Collective]>>(&self, collectives: C) -> u32 {
        self.measurer.measure(collectives)
    }
}

pub trait CollectiveMeasurer {
    fn measure<C: AsRef<[Collective]>>(&self, collectives: C) -> u32;
}

pub struct NaiveCollectiveMeasurer;
impl CollectiveMeasurer for NaiveCollectiveMeasurer {
    fn measure<C: AsRef<[Collective]>>(&self, collectives: C) -> u32 {
        collectives
            .as_ref()
            .iter()
            .map(|c| match c {
                Collective::AllGather { piece_bytes, tier } => 64 * piece_bytes * (tier + 1),
                Collective::Reduce { tier } => 64 * tier,
                Collective::Broadcast { tier } => 64 * 2u32.pow(*tier),
            })
            .max() // TODO: Is this a reasonable assumption? Should probably care about bandwidth.
            .unwrap_or(0u32)
    }
}

#[derive(Eq, PartialEq, Debug, Hash, Clone, PartialOrd, Ord)]
pub enum Collective {
    AllGather { piece_bytes: u32, tier: u32 },
    Reduce { tier: u32 },
    Broadcast { tier: u32 },
}

impl Collective {
    pub fn all_gather(piece_bytes: u32, tier: u32) -> Self {
        Collective::AllGather { piece_bytes, tier }
    }
}
