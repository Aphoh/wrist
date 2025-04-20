use crate::{
    network::Collective,
    sharding::{SeqModelSpec, ShardStrategy},
    utils::ValidationError,
};

use super::Operation;

pub struct DoubleOp<O1, O2> {
    pub op1: O1,
    pub op2: O2,
}

impl<O1, O2> Operation for DoubleOp<O1, O2>
where
    O1: Operation,
    O2: Operation,
{
    fn forward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        downstream_collective: Option<Collective>,
    ) -> (Vec<super::ComputeUnit>, Option<Collective>) {
        // 2nd operation takes the downstream collective
        let (cu2, coll2) = self.op2.forward(axes, strategy, downstream_collective);
        // 1st operation takes the collective from the 2nd operation
        let (mut cu1, coll1) = self.op1.forward(axes, strategy, coll2);
        cu1.extend(cu2);
        // Return the collective from the 1st operation
        (cu1, coll1)
    }

    fn backward(
        &self,
        axes: &crate::sharding::SeqModelSpec,
        strategy: &crate::sharding::ShardStrategy,
        upstream_collective: Option<crate::network::Collective>, // This is normally an all-reduce of the previous step's gradients
    ) -> (
        Vec<super::ComputeUnit>,
        Option<crate::network::Collective>,
    ) {
        // cu2 executes first, does the collective from upstream
        let (mut cu2, coll2) = self.op2.backward(axes, strategy, upstream_collective);
        let (cu1, coll1) = self.op1.backward(axes, strategy, coll2);
        cu2.extend(cu1);
        (cu2, coll1)
    }

    fn memory_bytes(
        &self,
        axes: &crate::sharding::SeqModelSpec,
        strategy: &crate::sharding::ShardStrategy,
    ) -> super::MemoryProfile {
        self.op1
            .memory_bytes(axes, strategy)
            .combine(&self.op2.memory_bytes(axes, strategy))
    }

    fn validate(
        &self,
        axes: &crate::sharding::SeqModelSpec,
        strategy: &crate::sharding::ShardStrategy,
    ) -> Result<(), ValidationError> {
        self.op1
            .validate(axes, strategy)
            .and_then(|_| self.op2.validate(axes, strategy))
    }
}
