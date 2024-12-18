use crate::{
    kernels::{Kernel, NamedKernel},
    network::{CollectiveType, NamedCollective},
    ops::ComputeUnit,
    sharding::{SeqModelSpec, ShardStrategy, ShardingType},
    utils::ValidationError,
};

use super::Operation;

pub struct AttentionOp {
    pub n_q_heads: u64,
    pub n_kv_heads: u64,
}

impl Operation for AttentionOp {
    fn forward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        downstream_collective: Option<NamedCollective>, // This is normally an all-gather that the next layer needs
    ) -> (Vec<super::ComputeUnit>, Option<NamedCollective>) {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let batch_size = axes.batch / batch;
        let sequence_length = axes.sequence / sequence; // TODO: how do I do this here lol
        let head_dim = axes.feature / self.n_q_heads;
        let kv_heads = self.n_kv_heads / feature;
        let query_heads = self.n_q_heads / feature;
        let qkv_out_size = (query_heads + 2 * kv_heads) * head_dim;
        let attn_out_size = query_heads * head_dim;

        let compute_units = vec![
            ComputeUnit::new(
                vec![
                    // TODO: layernorm
                    NamedKernel::matmul(
                        "input * w_qkv",
                        batch_size * sequence_length,
                        axes.feature,
                        qkv_out_size,
                    ),
                    // TODO: transposes
                    NamedKernel::new(
                        "attn forward",
                        Kernel::FlashAttentionFwd {
                            b: batch_size,
                            s: sequence_length,
                            kv_heads,
                            query_heads,
                            head_dim,
                        },
                    ),
                ],
                strategy.named_collective(
                    // TODO: Maybe w_out and w1 of the next linear layer should be in the same collective?
                    "w_out",
                    ShardingType::Data,
                    CollectiveType::AllGather,
                    2 * axes.feature * axes.feature / feature / batch,
                ),
            ),
            ComputeUnit::single(
                NamedKernel::matmul(
                    "attn out * w_out",
                    batch_size * sequence_length,
                    attn_out_size,
                    axes.feature,
                ),
                downstream_collective,
            ),
            ComputeUnit::conly(strategy.named_collective(
                "output acts reduce",
                ShardingType::Tensor,
                CollectiveType::AllReduce,
                batch_size * sequence_length * axes.feature,
            )),
        ];

        let w_qkv_gather = strategy.named_collective(
            "w_qkv gather",
            ShardingType::Data,
            CollectiveType::AllGather,
            2 * axes.feature * qkv_out_size / batch,
        );

        return (compute_units, w_qkv_gather);
    }

    fn backward(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
        upstream_collective: Option<NamedCollective>, // This is normally an all-reduce of the previous step's gradients
    ) -> (Vec<super::ComputeUnit>, Option<NamedCollective>) {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();

        let batch_size = axes.batch / batch;
        let sequence_length = axes.sequence / sequence; // TODO: how do I do this here lol
        let head_dim = axes.feature / self.n_q_heads;
        let kv_heads = self.n_kv_heads / feature;
        let query_heads = self.n_q_heads / feature;
        let qkv_out_size = (query_heads + 2 * kv_heads) * head_dim;
        let attn_out_size = query_heads * head_dim;

        let compute_units = vec![
            ComputeUnit::new(
                vec![
                    // input gradients
                    // upstream grads have shape [batch_size * sequence_length, feature]
                    NamedKernel::matmul(
                        "post flash grads = upstream_grad * w_out^T",
                        batch_size * sequence_length,
                        axes.feature,
                        attn_out_size,
                    ),
                    // attn out has shape [batch_size * sequence_length, attn_out_size]
                    NamedKernel::matmul(
                        "w_out grads = upstream_grad^T * attn out",
                        axes.feature,
                        batch_size * sequence_length,
                        attn_out_size,
                    ),
                    NamedKernel::new(
                        "pre flash grads = flash bwd(post flash grads)",
                        Kernel::FlashAttentionBwd {
                            b: batch_size,
                            s: sequence_length,
                            kv_heads,
                            query_heads,
                            head_dim,
                        },
                    ),
                    // pre flash grads have shape [batch_size * sequence_length, qkv_out_size]
                    NamedKernel::matmul(
                        "input grads = pre flash grads * w_qkv^T",
                        batch_size * sequence_length,
                        qkv_out_size,
                        axes.feature,
                    ),
                ],
                upstream_collective,
            ),
            ComputeUnit::single(
                // Input has shape [batch_size * sequence_length, feature]
                NamedKernel::matmul(
                    "w_qkv grads = input^T * pre flash grads",
                    axes.feature,
                    batch_size * sequence_length,
                    qkv_out_size,
                ),
                strategy.named_collective(
                    "w_out gather",
                    ShardingType::Data,
                    CollectiveType::AllGather,
                    2 * axes.feature * qkv_out_size / batch,
                ),
            ),
        ];

        let w_qkv_gather = strategy.named_collective(
            "w_qkv gather",
            ShardingType::Data,
            CollectiveType::AllGather,
            2 * axes.feature * qkv_out_size / batch,
        );

        return (compute_units, w_qkv_gather);
    }

    fn memory_bytes(&self, axes: &SeqModelSpec, strategy: &ShardStrategy) -> super::MemoryProfile {
        let SeqModelSpec {
            batch,
            sequence,
            feature,
            ..
        } = strategy.axis_splits();
        let batch_size = axes.batch / batch;
        let sequence_length = axes.sequence / sequence; // TODO: how do I do this here lol
        let head_dim = axes.feature / self.n_q_heads;
        let kv_heads = self.n_kv_heads / feature;
        let query_heads = self.n_q_heads / feature;
        let qkv_out_size = (query_heads + 2 * kv_heads) * head_dim;
        let attn_out_size = query_heads * head_dim;
        let input_act_size = batch_size * sequence_length * axes.feature;
        let output_act_size = batch_size * sequence_length * axes.feature;
        //println!("feature: {}, input_act_size: {}, attn_out_size: {}", feature, input_act_size, attn_out_act_size);
        return super::MemoryProfile {
            weight_size: axes.feature * (qkv_out_size + attn_out_size),
            activation_size: input_act_size.max(qkv_out_size).max(output_act_size),
            cache_for_backprop: input_act_size + qkv_out_size + output_act_size,
            input_act_size,
            output_act_size,
        };
    }

    fn validate(
        &self,
        axes: &SeqModelSpec,
        strategy: &ShardStrategy,
    ) -> Result<(), ValidationError> {
        let SeqModelSpec { feature, .. } = strategy.axis_splits();

        if self.n_q_heads % feature != 0 {
            return Err(ValidationError::InvalidQHeadSplit(self.n_q_heads, feature));
        }
        if self.n_kv_heads % feature != 0 {
            return Err(ValidationError::InvalidKVHeadSplit(
                self.n_kv_heads,
                feature,
            ));
        }
        Ok(())
    }
}
