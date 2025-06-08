import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.functional import scaled_dot_product_attention
from torch.distributed.device_mesh import DeviceMesh

from sample_model.parallel_dims import ParallelDims, apply_ddp


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer (Megatron-style)"""

    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.weight = nn.Parameter(torch.randn(out_features // self.tp_size, in_features))

    def forward(self, x):
        # No communication needed for column parallel
        return torch.nn.functional.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """Row-parallel linear layer (Megatron-style)"""

    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features // self.tp_size)
        )

    def forward(self, x):
        # Compute local matmul
        local_out = torch.nn.functional.linear(x, self.weight)

        if self.tp_group is not None:
            # Async all-reduce across tensor parallel group
            handle = dist.all_reduce(local_out, async_op=True, group=self.tp_group)
            # Can overlap other computation here if needed
            handle.wait()

        return local_out


class ParallelAttention(nn.Module):
    """Multi-head attention with tensor parallelism across heads"""

    def __init__(self, d_model, n_heads, tp_group):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.tp_group = tp_group
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.n_heads_per_rank = n_heads // tp_size
        self.head_dim = d_model // n_heads

        # Q, K, V projections (column parallel)
        self.qkv_proj = ColumnParallelLinear(d_model, 3 * d_model, tp_group)

        # Output projection (row parallel)
        self.out_proj = RowParallelLinear(d_model, d_model, tp_group)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Get Q, K, V (each rank gets subset of heads)
        qkv = self.qkv_proj(x)  # [B, L, 3 * d_model // tp_size]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads_per_rank, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Compute attention for local heads
        q = q.transpose(1, 2)  # [B, n_heads_per_rank, L, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention using PyTorch's memory-efficient implementation
        # scale is handled internally by scaled_dot_product_attention
        attn_out = scaled_dot_product_attention(q, k, v)

        # Reshape for output projection
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # Row-parallel output projection (includes all-reduce)
        out_proj = self.out_proj(attn_out)
        if self.tp_group is not None:
            # Async all-reduce across tensor parallel group
            handle = dist.all_reduce(out_proj, async_op=True, group=self.tp_group)
            # Can overlap other computation here if needed
            handle.wait()
        return out_proj


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, tp_group):
        super().__init__()
        self.attention = ParallelAttention(d_model, n_heads, tp_group)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Megatron-style FFN: column-parallel then row-parallel
        self.ffn = nn.Sequential(
            ColumnParallelLinear(d_model, d_model * 4, tp_group),
            nn.ReLU(),
            RowParallelLinear(d_model * 4, d_model, tp_group),
        )

    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class AsyncTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, tp_group):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, tp_group) for _ in range(n_layers)]
        )
        self.output_proj = ColumnParallelLinear(d_model, vocab_size, tp_group)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        # FSDP automatically prefetches next layer parameters
        # while current layer computes (due to forward_prefetch=True)
        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


def setup_fsdp_model(model: AsyncTransformer, parallel_dims: ParallelDims, device_mesh: DeviceMesh):
    """Setup FSDP with auto-wrapping and async prefetching"""
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }
    if parallel_dims.dp_enabled:
        dim_names = []
        if parallel_dims.dp_replicate_enabled:
            dim_names.append("dp_replicate")
        if parallel_dims.dp_shard_enabled:
            dim_names.append("dp_shard")

        dp_mesh = device_mesh[tuple(dim_names)]

        if parallel_dims.dp_shard_enabled:
            for layer in model.layers:
                fully_shard(layer, mesh=dp_mesh, **fsdp_kwargs)

            fully_shard(model, mesh=dp_mesh, **fsdp_kwargs)
        elif parallel_dims.dp_replicate_enabled:
            apply_ddp(model, dp_mesh=dp_mesh)
    return model


@torch.compile(fullgraph=True, dynamic=False)
def training_step(model, input_ids, labels, optimizer):
    """Compiled training step with forward/backward pass"""
    # Forward pass
    logits = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Backward pass with FSDP gradient synchronization
    optimizer.zero_grad()
    loss.backward()  # FSDP handles gradient sync automatically

    # Gradient clipping across shards
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step
    optimizer.step()

    return loss
