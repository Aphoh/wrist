import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from typing import Optional, List, Dict, Any
from contextlib import nullcontext
from torch.distributed._functional_collectives import all_gather_tensor

from sample_model.fake import FakeStore
from sample_model.parallel_dims import ParallelDims


class ParameterBucket:
    """Groups parameters for async communication (mimics Megatron's _ParamAndGradBucket)"""

    def __init__(
        self,
        params: List[nn.Parameter],
        bucket_id: int,
        dp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.params = params
        self.bucket_id = bucket_id
        self.dp_group = dp_group

        if dp_group is not None:
            self.dp_size = dist.get_world_size(dp_group)
            self.dp_rank = dist.get_rank(dp_group)
        else:
            self.dp_size = 1
            self.dp_rank = 0

        # Create flattened parameter buffer
        self.param_numel = sum(p.numel() for p in params)
        self.param_data = torch.zeros(
            self.param_numel, dtype=torch.bfloat16
        )

        # Create parameter shards for FSDP
        self.shard_size = self.param_numel // self.dp_size
        self.param_shard = self.param_data[
            self.dp_rank * self.shard_size : (self.dp_rank + 1) * self.shard_size
        ]

        # Map original parameters to views in the buffer
        self._map_params_to_buffer()

        # Async handles
        self.allgather_handle = None
        self.reduce_scatter_handle = None

    def _map_params_to_buffer(self):
        """Map original parameters to views in flattened buffer"""
        offset = 0
        for param in self.params:
            numel = param.numel()
            param_view = self.param_data[offset : offset + numel].view(param.shape)
            #param_view.copy_(param.data)
            param.data = param_view
            offset += numel

    def start_param_allgather(self):
        """Start async all-gather of parameter shards"""
        if self.dp_group is not None and self.dp_size > 1:
            #self.allgather_handle = dist.all_gather_into_tensor(
            #    self.param_data, self.param_shard, group=self.dp_group, async_op=True
            #)
            self.allgather_handle = dist.

    def finish_param_allgather(self):
        """Wait for parameter all-gather to complete"""
        if self.allgather_handle is not None:
            self.allgather_handle.wait()
            self.allgather_handle = None

    def start_grad_reduce_scatter(self):
        """Start async reduce-scatter of gradients"""
        if self.dp_group is not None and self.dp_size > 1:
            # Collect gradients from all parameters in bucket
            grad_data = torch.zeros_like(self.param_data)
            offset = 0
            for param in self.params:
                if param.grad is not None:
                    numel = param.numel()
                    grad_data[offset : offset + numel] = param.grad.flatten()
                    offset += numel

            # Start async reduce-scatter
            grad_shard = torch.zeros_like(self.param_shard)
            self.reduce_scatter_handle = dist.reduce_scatter_tensor(
                grad_shard, grad_data, group=self.dp_group, async_op=True
            )
            return grad_shard
        else:
            # No FSDP, return flattened gradients
            grad_data = torch.zeros_like(self.param_data)
            offset = 0
            for param in self.params:
                if param.grad is not None:
                    numel = param.numel()
                    grad_data[offset : offset + numel] = param.grad.flatten()
                    offset += numel
            return grad_data

    def finish_grad_reduce_scatter(self, grad_shard: torch.Tensor):
        """Wait for gradient reduce-scatter and apply to parameter shard"""
        if self.reduce_scatter_handle is not None:
            self.reduce_scatter_handle.wait()
            self.reduce_scatter_handle = None

        # Update parameter shard gradients
        if not hasattr(self, "param_shard_grad"):
            self.param_shard_grad = torch.zeros_like(self.param_shard)
        self.param_shard_grad.copy_(grad_shard)


class BucketGroup:
    """Groups multiple buckets for communication management (mimics _ParamAndGradBucketGroup)"""

    def __init__(self, buckets: List[ParameterBucket]):
        self.buckets = buckets
        self.next_bucket_group = None

    def start_param_sync(self):
        """Start async parameter all-gather for all buckets in group"""
        for bucket in self.buckets:
            bucket.start_param_allgather()

    def finish_param_sync(self):
        """Finish parameter sync and dispatch next bucket group"""
        for bucket in self.buckets:
            bucket.finish_param_allgather()

        # Dispatch next bucket group's parameter prefetch (key Megatron pattern!)
        if self.next_bucket_group is not None:
            self.next_bucket_group.start_param_sync()

    def start_grad_sync(self):
        """Start async gradient reduce-scatter for all buckets"""
        self.grad_shards = []
        for bucket in self.buckets:
            grad_shard = bucket.start_grad_reduce_scatter()
            self.grad_shards.append(grad_shard)

    def finish_grad_sync(self):
        """Finish gradient sync - enables optimizer updates for this bucket group"""
        for bucket, grad_shard in zip(self.buckets, self.grad_shards):
            bucket.finish_grad_reduce_scatter(grad_shard)


class AsyncRowParallelLinear(torch.autograd.Function):
    """Row-parallel linear with async all-reduce and overlapped weight gradient computation"""

    @staticmethod
    def forward(ctx, input, weight, tp_group):
        ctx.save_for_backward(input, weight)
        ctx.tp_group = tp_group
        # Local matrix multiplication
        output = F.linear(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        tp_group = ctx.tp_group

        # Step 1: Compute input gradient first (this needs to be all-reduced)
        grad_input = grad_output.matmul(weight)

        # Step 2: Start async all-reduce of input gradients
        all_reduce_handle = None
        if tp_group is not None:
            all_reduce_handle = dist.all_reduce(
                grad_input, group=tp_group, async_op=True
            )

        # Step 3: While all-reduce is happening, compute weight gradients
        # This computation overlaps with the communication!
        grad_weight = grad_output.transpose(-2, -1).matmul(input)

        # Step 4: Wait for all-reduce to complete
        if all_reduce_handle is not None:
            all_reduce_handle.wait()

        return grad_input, grad_weight, None


class AsyncColumnParallelLinear(torch.autograd.Function):
    """Column-parallel linear with local computation only"""

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return F.linear(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        # Column parallel: no all-reduce needed, just local gradients
        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.transpose(-2, -1).matmul(input)

        return grad_input, grad_weight


class AsyncTensorParallelLinear(nn.Module):
    """Linear layer with tensor parallelism and optimized async communication"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_mode: str,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.parallel_mode = parallel_mode  # 'column' or 'row'

        if tp_group is not None:
            self.tp_size = dist.get_world_size(tp_group)
            self.tp_rank = dist.get_rank(tp_group)
        else:
            self.tp_size = 1
            self.tp_rank = 0

        if parallel_mode == "column":
            # Column parallel: split output dimension
            self.out_features_per_rank = out_features // self.tp_size
            self.weight = nn.Parameter(
                torch.randn(
                    self.out_features_per_rank, in_features, dtype=torch.bfloat16
                )
            )
        elif parallel_mode == "row":
            # Row parallel: split input dimension
            self.in_features_per_rank = in_features // self.tp_size
            self.weight = nn.Parameter(
                torch.randn(
                    out_features, self.in_features_per_rank, dtype=torch.bfloat16
                )
            )
        else:
            raise ValueError(f"Unknown parallel_mode: {parallel_mode}")

    def forward(self, x):
        if self.parallel_mode == "column":
            # Column parallel: local computation only
            return AsyncColumnParallelLinear.apply(x, self.weight)
        else:
            # Row parallel: async all-reduce with overlapped weight gradient computation
            return AsyncRowParallelLinear.apply(x, self.weight, self.tp_group)


class AsyncAttention(nn.Module):
    """Multi-head attention with tensor parallelism"""

    def __init__(
        self, d_model: int, n_heads: int, tp_group: Optional[dist.ProcessGroup] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.tp_group = tp_group

        tp_size = dist.get_world_size(tp_group) if tp_group else 1
        self.n_heads_per_rank = n_heads // tp_size
        self.head_dim = d_model // n_heads

        # QKV projection (column parallel)
        self.qkv_proj = AsyncTensorParallelLinear(
            d_model, 3 * d_model, "column", tp_group
        )

        # Output projection (row parallel)
        self.out_proj = AsyncTensorParallelLinear(d_model, d_model, "row", tp_group)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # QKV projection (column parallel - no communication)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads_per_rank, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Attention computation
        q = q.transpose(1, 2)  # [B, n_heads_per_rank, L, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # Output projection (row parallel - async all-reduce in backward)
        return self.out_proj(attn_out)


class AsyncTransformerBlock(nn.Module):
    """Transformer block with bucket-based FSDP and async parameter management"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_id: int,
        tp_group: Optional[dist.ProcessGroup] = None,
        dp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_id = layer_id
        self.attention = AsyncAttention(d_model, n_heads, tp_group)

        # Layer norms (not parallelized)
        self.norm1 = nn.LayerNorm(d_model, dtype=torch.bfloat16)
        self.norm2 = nn.LayerNorm(d_model, dtype=torch.bfloat16)

        # FFN layers (will be managed by bucket system)
        self.ffn_up = nn.Linear(d_model, d_model * 4, dtype=torch.bfloat16)
        self.ffn_down = nn.Linear(d_model * 4, d_model, dtype=torch.bfloat16)

        # Bucket group for this layer's parameters (created later)
        self.bucket_group: Optional[BucketGroup] = None

    def set_bucket_group(self, bucket_group: BucketGroup):
        """Assign bucket group for parameter management"""
        self.bucket_group = bucket_group

    def forward(self, x):
        # Wait for parameter prefetch if needed
        if self.bucket_group is not None:
            self.bucket_group.finish_param_sync()

        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn_up(x)
        x = F.gelu(x)
        x = self.ffn_down(x)
        x = residual + x

        return x


class AsyncTransformer(nn.Module):
    """Transformer with bucket-based FSDP and async optimizations"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tp_group: Optional[dist.ProcessGroup] = None,
        dp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()

        # Embedding (not parallelized for simplicity)
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=torch.bfloat16)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                AsyncTransformerBlock(d_model, n_heads, i, tp_group, dp_group)
                for i in range(n_layers)
            ]
        )

        # Output projection (column parallel)
        self.output_proj = AsyncTensorParallelLinear(
            d_model, vocab_size, "column", tp_group
        )

        # Create bucket groups for FSDP parameter management
        self.bucket_groups = self._create_bucket_groups(dp_group)

    def _create_bucket_groups(
        self, dp_group: Optional[dist.ProcessGroup]
    ) -> List[BucketGroup]:
        """Create parameter buckets following Megatron's pattern"""
        bucket_groups = []

        # Group parameters by layer (each layer = one bucket for simplicity)
        # In reverse order to match backward pass order
        for i, layer in enumerate(reversed(self.layers)):
            layer_params = []

            # Collect all parameters from this layer
            for param in layer.parameters():
                if param.requires_grad:
                    layer_params.append(param)

            if layer_params:
                bucket = ParameterBucket(layer_params, i, dp_group)
                bucket_group = BucketGroup([bucket])
                bucket_groups.append(bucket_group)

                # Set reverse reference
                layer.set_bucket_group(bucket_group)

        # Link bucket groups for prefetching pipeline
        for i in range(len(bucket_groups) - 1):
            bucket_groups[i].next_bucket_group = bucket_groups[i + 1]

        return bucket_groups

    def start_prefetch_pipeline(self):
        """Start the parameter prefetch pipeline"""
        if self.bucket_groups:
            self.bucket_groups[0].start_param_sync()

    def forward(self, input_ids):
        # Start parameter prefetch pipeline
        self.start_prefetch_pipeline()

        x = self.embedding(input_ids)

        # Forward through layers (bucket groups handle async prefetching)
        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


class AsyncOptimizer:
    """Bucket-based optimizer with async gradient communication and layer-wise updates"""

    def __init__(self, model: AsyncTransformer, lr: float = 1e-4):
        self.model = model
        self.lr = lr

        # Create FP32 master parameters for each bucket
        self.master_params: Dict[int, torch.Tensor] = {}
        for bucket_group in model.bucket_groups:
            for bucket in bucket_group.buckets:
                self.master_params[bucket.bucket_id] = (
                    bucket.param_shard.detach().clone().float()
                )

        # Track which buckets have completed gradient reduce-scatter
        self.ready_for_update: Dict[int, bool] = {}

    def zero_grad(self):
        """Zero all gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Reset update readiness
        self.ready_for_update.clear()

    def start_gradient_communication(self):
        """Start async gradient reduce-scatter for all bucket groups"""
        # Start gradient communication for all buckets (in backward pass order)
        for bucket_group in self.model.bucket_groups:
            bucket_group.start_grad_sync()

    def step_bucket_by_bucket(self):
        """
        Perform optimizer updates bucket by bucket as gradients become ready.
        This implements the key Megatron pattern: last layer → first layer updates
        """
        # Process buckets in order (last layer → first layer in backward pass)
        for bucket_group in self.model.bucket_groups:
            # Wait for this bucket group's gradient reduce-scatter to complete
            bucket_group.finish_grad_sync()

            # Update parameters for this bucket group
            for bucket in bucket_group.buckets:
                if bucket.bucket_id not in self.ready_for_update:
                    self._update_bucket_parameters(bucket)
                    self.ready_for_update[bucket.bucket_id] = True

    def _update_bucket_parameters(self, bucket: ParameterBucket):
        """Update parameters for a single bucket using FP32 master weights"""
        with torch.no_grad():
            if hasattr(bucket, "param_shard_grad"):
                # Update FP32 master parameter
                self.master_params[bucket.bucket_id] -= (
                    self.lr * bucket.param_shard_grad.float()
                )

                # Copy back to BF16 parameter shard
                bucket.param_shard.copy_(
                    self.master_params[bucket.bucket_id].to(torch.bfloat16)
                )

    def step(self):
        """
        Complete optimizer step with proper async handling.
        This matches Megatron's step_with_ready_grads pattern.
        """
        # Start gradient communication for all buckets
        self.start_gradient_communication()

        # Perform bucket-by-bucket updates as gradients become ready
        self.step_bucket_by_bucket()

        # Update non-FSDP parameters (TP parameters, layer norms, etc.)
        self._update_non_fsdp_parameters()

        # Start parameter prefetch for next iteration (key Megatron pattern!)
        self.model.start_prefetch_pipeline()

    def _update_non_fsdp_parameters(self):
        """Update parameters not managed by FSDP buckets"""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # Check if this parameter is managed by bucket system
                    is_bucketed = False
                    for bucket_group in self.model.bucket_groups:
                        for bucket in bucket_group.buckets:
                            if any(p is param for p in bucket.params):
                                is_bucketed = True
                                break
                        if is_bucketed:
                            break

                    # Update non-bucketed parameters directly
                    if not is_bucketed:
                        param -= self.lr * param.grad


def create_async_transformer(
    vocab_size: int = 50000,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    tp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
    lr: float = 1e-4,
):
    """Create async transformer model and optimizer"""

    model = AsyncTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        tp_group=tp_group,
        dp_group=dp_group,
    )

    optimizer = AsyncOptimizer(model, lr=lr)

    return model, optimizer


def training_step(
    model: AsyncTransformer,
    optimizer: AsyncOptimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Single training step demonstrating Megatron's async patterns:
    1. Forward pass with parameter prefetching pipeline
    2. Backward pass with async gradient communication
    3. Layer-wise optimizer updates as gradients become ready
    4. Parameter prefetch for next iteration
    """

    # Forward pass (includes async parameter prefetching pipeline)
    logits = model(input_ids)

    # Compute loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Backward pass (gradients computed, but communication starts in optimizer.step())
    optimizer.zero_grad()
    loss.backward()

    # Optimizer step with async gradient communication and layer-wise updates
    optimizer.step()

    return loss

class TrainingStepModule(nn.Module):
    """Module to encapsulate the training step for async transformer"""

    def __init__(self, model: AsyncTransformer, optimizer: AsyncOptimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        return training_step(self.model, self.optimizer, input_ids, labels)


# Example usage:
if __name__ == "__main__":
    # Initialize fake distributed groups for testing
    world_size = 8
    rank = 0
    torch.distributed.init_process_group(
        backend="fake",
        world_size=world_size,
        store=FakeStore(),
        rank=rank,
    )

    parallel_dims = ParallelDims(dp_shard=4, tp=2, world_size=8)
    mesh = parallel_dims.build_mesh()
    # Create model and optimizer with fake groups for demonstration
    tp_group = mesh["tp"].get_group()
    dp_group = mesh["dp_shard"].get_group()
    with torch.device("meta"):
        model, optimizer = create_async_transformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=4,
            tp_group=tp_group,
            dp_group=dp_group,
        )
        training_step_module = TrainingStepModule(model, optimizer)
        input_ids = torch.randint(0, 1000, (32, 64), dtype=torch.int64)
        labels = torch.randint(0, 1000, (32, 64), dtype=torch.int64)

        export = torch.export.export(training_step_module, args=(input_ids, labels), strict=False)
        print(export.graph.python_code("self").src)
