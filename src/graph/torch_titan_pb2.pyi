from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NodeValue(_message.Message):
    __slots__ = ("null_value", "bool_value", "int_value", "float_value", "string_value", "device_value", "dtype_value", "layout_value", "memory_format_value", "shape_value", "sequence_value", "node_ref_value", "repr_value")
    NULL_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_VALUE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_VALUE_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_VALUE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FORMAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_VALUE_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_VALUE_FIELD_NUMBER: _ClassVar[int]
    NODE_REF_VALUE_FIELD_NUMBER: _ClassVar[int]
    REPR_VALUE_FIELD_NUMBER: _ClassVar[int]
    null_value: _struct_pb2.NullValue
    bool_value: bool
    int_value: int
    float_value: float
    string_value: str
    device_value: str
    dtype_value: str
    layout_value: str
    memory_format_value: str
    shape_value: IntList
    sequence_value: SequenceValue
    node_ref_value: str
    repr_value: str
    def __init__(self, null_value: _Optional[_Union[_struct_pb2.NullValue, str]] = ..., bool_value: bool = ..., int_value: _Optional[int] = ..., float_value: _Optional[float] = ..., string_value: _Optional[str] = ..., device_value: _Optional[str] = ..., dtype_value: _Optional[str] = ..., layout_value: _Optional[str] = ..., memory_format_value: _Optional[str] = ..., shape_value: _Optional[_Union[IntList, _Mapping]] = ..., sequence_value: _Optional[_Union[SequenceValue, _Mapping]] = ..., node_ref_value: _Optional[str] = ..., repr_value: _Optional[str] = ...) -> None: ...

class IntList(_message.Message):
    __slots__ = ("dims",)
    DIMS_FIELD_NUMBER: _ClassVar[int]
    dims: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dims: _Optional[_Iterable[int]] = ...) -> None: ...

class SequenceValue(_message.Message):
    __slots__ = ("elements",)
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedCompositeFieldContainer[NodeValue]
    def __init__(self, elements: _Optional[_Iterable[_Union[NodeValue, _Mapping]]] = ...) -> None: ...

class NamedNodeValue(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: NodeValue
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[NodeValue, _Mapping]] = ...) -> None: ...

class CollectiveMeta(_message.Message):
    __slots__ = ("group_ranks", "comm_tensor_size", "group_desc", "group_name")
    GROUP_RANKS_FIELD_NUMBER: _ClassVar[int]
    COMM_TENSOR_SIZE_FIELD_NUMBER: _ClassVar[int]
    GROUP_DESC_FIELD_NUMBER: _ClassVar[int]
    GROUP_NAME_FIELD_NUMBER: _ClassVar[int]
    group_ranks: _containers.RepeatedScalarFieldContainer[int]
    comm_tensor_size: int
    group_desc: str
    group_name: str
    def __init__(self, group_ranks: _Optional[_Iterable[int]] = ..., comm_tensor_size: _Optional[int] = ..., group_desc: _Optional[str] = ..., group_name: _Optional[str] = ...) -> None: ...

class TensorInfo(_message.Message):
    __slots__ = ("shape", "stride", "dtype", "device", "layout")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    shape: IntList
    stride: IntList
    dtype: str
    device: str
    layout: str
    def __init__(self, shape: _Optional[_Union[IntList, _Mapping]] = ..., stride: _Optional[_Union[IntList, _Mapping]] = ..., dtype: _Optional[str] = ..., device: _Optional[str] = ..., layout: _Optional[str] = ...) -> None: ...

class NodeData(_message.Message):
    __slots__ = ("name", "op", "target", "args", "kwargs", "tensor_info", "collective_meta")
    NAME_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    TENSOR_INFO_FIELD_NUMBER: _ClassVar[int]
    COLLECTIVE_META_FIELD_NUMBER: _ClassVar[int]
    name: str
    op: str
    target: str
    args: _containers.RepeatedCompositeFieldContainer[NodeValue]
    kwargs: _containers.RepeatedCompositeFieldContainer[NamedNodeValue]
    tensor_info: TensorInfo
    collective_meta: CollectiveMeta
    def __init__(self, name: _Optional[str] = ..., op: _Optional[str] = ..., target: _Optional[str] = ..., args: _Optional[_Iterable[_Union[NodeValue, _Mapping]]] = ..., kwargs: _Optional[_Iterable[_Union[NamedNodeValue, _Mapping]]] = ..., tensor_info: _Optional[_Union[TensorInfo, _Mapping]] = ..., collective_meta: _Optional[_Union[CollectiveMeta, _Mapping]] = ...) -> None: ...

class GraphData(_message.Message):
    __slots__ = ("nodes", "output_node_index")
    NODES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_NODE_INDEX_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeData]
    output_node_index: int
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeData, _Mapping]]] = ..., output_node_index: _Optional[int] = ...) -> None: ...

class GraphModuleData(_message.Message):
    __slots__ = ("graph", "user_preserved_attributes", "buffers")
    class BuffersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TensorInfo
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[TensorInfo, _Mapping]] = ...) -> None: ...
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    USER_PRESERVED_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    BUFFERS_FIELD_NUMBER: _ClassVar[int]
    graph: GraphData
    user_preserved_attributes: _containers.RepeatedCompositeFieldContainer[NamedNodeValue]
    buffers: _containers.MessageMap[str, TensorInfo]
    def __init__(self, graph: _Optional[_Union[GraphData, _Mapping]] = ..., user_preserved_attributes: _Optional[_Iterable[_Union[NamedNodeValue, _Mapping]]] = ..., buffers: _Optional[_Mapping[str, TensorInfo]] = ...) -> None: ...

class ParallelConfig(_message.Message):
    __slots__ = ("dp_replicate", "dp_shard", "cp", "tp", "pp", "world_size", "enable_loss_parallel")
    DP_REPLICATE_FIELD_NUMBER: _ClassVar[int]
    DP_SHARD_FIELD_NUMBER: _ClassVar[int]
    CP_FIELD_NUMBER: _ClassVar[int]
    TP_FIELD_NUMBER: _ClassVar[int]
    PP_FIELD_NUMBER: _ClassVar[int]
    WORLD_SIZE_FIELD_NUMBER: _ClassVar[int]
    ENABLE_LOSS_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool
    def __init__(self, dp_replicate: _Optional[int] = ..., dp_shard: _Optional[int] = ..., cp: _Optional[int] = ..., tp: _Optional[int] = ..., pp: _Optional[int] = ..., world_size: _Optional[int] = ..., enable_loss_parallel: bool = ...) -> None: ...

class TraceResult(_message.Message):
    __slots__ = ("parallel_dims", "graph_module")
    PARALLEL_DIMS_FIELD_NUMBER: _ClassVar[int]
    GRAPH_MODULE_FIELD_NUMBER: _ClassVar[int]
    parallel_dims: ParallelConfig
    graph_module: GraphModuleData
    def __init__(self, parallel_dims: _Optional[_Union[ParallelConfig, _Mapping]] = ..., graph_module: _Optional[_Union[GraphModuleData, _Mapping]] = ...) -> None: ...
