use protobuf_codegen;
use protoc_bin_vendored;
fn main() {
    protobuf_codegen::Codegen::new()
        .protoc()
        .protoc_path(&protoc_bin_vendored::protoc_bin_path().unwrap())
        .includes(&["src/graph"])
        .input("src/graph/torch_titan.proto")
        .cargo_out_dir("protos")
        .run_from_script();
}
