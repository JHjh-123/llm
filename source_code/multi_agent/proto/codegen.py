from grpc_tools import protoc
import os
import sys

def main():
    proto_dir = os.path.dirname(os.path.abspath(__file__))
    
    args = [
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        os.path.join(proto_dir, "agent.proto"),
    ]
    
    print(f"Running protoc with args: {args}")
    if protoc.main(args) != 0:
        print("Error: protoc compilation failed.")
        sys.exit(1)
    
    # Patch generated import in agent_pb2_grpc.py to support relative package imports
    grpc_file = os.path.join(proto_dir, "agent_pb2_grpc.py")
    if os.path.exists(grpc_file):
        content = open(grpc_file, "r", encoding="utf-8").read()
        content = content.replace("import agent_pb2 as agent__pb2", "from . import agent_pb2 as agent__pb2")
        open(grpc_file, "w", encoding="utf-8").write(content)
        print("Successfully patched agent_pb2_grpc.py relative import.")
        
    print("gRPC codegen completed successfully.")

if __name__ == "__main__":
    main()
