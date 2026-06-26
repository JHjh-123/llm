import sys
import grpc
import json
from multi_agent.proto import agent_pb2, agent_pb2_grpc

SERVER_ADDRESS = "localhost:50051"

mock_pb_message = agent_pb2.Message(
    message_id="msg_test_001",
    task_id="task_test_001",
    parent_id="",
    sender="test_runner",
    receiver="agent",
    mode="structured",
    content="This is a test content representing intermediate agent outcomes.",
    payload_json="{}",
    created_at=1234567.8,
    error=""
)


def test_grpc_endpoints():
    print("====================================================")
    print("Verifying gRPC Microservice Agent Endpoints Availability")
    print("====================================================")

    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = agent_pb2_grpc.AgentServiceStub(channel)

    endpoints = [
        {
            "name": "RouteTask",
            "call": lambda: stub.RouteTask(agent_pb2.RouteRequest(
                task="Design a memory system schema.",
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        },
        {
            "name": "PlanTask",
            "call": lambda: stub.PlanTask(agent_pb2.PlanRequest(
                task="Design a memory system schema.",
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        },
        {
            "name": "ResearchTask",
            "call": lambda: stub.ResearchTask(agent_pb2.ResearchRequest(
                task="Design a memory system schema.",
                plan=mock_pb_message,
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        },
        {
            "name": "ExecuteTask",
            "call": lambda: stub.ExecuteTask(agent_pb2.ExecuteRequest(
                task="Design a memory system schema.",
                findings=mock_pb_message,
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        },
        {
            "name": "SummarizeTask",
            "call": lambda: stub.SummarizeTask(agent_pb2.SummarizeRequest(
                task="Design a memory system schema.",
                execution=mock_pb_message,
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        },
        {
            "name": "VerifyTask",
            "call": lambda: stub.VerifyTask(agent_pb2.VerifyRequest(
                task="Design a memory system schema.",
                plan=mock_pb_message,
                summary=mock_pb_message,
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        },
        {
            "name": "ReviewCode",
            "call": lambda: stub.ReviewCode(agent_pb2.ReviewRequest(
                task="Design a memory system schema.",
                code="print('hello')",
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        },
        {
            "name": "DebugCode",
            "call": lambda: stub.DebugCode(agent_pb2.DebugRequest(
                task="Design a memory system schema.",
                code="print('hello')",
                error="SyntaxError",
                stdout="",
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        },
        {
            "name": "ArchiveMemory",
            "call": lambda: stub.ArchiveMemory(agent_pb2.ArchiveRequest(
                mode="structured",
                task_id="task_test_001"
            ), timeout=600)
        }
    ]

    success_count = 0
    for ep in endpoints:
        print(f"Testing [/{ep['name']}] gRPC endpoint...")
        try:
            resp = ep["call"]()
            print(f"  -> SUCCESS!")
            print(f"  -> Returned Message ID: {resp.message_id}")
            print(f"  -> Content Preview: '{resp.content[:60]}...'")
            success_count += 1
        except Exception as e:
            print(f"  -> FAILED! Error: {e}")
        print("-" * 50)

    print(f"\nVerification finished. {success_count}/{len(endpoints)} endpoints are functional.")
    print("====================================================")

    if success_count == len(endpoints):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    test_grpc_endpoints()
