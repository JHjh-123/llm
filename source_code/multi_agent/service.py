from __future__ import annotations

import json
import os
from concurrent import futures
from dataclasses import asdict

import grpc

from multi_agent.runner import ExperimentRunner
from multi_agent.agents import AgentContext
from multi_agent.metrics import MetricsCollector
from multi_agent.protocol import Message
from multi_agent.proto import agent_pb2, agent_pb2_grpc

# Instantiate the backend runner containing all 9 agents and shared memory/state storages
runner = ExperimentRunner()


def _build_context(mode: str, task_id: str, task: str) -> AgentContext:
    metrics = MetricsCollector(mode=mode, task=task)
    return AgentContext(
        mode=mode,
        task_id=task_id,
        memory=runner.memory,
        state_store=runner.state_store,
        metrics=metrics,
        enable_memory_search=os.getenv("FEATURE_MEMORY_SEARCH", "1") == "1",
        enable_memory_write=os.getenv("FEATURE_MEMORY_WRITE", "1") == "1",
        enable_state_exchange=os.getenv("FEATURE_STATE_EXCHANGE", "1") == "1",
        security_reviewer=runner.security_reviewer,
        debugger=runner.debugger,
    )


def message_to_protobuf(msg: Message) -> agent_pb2.Message:
    return agent_pb2.Message(
        message_id=msg.message_id,
        task_id=msg.task_id,
        parent_id=msg.parent_id or "",
        sender=msg.sender,
        receiver=msg.receiver,
        mode=msg.mode,
        content=msg.content,
        payload_json=json.dumps(msg.payload, ensure_ascii=False) if msg.payload else "{}",
        created_at=msg.created_at,
        error=json.dumps(msg.error, ensure_ascii=False) if msg.error else ""
    )


def protobuf_to_message(pb_msg: agent_pb2.Message) -> Message:
    payload = {}
    if pb_msg.payload_json:
        try:
            payload = json.loads(pb_msg.payload_json)
        except Exception:
            pass
            
    error = None
    if pb_msg.error:
        try:
            error = json.loads(pb_msg.error)
        except Exception:
            error = {"message": pb_msg.error}
            
    return Message(
        message_id=pb_msg.message_id,
        task_id=pb_msg.task_id,
        parent_id=pb_msg.parent_id if pb_msg.parent_id else None,
        sender=pb_msg.sender,
        receiver=pb_msg.receiver,
        mode=pb_msg.mode,
        content=pb_msg.content,
        payload=payload,
        created_at=pb_msg.created_at,
        error=error
    )


class AgentServiceHandler(agent_pb2_grpc.AgentServiceServicer):
    def RouteTask(self, request, context):
        ctx = _build_context(request.mode, request.task_id, request.task)
        try:
            msg = runner.router.route(request.task, ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))

    def PlanTask(self, request, context):
        ctx = _build_context(request.mode, request.task_id, request.task)
        try:
            msg = runner.planner.plan(request.task, ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))

    def ResearchTask(self, request, context):
        ctx = _build_context(request.mode, request.task_id, request.task)
        try:
            plan_msg = protobuf_to_message(request.plan)
            msg = runner.researcher.research(request.task, plan_msg, ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))

    def ExecuteTask(self, request, context):
        ctx = _build_context(request.mode, request.task_id, request.task)
        try:
            findings_msg = protobuf_to_message(request.findings)
            msg = runner.executor.execute(request.task, findings_msg, ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))

    def SummarizeTask(self, request, context):
        ctx = _build_context(request.mode, request.task_id, request.task)
        try:
            exec_msg = protobuf_to_message(request.execution)
            msg = runner.summarizer.summarize(request.task, exec_msg, ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))

    def VerifyTask(self, request, context):
        ctx = _build_context(request.mode, request.task_id, request.task)
        try:
            plan_msg = protobuf_to_message(request.plan)
            sum_msg = protobuf_to_message(request.summary)
            msg = runner.verifier.verify(request.task, plan_msg, sum_msg, ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))

    def ReviewCode(self, request, context):
        ctx = _build_context(request.mode, request.task_id, request.task)
        try:
            msg = runner.security_reviewer.review(request.task, request.code, ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))

    def DebugCode(self, request, context):
        ctx = _build_context(request.mode, request.task_id, request.task)
        try:
            msg = runner.debugger.debug(request.task, request.code, request.error, request.stdout, ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))

    def ArchiveMemory(self, request, context):
        ctx = _build_context(request.mode, request.task_id, "archive")
        try:
            msg = runner.archivist.archive(ctx)
            return message_to_protobuf(msg)
        except Exception as e:
            return agent_pb2.Message(error=json.dumps({"message": str(e)}))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentServiceHandler(), server)
    server.add_insecure_port('[::]:50051')
    print("Starting gRPC Server on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
