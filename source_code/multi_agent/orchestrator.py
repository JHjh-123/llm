from __future__ import annotations

from typing import Any, Protocol, TypedDict

from multi_agent.agents import AgentContext, ExecutorAgent, PlannerAgent, ResearchAgent, SummarizerAgent
from multi_agent.metrics import MetricsCollector
from multi_agent.protocol import Message


class AgentBundle(TypedDict):
    planner: PlannerAgent
    researcher: ResearchAgent
    executor: ExecutorAgent
    summarizer: SummarizerAgent


class Orchestrator(Protocol):
    name: str

    def run(self, task: str, ctx: AgentContext, metrics: MetricsCollector) -> Message:
        raise NotImplementedError


class SequentialOrchestrator:
    name = "sequential"

    def __init__(self, agents: AgentBundle) -> None:
        self.agents = agents

    def run(self, task: str, ctx: AgentContext, metrics: MetricsCollector) -> Message:
        plan = self.agents["planner"].plan(task, ctx)
        metrics.record_message(plan)

        findings = self.agents["researcher"].research(task, plan, ctx)
        metrics.record_message(findings)

        execution = self.agents["executor"].execute(task, findings, ctx)
        metrics.record_message(execution)

        summary = self.agents["summarizer"].summarize(task, execution, ctx)
        metrics.record_message(summary)
        return summary


class LangGraphOrchestrator:
    name = "langgraph"

    def __init__(self, agents: AgentBundle) -> None:
        self.agents = agents
        self.graph = self._build_graph()

    def run(self, task: str, ctx: AgentContext, metrics: MetricsCollector) -> Message:
        state = self.graph.invoke({"task": task, "ctx": ctx, "metrics": metrics})
        return state["summary"]

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        class GraphState(TypedDict, total=False):
            task: str
            ctx: AgentContext
            metrics: MetricsCollector
            plan: Message
            findings: Message
            execution: Message
            summary: Message

        graph = StateGraph(GraphState)

        def plan_node(state: GraphState) -> GraphState:
            message = self.agents["planner"].plan(state["task"], state["ctx"])
            state["metrics"].record_message(message)
            return {"plan": message}

        def research_node(state: GraphState) -> GraphState:
            message = self.agents["researcher"].research(state["task"], state["plan"], state["ctx"])
            state["metrics"].record_message(message)
            return {"findings": message}

        def execute_node(state: GraphState) -> GraphState:
            message = self.agents["executor"].execute(state["task"], state["findings"], state["ctx"])
            state["metrics"].record_message(message)
            return {"execution": message}

        def summarize_node(state: GraphState) -> GraphState:
            message = self.agents["summarizer"].summarize(state["task"], state["execution"], state["ctx"])
            state["metrics"].record_message(message)
            return {"summary": message}

        graph.add_node("plan", plan_node)
        graph.add_node("research", research_node)
        graph.add_node("execute", execute_node)
        graph.add_node("summarize", summarize_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "research")
        graph.add_edge("research", "execute")
        graph.add_edge("execute", "summarize")
        graph.add_edge("summarize", END)
        return graph.compile()


def build_orchestrator(kind: str, agents: AgentBundle) -> Orchestrator:
    if kind == "langgraph":
        try:
            return LangGraphOrchestrator(agents)
        except ModuleNotFoundError:
            return SequentialOrchestrator(agents)
    return SequentialOrchestrator(agents)
