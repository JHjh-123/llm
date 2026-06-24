from __future__ import annotations

from typing import Any, Protocol, TypedDict

from multi_agent.agents import AgentContext, ExecutorAgent, PlannerAgent, ResearchAgent, SummarizerAgent, VerifierAgent, RouterAgent, SecurityReviewerAgent, DebuggerAgent, MemoryArchivistAgent
from multi_agent.metrics import MetricsCollector
from multi_agent.protocol import Message


class AgentBundle(TypedDict):
    planner: PlannerAgent
    researcher: ResearchAgent
    executor: ExecutorAgent
    summarizer: SummarizerAgent
    verifier: VerifierAgent
    router: RouterAgent
    security_reviewer: SecurityReviewerAgent
    debugger: DebuggerAgent
    archivist: MemoryArchivistAgent


class Orchestrator(Protocol):
    name: str

    def run(self, task: str, ctx: AgentContext, metrics: MetricsCollector) -> Message:
        raise NotImplementedError


class SequentialOrchestrator:
    name = "sequential"

    def __init__(self, agents: AgentBundle) -> None:
        self.agents = agents

    def run(self, task: str, ctx: AgentContext, metrics: MetricsCollector) -> Message:
        route = "full_pipeline"
        router = self.agents.get("router")
        if router:
            try:
                route_msg = router.route(task, ctx)
                metrics.record_message(route_msg)
                if route_msg.payload and "state" in route_msg.payload:
                    route = route_msg.payload["state"].get("route", "full_pipeline")
            except Exception:
                pass

        if route == "direct_summarize":
            from multi_agent.protocol import make_text_message
            execution_dummy = make_text_message("router", "summarizer", f"Direct query: {task}", ctx.task_id)
            summary = self.agents["summarizer"].summarize(task, execution_dummy, ctx)
            metrics.record_message(summary)
            return summary

        current_task = task
        max_retries = 2
        for retry in range(max_retries + 1):
            plan = self.agents["planner"].plan(current_task, ctx)
            metrics.record_message(plan)

            findings = self.agents["researcher"].research(current_task, plan, ctx)
            metrics.record_message(findings)

            execution = self.agents["executor"].execute(current_task, findings, ctx)
            metrics.record_message(execution)

            summary = self.agents["summarizer"].summarize(current_task, execution, ctx)
            metrics.record_message(summary)

            verification = self.agents["verifier"].verify(task, plan, summary, ctx)
            metrics.record_message(verification)

            approved = True
            feedback = ""
            if verification.payload and "state" in verification.payload:
                approved = bool(verification.payload["state"].get("approved", True))
                feedback = str(verification.payload["state"].get("feedback", ""))

            if approved or retry == max_retries:
                return summary
            else:
                current_task = f"{task}\n\n[Verifier Feedback (Attempt {retry + 1})]: {feedback}"
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
            verifier_feedback: str
            retry_count: int
            route: str

        graph = StateGraph(GraphState)

        def route_node(state: GraphState) -> GraphState:
            router = self.agents.get("router")
            route_choice = "full_pipeline"
            if router:
                try:
                    message = router.route(state["task"], state["ctx"])
                    state["metrics"].record_message(message)
                    if message.payload and "state" in message.payload:
                        route_choice = message.payload["state"].get("route", "full_pipeline")
                except Exception:
                    pass
            return {"route": route_choice}

        def check_route(state: GraphState) -> str:
            r = state.get("route", "full_pipeline")
            if r == "direct_summarize":
                return "summarize_direct"
            return "plan"

        def plan_node(state: GraphState) -> GraphState:
            task = state["task"]
            feedback = state.get("verifier_feedback", "")
            if feedback:
                task = f"{task}\n\n[Verifier Feedback]: {feedback}"
            message = self.agents["planner"].plan(task, state["ctx"])
            state["metrics"].record_message(message)
            return {"plan": message}

        def research_node(state: GraphState) -> GraphState:
            task = state["task"]
            feedback = state.get("verifier_feedback", "")
            if feedback:
                task = f"{task}\n\n[Verifier Feedback]: {feedback}"
            message = self.agents["researcher"].research(task, state["plan"], state["ctx"])
            state["metrics"].record_message(message)
            return {"findings": message}

        def execute_node(state: GraphState) -> GraphState:
            task = state["task"]
            feedback = state.get("verifier_feedback", "")
            if feedback:
                task = f"{task}\n\n[Verifier Feedback]: {feedback}"
            message = self.agents["executor"].execute(task, state["findings"], state["ctx"])
            state["metrics"].record_message(message)
            return {"execution": message}

        def summarize_node(state: GraphState) -> GraphState:
            task = state["task"]
            feedback = state.get("verifier_feedback", "")
            if feedback:
                task = f"{task}\n\n[Verifier Feedback]: {feedback}"
            message = self.agents["summarizer"].summarize(task, state["execution"], state["ctx"])
            state["metrics"].record_message(message)
            return {"summary": message}

        def summarize_direct_node(state: GraphState) -> GraphState:
            from multi_agent.protocol import make_text_message
            dummy = make_text_message("router", "summarizer", f"Direct query: {state['task']}", state["ctx"].task_id)
            message = self.agents["summarizer"].summarize(state["task"], dummy, state["ctx"])
            state["metrics"].record_message(message)
            return {"summary": message}

        def verify_node(state: GraphState) -> GraphState:
            message = self.agents["verifier"].verify(state["task"], state["plan"], state["summary"], state["ctx"])
            state["metrics"].record_message(message)
            
            approved = True
            feedback = ""
            if message.payload and "state" in message.payload:
                approved = bool(message.payload["state"].get("approved", True))
                feedback = str(message.payload["state"].get("feedback", ""))

            retry_count = state.get("retry_count", 0)
            return {
                "verifier_feedback": feedback if not approved else "",
                "retry_count": retry_count + 1
            }

        def check_verification(state: GraphState) -> str:
            feedback = state.get("verifier_feedback", "")
            retry_count = state.get("retry_count", 0)
            if not feedback or retry_count >= 3:
                return "end"
            return "replan"

        graph.add_node("route", route_node)
        graph.add_node("plan", plan_node)
        graph.add_node("research", research_node)
        graph.add_node("execute", execute_node)
        graph.add_node("summarize", summarize_node)
        graph.add_node("summarize_direct", summarize_direct_node)
        graph.add_node("verify", verify_node)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            check_route,
            {
                "summarize_direct": "summarize_direct",
                "plan": "plan"
            }
        )
        graph.add_edge("plan", "research")
        graph.add_edge("research", "execute")
        graph.add_edge("execute", "summarize")
        graph.add_edge("summarize", "verify")
        graph.add_edge("summarize_direct", END)

        graph.add_conditional_edges(
            "verify",
            check_verification,
            {
                "end": END,
                "replan": "plan"
            }
        )

        return graph.compile()


def build_orchestrator(kind: str, agents: AgentBundle) -> Orchestrator:
    if kind == "langgraph":
        try:
            return LangGraphOrchestrator(agents)
        except ModuleNotFoundError:
            return SequentialOrchestrator(agents)
    return SequentialOrchestrator(agents)
