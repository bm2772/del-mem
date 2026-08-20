
from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from .ctc_graph import CueTagContentGraph
from .experience_bank import ExperienceBank
from .llm_client import LLMClient
from .nodes import answer_node, reflect_node, retrieve_node, route_after_reflect, route_passthrough_node
from .state import IterRetState


def build_graph(llm: LLMClient, graph: CueTagContentGraph, bank: ExperienceBank):
    workflow = StateGraph(IterRetState)

    workflow.add_node("retrieve", partial(retrieve_node, graph=graph, bank=bank, llm=llm))
    workflow.add_node("reflect", partial(reflect_node, graph=graph, bank=bank, llm=llm))
    workflow.add_node("route", route_passthrough_node)
    workflow.add_node("answer", partial(answer_node, llm=llm))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "reflect")
    workflow.add_edge("reflect", "route")
    workflow.add_conditional_edges("route", route_after_reflect, {"retrieve": "retrieve", "answer": "answer"})
    workflow.add_edge("answer", END)

    return workflow.compile()
