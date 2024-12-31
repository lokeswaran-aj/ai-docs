from datetime import datetime, timezone

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph.configuration import Settings
from graph.prompt import SYSTEM_PROMPT
from graph.state import State
from graph.tool import TOOLS
from graph.utils import load_chat_model


async def call_agent(state: State, message: str) -> State:
    chat_model = load_chat_model().bind_tools(tools=TOOLS)
    system_message = SYSTEM_PROMPT.format(
        system_message=datetime.now(tz=timezone.utc).isoformat()
    )
    response = await chat_model.ainvoke(
        [SystemMessage(content=system_message), *state.messages]
    )

    return {
        "messages": [AIMessage(id=response.id, content=response.content)],
    }


graph_builder = StateGraph(State, config_schema=Settings)

graph_builder.add_node("agent", call_agent)
graph_builder.add_node("tools", ToolNode(TOOLS))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent", tools_condition, {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", END)

graph = graph_builder.compile()

graph.name = "AI Docs"
