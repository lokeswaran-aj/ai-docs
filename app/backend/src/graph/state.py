from langgraph.graph import MessagesState


class State(MessagesState):
    documents: list[str]
