"""A LangGraph agent with both kinds of LangGraph memory.

- Short-term (per-conversation) memory: a checkpointer (SqliteSaver) stores the
  full message history for each `thread_id`, so a conversation can be resumed.
- Long-term (cross-conversation) memory: a store (SqliteStore) holds facts about
  each `user_id`. The agent saves facts with the `remember` tool, and every turn
  starts by loading them into the system prompt.

Both live in memory.db, so they survive restarts.
"""

import uuid
from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedStore, ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

DB_PATH = "memory.db"
MODEL = "qwen3.5:4b"  # served by the local Ollama daemon


@tool
def remember(
    fact: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Save a durable fact about the user (name, preferences, projects, etc.)
    so it is available in future conversations."""
    user_id = config["configurable"]["user_id"]
    store.put(("memories", user_id), str(uuid.uuid4()), {"fact": fact})
    return f"Saved: {fact}"


@tool
def forget(
    fact_substring: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Delete saved facts about the user that contain the given text."""
    user_id = config["configurable"]["user_id"]
    ns = ("memories", user_id)
    matches = [
        item
        for item in store.search(ns, limit=1000)
        if fact_substring.lower() in item.value["fact"].lower()
    ]
    for item in matches:
        store.delete(ns, item.key)
    return f"Removed {len(matches)} fact(s)." if matches else "No matching facts."


TOOLS = [remember, forget]


def build_graph(model, checkpointer, store):
    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
        user_id = config["configurable"]["user_id"]
        facts = [m.value["fact"] for m in store.search(("memories", user_id), limit=100)]
        memory_block = "\n".join(f"- {f}" for f in facts) or "(nothing yet)"
        system = SystemMessage(
            "You are a helpful assistant with long-term memory.\n"
            "When the user tells you something worth remembering across "
            "conversations, call `remember`. If they ask you to forget "
            "something, call `forget`.\n\n"
            f"What you know about this user:\n{memory_block}"
        )
        response = model_with_tools.invoke([system, *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, store=store)


def make_model():
    return ChatOllama(
        model=MODEL,
        # Thinking makes a 4B model on CPU much slower; tool calls work without it.
        reasoning=False,
        num_ctx=8192,
    )


def main():
    import argparse

    p = argparse.ArgumentParser(description="Chat with a LangGraph agent that has memory.")
    p.add_argument("--user", default="default-user", help="whose long-term memories to use")
    p.add_argument("--thread", default=None, help="resume a conversation (default: new thread)")
    args = p.parse_args()
    thread_id = args.thread or str(uuid.uuid4())

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer, SqliteStore.from_conn_string(
        DB_PATH
    ) as store:
        checkpointer.setup()
        store.setup()
        graph = build_graph(make_model(), checkpointer, store)
        config = {"configurable": {"thread_id": thread_id, "user_id": args.user}}

        print(f"user={args.user} thread={thread_id}  (Ctrl-D to quit)")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\nResume with: uv run agent.py --user {args.user} --thread {thread_id}")
                break
            if not text:
                continue
            result = graph.invoke({"messages": [("user", text)]}, config)
            print("agent>", result["messages"][-1].text)


if __name__ == "__main__":
    main()
