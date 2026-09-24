# langgraph-memory-agent

A LangGraph chat agent (qwen3.5:4b via local Ollama) that uses both kinds of LangGraph memory:

| Memory | LangGraph piece | Keyed by | What it keeps |
|---|---|---|---|
| Short-term | `SqliteSaver` checkpointer | `thread_id` | Full message history of one conversation |
| Long-term | `SqliteStore` store | `user_id` | Facts saved with the `remember` tool, shared across all of that user's conversations |

Both are stored in `memory.db`, created in the directory you run from (git-ignored).

## Run

Requires [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com).

```bash
ollama pull qwen3.5:4b
cp .env.example .env                        # edit settings as needed
uv run agent.py --user alice                 # new conversation
uv run agent.py --user alice --thread <id>   # resume one (id is printed on exit)
```

Tell it something ("I prefer metric units"), quit, start a new thread, and it will still know.
