import argparse
import asyncio

from graph import CHECKPOINT_PATH, open_persistent_graph


async def main(thread_id: str, message: str):
    # The same thread ID loads the same conversation, even in a new Python process.
    config = {"configurable": {"thread_id": thread_id}}
    async with open_persistent_graph() as graph:
        before = await graph.aget_state(config)
        if before.next:
            raise RuntimeError("This thread has an unfinished run; use another test thread ID.")
        print(f"Database: {CHECKPOINT_PATH}")
        print(f"Thread: {thread_id}; messages before: {len(before.values.get('messages', []))}")

        # LangGraph restores history. Supply only this turn's new user message.
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]}, config=config
        )
        print(f"Messages after: {len(result['messages'])}")
        print("\nRESPUESTA FINAL:\n")
        print(result["messages"][-1].content)
    # Leaving the async context closes the database connection before Python exits.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HARVIS SQLite conversation test")
    parser.add_argument("thread_id", help="Reuse this ID to continue a saved conversation")
    parser.add_argument("message", help="New user message (put it in quotes)")
    args = parser.parse_args()
    asyncio.run(main(args.thread_id, args.message))
