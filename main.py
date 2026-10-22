from graph import graph


result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Tengo 23 acciones que valen 182.40 euros. "
                    "¿Cuánto vale mi posición?"
                )
            }
        ]
    }
)


print("\nRESPUESTA FINAL:\n")

print(
    result["messages"][-1].content
)