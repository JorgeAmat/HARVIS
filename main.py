from graph import graph


if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Tengo 73 acciones a 181.4 euros cada una. Su precio sube un 15%. "
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
