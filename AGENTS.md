# HARVIS — Project Instructions

## Project goal

HARVIS is Jorge's long-term personal AI assistant.

The goal is not just to build something that works.

The project must remain:

- understandable
- inspectable
- modular
- maintainable
- portable
- secure
- usable for many years

Jorge is learning agentic AI while building HARVIS.

Therefore, understanding the architecture is as important as shipping features.


## Core principle

Do not hide important behavior behind unnecessary abstractions.

Prefer explicit, readable Python over clever or highly abstract code.

Every important agent behavior should be inspectable.


## Teaching requirement

Jorge wants to understand every important change.

Before making a meaningful architectural change:

1. Explain briefly what problem is being solved.
2. Explain what files will change.
3. Explain the new concept being introduced.
4. Keep the implementation as small as reasonably possible.

After making changes:

1. Summarize what changed.
2. Explain the execution flow.
3. Point out the most important lines of code.
4. Explain how to test it.

Do not add large amounts of code without explanation.


## Development style

Prefer incremental development.

Good:

1. Add one capability.
2. Run it.
3. Inspect behavior.
4. Test it.
5. Continue.

Avoid implementing many major features simultaneously.

Do not over-engineer early versions.


## Current architecture

The project currently separates:

### graph.py

The HARVIS brain.

Contains:

- LangGraph state
- agent nodes
- tools
- routing
- model interaction

### app.py

The user interface.

Currently intended to use Chainlit.

It should communicate with graph.py rather than contain agent logic.

### main.py

Local testing / debugging entrypoint.

Use this to test the graph independently from the UI.

This separation should normally be preserved.


## Planned architecture

HARVIS will gradually gain:

1. Chainlit conversational UI
2. persistent conversation state
3. Obsidian-based human-readable long-term memory
4. semantic memory / retrieval
5. Gmail integration
6. Google Calendar integration
7. voice input and output
8. MCP integrations
9. human approval for consequential actions
10. potentially multi-agent architecture only when justified


## Memory philosophy

Long-term personal memory must remain portable and human-readable.

Obsidian Markdown files will be the main human-readable source of truth.

Vector databases may be used as indexes, but must not become the only copy of important memories.

Architecture principle:

Obsidian / raw files
        ↓
semantic index
        ↓
retrieval

Deleting or rebuilding the semantic index must not destroy the original memories.


## Personal data

HARVIS may eventually contain highly personal information.

Treat personal information conservatively.

Never:

- commit secrets
- commit API keys
- expose tokens
- print full credentials
- put secrets directly in source code

Secrets belong in `.env`.

`.env` must remain excluded from Git.


## External actions

Reading and analysing information may be automatic.

Consequential actions require explicit user approval unless Jorge later changes this policy.

Examples:

- sending an email
- deleting an email
- changing a calendar event
- deleting files
- sending messages
- making purchases
- submitting forms

Preferred flow:

HARVIS proposes action
        ↓
Jorge reviews
        ↓
Jorge approves
        ↓
action executes


## Email design

Initial Gmail capabilities should be:

- search email
- read email
- summarize email
- prepare draft

Sending should require explicit human approval.


## Agent design

Start with one capable agent plus tools.

Do not introduce multi-agent systems simply because they are fashionable.

Use multiple agents only when there is a concrete architectural reason.


## LangGraph

LangGraph should remain the primary orchestration layer.

Important concepts should remain visible:

- state
- nodes
- edges
- conditional routing
- tool calls
- observations
- loops
- checkpoints

Prefer architectures that can be visualized as a graph.


## Tools

Tools should:

- have clear names
- have clear docstrings
- do one main thing
- return predictable outputs
- expose errors clearly
- be testable independently

Do not allow an LLM to execute arbitrary destructive operations.


## Code quality

Use:

- descriptive names
- type hints where useful
- small functions
- comments for non-obvious logic
- explicit error handling

Avoid:

- unnecessary design patterns
- premature abstractions
- giant functions
- duplicated logic
- hidden global state


## Dependencies

Before adding a new dependency:

1. Explain why it is needed.
2. Prefer mature and maintained libraries.
3. Avoid dependencies when standard Python is sufficient.


## Testing

As HARVIS grows, add tests for:

- tool behavior
- memory retrieval
- routing
- persistence
- permissions / approval flows

Agent behavior should eventually have evaluation datasets as well.


## Do not

Do not:

- rewrite the whole project without being asked
- replace LangGraph without discussing it
- introduce five frameworks for the same problem
- silently change architecture
- modify the Obsidian memory vault destructively
- send external messages without approval
- remove working code solely to make it more abstract


## Current priority

The immediate goal is:

HARVIS v0.1

Chainlit UI
    ↓
LangGraph agent
    ↓
Groq model
    ↓
tools

Then:

HARVIS v0.2

persistent conversational state

Then:

HARVIS v0.3

Obsidian long-term memory

Proceed incrementally.