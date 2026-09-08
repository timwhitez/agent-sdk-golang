# Agent SDK development guide

Go SDK for model invocation, tool execution, events, and compaction.
This is the shared entry point for coding agents. Read code and nearby tests
first; detailed references are optional and task-specific.

## Start and validate

Toolchain and dependencies are defined in [go.mod](go.mod).

```sh
go test ./...
go vet ./...
go build ./...
```

Use focused package tests while iterating, `gofmt` for changed Go files, and
`go test -race` for affected concurrent/stateful code. Provider changes normally
use local HTTP/SSE fixtures; add a live probe only when it provides missing evidence.

Keep changes within the requested scope and preserve unrelated edits. Reuse
existing lifecycle, serialization, artifact and cancellation mechanisms.

## Important boundaries

- The Query Driver owns request/tool-block execution; hosts own sessions,
  confirmation and durable application state. SDK code must not depend on Goode.
- Each accepted Tool Call needs one terminal history result. Publication and
  delivery are different; unknown side effects remain indeterminate.
- Preserve history/checkpoint authority, cancellation bounds and artifact ownership.
  Integrity hashes protect data; a Git revision alone does not prove compatibility.
- Event correlation is explicit metadata, not a second event sequence or proof
  of complete provenance. Avoid secrets/raw content in new diagnostics.
- When changing a contract, update the relevant tests and documentation rather
  than adding permanent workflow restrictions to this file.

## Code and references

| Area | Code | Reference |
|---|---|---|
| Driver, Frame, events, history | `sdk/agent` | [Architecture](agent_docs/architecture.md) |
| Compaction and checkpoints | `sdk/agent/compaction` | [Architecture](agent_docs/architecture.md) |
| Provider requests and streaming | `sdk/llm` | [Providers](agent_docs/providers.md) |
| Tools and sandbox | `sdk/tools` | [Tools/sandbox](agent_docs/tools-and-sandbox.md) |
| Artifact/accounting contracts and validation | `sdk/artifact`, `sdk/accounting` | [Tests](agent_docs/testing.md), [building](agent_docs/building.md) |

Consult public API comments for precise behavior and current defaults. Plans,
historical benchmark values and old release receipts are not current guarantees.
