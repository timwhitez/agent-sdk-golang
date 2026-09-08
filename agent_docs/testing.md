# Testing

## Local loop

```sh
go test ./sdk/agent                 # choose the affected package
go test ./...
go vet ./...
go build ./...
```

Format changed Go files with `gofmt`. For concurrent/stateful changes, run
`go test -race` on the affected packages; widen the scope when behavior crosses
package boundaries. Documentation-only changes need link/fact checks, not a
fabricated runtime or performance claim.

Use the existing fixture closest to the behavior. Tests and public API comments
are the source of current contract details; this page is not a copy of every test.

## Find relevant coverage

| Change | Useful starting points |
|---|---|
| Frame/request ownership and events | [execution_frame_test.go](../sdk/agent/execution_frame_test.go), [frame_correlation_test.go](../sdk/agent/frame_correlation_test.go), [frame_event_boundary_test.go](../sdk/agent/frame_event_boundary_test.go), `event_envelope*_test.go` |
| Tool outcomes, topology and publication | [tool_terminal_authority_test.go](../sdk/agent/tool_terminal_authority_test.go), [tool_outcome_projection_test.go](../sdk/agent/tool_outcome_projection_test.go), [history_mutation_test.go](../sdk/agent/history_mutation_test.go) |
| Compaction/ledger/history | [compaction_publication_test.go](../sdk/agent/compaction_publication_test.go), [manual_publication_test.go](../sdk/agent/manual_publication_test.go), `agent_compaction*_test.go`, [compaction package](../sdk/agent/compaction) |
| Provider wire/cache behavior | [llm tests](../sdk/llm), [cache_plan_wire_test.go](../sdk/llm/cache_plan_wire_test.go), [cache_request_test.go](../sdk/llm/cache_request_test.go), provider HTTP/SSE fixtures |
| Artifact, accounting and sandbox safety | [artifact](../sdk/artifact), [accounting](../sdk/accounting), [tools](../sdk/tools), [agent_artifact_boundary_test.go](../sdk/agent/agent_artifact_boundary_test.go) |

For a specific contract, search the implementation and tests:

```sh
rg -n '^func (Test|Benchmark)' sdk/agent
rg -n 'CommitCompactionHistory' sdk
```

## Evidence that matters

- Assert the observable contract, including failure/cancellation and forbidden
  effects where relevant. Green counters alone do not prove publication or delivery.
- Keep golden/compatibility fixtures for serialization changes. Use failure
  injection for state/artifact writes and source/ownership conflicts.
- Check SDK versus host responsibility before duplicating an adapter or test.
  Cross-repo changes need the actual selected SDK module, not an assumed checkout.
- Benchmarks must name the measured operation and comparable setup. SDK invocation
  counts are not HTTP attempts; harness overhead is not model-quality evidence.

Local fixtures should establish deterministic behavior without credentials.
Use a targeted live-provider probe only for behavior local fixtures cannot
establish, and report unavailable/not-run coverage honestly.
