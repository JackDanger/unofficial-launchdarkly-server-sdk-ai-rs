# Test coverage against the JavaScript SDK

Every Jest test in `@launchdarkly/server-sdk-ai` 2.0.19 (`__tests__/`) was ported to a
Rust integration test of the same name (in `snake_case`), grouped in `mod`s mirroring
the `describe` blocks. The JavaScript suites mock internals with Jest (`jest.mock`,
`jest.spyOn`); the Rust ports use real objects over test doubles in
`tests/common/mod.rs` (a fake `LdClientMin` that records every `track` call and log
line, scripted runners and providers) and assert the observable result: the events
sent, their data and metric values, and the log text.

| JavaScript test file | JS tests | Ported | Rust file |
|---|---:|---:|---|
| `LDAIConfigTrackerImpl.test.ts` | 50 | 50 | `tests/config_tracker.rs` |
| `LDGraphTrackerImpl.test.ts` | 22 | 22 | `tests/graph_tracker.rs` |
| `LDAIClientImpl.test.ts` | 48 | 48 | `tests/client.rs` |
| `agentGraph.test.ts` | 11 | 11 | `tests/agent_graph.rs` |
| `AgentGraphDefinition.test.ts` (44 `it` + one `it.each` over 7 graphs) | 51 | 51 | `tests/agent_graph_definition.rs` |
| `ManagedAgentGraph.test.ts` | 8 | 8 | `tests/managed_agent_graph.rs` |
| `Judge.test.ts` | 39 | 38 | `tests/judge.rs` |
| `Evaluator.test.ts` | 4 | 4 | `tests/evaluator.rs` |
| `ManagedModel.test.ts` + `ManagedModelRun.test.ts` | 10 | 10 | `tests/managed_model.rs` |
| `ManagedAgent.test.ts` | 7 | 7 | `tests/managed_agent.rs` |
| `RunnerFactory.test.ts` + `RunnerProtocol.test.ts` | 19 | 19 | `tests/runner_factory.rs` |
| **Total** | **269** | **268** | |

Additional Rust-only tests: `tests/mustache_conformance.rs` (60 templates and 6
malformed templates rendered by mustache.js 4.2.0, in `tests/fixtures/`),
`tests/openai_compatible.rs` (the optional provider against a local mock server), and
unit tests in `src/mustache.rs`.

## Not ported

- `Judge.test.ts` › *handles non-Error exceptions* — JavaScript can throw a non-`Error`
  value, which yields `errorMessage: 'Unknown error'`. Rust runner errors are
  `Box<dyn Error>` and always have a message, so the case cannot occur.

## Adapted (ported, asserting the equivalent behaviour)

- **Spies on private methods** (`_evaluate`, `_getProviderFactory`) became assertions on
  their effects: the default flag value passed to `variation`, the returned config, the
  usage event, the `Attempting to create runner with provider:` debug order.
- **Mocked `Judge`, `RunnerFactory`, tracker and evaluator** became real objects over a
  registered fake provider and scripted runners. "`trackJudgeResult` was called" is
  asserted as the judge-metric event on the model's tracker, with `judgeConfigKey`.
- **`Date.now` / `Math.random` mocks** became `LdAiConfigTracker::with_clock` and
  `Judge::with_random_source`.
- **Provider packages** (`openai`, `langchain`, `vercel`): "not installed" is an empty
  registry; the multi-provider fallbacks are registered with `with_multi_provider`.
- **Default parameters** (`sampleRate = 1.0`, `multiTurn = true`) are passed explicitly,
  since Rust has no default arguments.
- **Object identity (`toBe`) and `toBeInstanceOf`** are field comparisons or are
  guaranteed by the type system.
- **Logger calls with extra arguments** (`warn(msg, trackData)`,
  `error('Judge evaluation failed:', err)`) are single strings in Rust; the tests assert
  the exact joined text.
- **`createJudge` › handles errors gracefully** mocks `_judgeConfig` to throw, which
  cannot happen in Rust; the port feeds a malformed flag value and asserts `None`.
- **Background promises** ("returns before evaluations resolve", stream metrics) gate a
  runner on a `tokio::sync::Notify` or poll with a deadline.
