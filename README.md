# unofficial-launchdarkly-server-sdk-ai

> [!WARNING]
> **This is an UNOFFICIAL, community port.** It is not affiliated with, endorsed by,
> or supported by LaunchDarkly. Use it at your own risk; LaunchDarkly support cannot
> help you with it. For LaunchDarkly's official SDKs, see
> [docs.launchdarkly.com/sdk](https://docs.launchdarkly.com/sdk).

A Rust port of LaunchDarkly's AI SDK for server-side JavaScript,
[`@launchdarkly/server-sdk-ai`](https://github.com/launchdarkly/js-core/tree/main/packages/sdk/server-ai),
built on the official Rust core SDK,
[`launchdarkly-server-sdk`](https://crates.io/crates/launchdarkly-server-sdk).

It reads AI Configs (completion, agent and judge modes, and agent graphs) from
LaunchDarkly flags, renders their prompts with Mustache, runs judges, and reports the
same `$ld:ai:*` events with the same data payloads as the JavaScript SDK, so
LaunchDarkly's AI Config monitoring, judges and graph views work as they do for
JavaScript applications.

**Tracks:** `@launchdarkly/server-sdk-ai` **2.0.19**. Every JavaScript test in that
release was ported or is accounted for in [TESTS.md](TESTS.md).

## Quick start

```toml
[dependencies]
launchdarkly-server-sdk = "3.3"
unofficial-launchdarkly-server-sdk-ai = { git = "https://github.com/JackDanger/unofficial-launchdarkly-server-sdk-ai-rs" }
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use std::{sync::Arc, time::Duration};

use launchdarkly_server_sdk::{Client, ConfigBuilder};
use serde_json::json;
use unofficial_launchdarkly_server_sdk_ai::{
    init_ai, ContextBuilder, LdAiCompletionConfigDefault, LdAiMetrics, LdMessage,
    LdModelConfig, LdTokenUsage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::new(&std::env::var("LAUNCHDARKLY_SDK_KEY")?).build()?;
    let ld_client = Arc::new(Client::build(config)?);
    ld_client.start_with_default_executor();
    ld_client.wait_for_initialization(Duration::from_secs(10)).await;

    let ai_client = init_ai(Arc::clone(&ld_client));
    let context = ContextBuilder::new("example-user-key").name("Sandy").build()?;

    let default_value = LdAiCompletionConfigDefault {
        enabled: Some(true),
        model: Some(LdModelConfig::named("gpt-4o")),
        messages: Some(vec![LdMessage::system("You are a helpful assistant.")]),
        ..Default::default()
    };
    let ai_config = ai_client
        .completion_config(
            "my-ai-config",
            &context,
            Some(default_value),
            json!({ "userName": "Sandy" }).as_object(),
            None,
        )
        .await;

    if ai_config.enabled {
        let tracker = ai_config.create_tracker();
        let reply: Result<String, Box<dyn std::error::Error>> = tracker
            .track_metrics_of(
                |_reply| LdAiMetrics {
                    success: true,
                    tokens: Some(LdTokenUsage { total: 30, input: 20, output: 10 }),
                    ..Default::default()
                },
                || async { Ok("send ai_config.messages to your model here".to_string()) },
            )
            .await;
        println!("{reply:?}");
    }

    ld_client.flush();
    ld_client.close();
    Ok(())
}
```

### Managed models and judges against a self-hosted model

With the `openai-compatible` feature, register a provider for any OpenAI-compatible
chat-completions endpoint (vLLM, llama.cpp, Ollama, LiteLLM, OpenAI). Managed models
then track every run, and the judges attached to the AI Config run in the background
using structured output (`response_format: {"type": "json_schema", ...}`):

```rust,ignore
use unofficial_launchdarkly_server_sdk_ai::{openai_compatible::OpenAiCompatibleProvider, LdAiClient, RunnerFactory};

let factory = RunnerFactory::new().with_provider(
    "openai", // matched against the AI Config's provider.name
    Arc::new(OpenAiCompatibleProvider::new("http://vllm:8000/v1").with_api_key(None)),
);
let ai_client = LdAiClient::with_runner_factory(ld_client, factory);

if let Some(model) = ai_client.create_model("chat", &context, None, None, None).await {
    let result = model.run("What can you help me with?").await?;
    println!("{}", result.content);
    for judgement in result.evaluations.await {
        println!("{:?} {:?}", judgement.metric_key, judgement.score);
    }
}

if let Some(judge) = ai_client.create_judge("relevance-judge", &context, None, None, None, 1.0).await {
    let result = judge.evaluate("user question", "model answer", None).await;
}
```

## Cargo features

| Feature | Default | What it enables |
|---|---|---|
| `ld-client` | yes | `LdClientMin` for `launchdarkly_server_sdk::Client` (and a `log`-crate logger). Turn it off to supply your own client. |
| `openai-compatible` | no | `openai_compatible::OpenAiCompatibleProvider`, a small `reqwest` provider for OpenAI-compatible chat completions, with JSON-schema structured output for judges. It does not execute tools. |
| `preserve-order` | no | Enables `serde_json/preserve_order`, so the order of graph edge maps from flag JSON is kept as declared (see Differences). |

Requires Rust 1.97 or later (the MSRV of `launchdarkly-server-sdk` 3.3).

## API map

Method names are the JavaScript names in `snake_case`; optional JavaScript parameters
are `Option`s in the same positions.

| JavaScript | Rust |
|---|---|
| `initAi(ldClient)` | `init_ai(Arc<Client>)` / `LdAiClient::new` / `LdAiClient::with_runner_factory` |
| `LDClientMin` | trait `LdClientMin` (`variation`, `track`, `logger`) |
| `LDAIClient.completionConfig`, `agentConfig`, `judgeConfig`, `agentConfigs` | `completion_config`, `agent_config`, `judge_config`, `agent_configs` |
| `completionConfigTemplate`, `agentConfigTemplate`, `judgeConfigTemplate` | `completion_config_template`, `agent_config_template`, `judge_config_template` |
| `createModel`, `createAgent`, `createJudge` | `create_model`, `create_agent`, `create_judge` |
| `createTracker(token, ctx)`, `agentGraph`, `createGraphTracker` | `create_tracker`, `agent_graph`, `create_graph_tracker` |
| `LDAICompletionConfig` / `LDAIAgentConfig` / `LDAIJudgeConfig` | `LdAiCompletionConfig` / `LdAiAgentConfig` / `LdAiJudgeConfig` (`.create_tracker()`) |
| `LDAIConfigTracker` (`trackDuration`, `trackTokens`, `trackMetricsOf`, …) | `LdAiConfigTracker` (`track_duration`, `track_tokens`, `track_metrics_of`, …) |
| `LDGraphTracker` / `LDGraphTrackerImpl` | `LdGraphTracker` |
| `Judge`, `Evaluator`, `stripLegacyJudgeMessages` | `Judge`, `Evaluator`, `judge::strip_legacy_judge_messages` |
| `ManagedModel`, `ManagedAgent`, `ManagedAgentGraph` | same names |
| `AgentGraphDefinition`, `AgentGraphNode` (`traverse`, `reverseTraverse`, …) | same names (`traverse`, `reverse_traverse`, …) |
| `Runner`, `AgentGraphRunner`, `AIProvider`, `RunnerFactory` | traits `Runner`, `AgentGraphRunner`, `AiProvider`; registry `RunnerFactory` |

## Events: identical to JavaScript

LaunchDarkly's AI Config dashboards and judges read custom events by key. This crate
sends exactly the keys, metric values and data payloads of the JavaScript SDK 2.0.19,
verified by the ported test suites:

| Event | Metric value | Data |
|---|---|---|
| `$ld:ai:duration:total` | duration in ms | track data¹ |
| `$ld:ai:tokens:total` / `:input` / `:output` (each only when > 0) | token count | track data |
| `$ld:ai:tokens:ttf` | time to first token, ms | track data |
| `$ld:ai:generation:success` / `$ld:ai:generation:error` | 1 | track data |
| `$ld:ai:feedback:user:positive` / `:negative` | 1 | track data |
| `$ld:ai:tool_call` | 1 | track data + `toolKey` |
| judge metric key (e.g. `$ld:ai:judge:relevance`) | score 0–1 | track data + `judgeConfigKey` |
| `$ld:ai:graph:invocation_success` / `_failure`, `$ld:ai:graph:path` (+`path`), `$ld:ai:graph:duration:total`, `$ld:ai:graph:total_tokens` | 1 / ms / total | graph track data² |
| `$ld:ai:graph:redirect` (+`sourceKey`, `redirectedTarget`), `$ld:ai:graph:handoff_success` / `_failure` (+`sourceKey`, `targetKey`) | 1 | graph track data |
| `$ld:ai:usage:completion-config`, `-template`, `create-chat`, `create-agent`, `judge-config`, `-template`, `create-judge`, `agent-config`, `-template`, `agent-graph` | 1 | the config key |
| `$ld:ai:usage:agent-configs` | number of configs | number of configs |
| `$ld:ai:sdk:info` (context `{kind: "ld_ai", key: "ld-internal-tracking", anonymous: true}`) | 1 | `aiSdkName`, `aiSdkVersion`, `aiSdkLanguage` |

¹ `{runId, configKey, variationKey, version, modelName, providerName, modelVersion, modelKey?, graphKey?}`
² `{runId, graphKey, variationKey?, version}`

Resumption tokens are the same URL-safe, unpadded Base64 JSON as JavaScript, so a
tracker can be resumed across a JavaScript and a Rust service.

## Differences from the JavaScript SDK

- **SDK identity.** `$ld:ai:sdk:info` reports `aiSdkName: "unofficial-launchdarkly-server-sdk-ai"`, this crate's version and `aiSdkLanguage: "rust"`, not the JavaScript SDK's name, so LaunchDarkly never counts this port as its own SDK.
- **Providers.** JavaScript's `RunnerFactory` dynamically imports `@launchdarkly/server-sdk-ai-openai`, `-langchain` and `-vercel`. Rust cannot load packages at runtime, so `RunnerFactory` is a registry: register `AiProvider`s by name (`with_provider` for a named provider like `openai`, `with_multi_provider` for fallbacks like `langchain`/`vercel`). Selection is unchanged: an explicit `default_ai_provider` is the only one tried; otherwise the config's `provider.name` (if registered), then the multi-providers. With nothing registered, `create_model`/`create_agent`/`create_judge` return `None` with the same warning. `SupportedAIProvider` becomes any registered name.
- **Errors instead of exceptions.** Runner failures are `Result::Err(BoxError)`; `ManagedModel::run` and `track_metrics_of` return them after tracking the error, as JavaScript rethrows. `create_tracker`/`create_graph_tracker` return `Err` for a malformed token (JavaScript throws from `JSON.parse`). A judge error's `errorMessage` is the error's `Display` text; JavaScript's `'Unknown error'` for non-`Error` throws cannot occur.
- **Malformed templates.** mustache.js throws on templates such as an unclosed section, which rejects the JavaScript call. Here the config is returned with that message uninterpolated and a warning is logged.
- **Background work.** A JavaScript promise runs whether or not it is awaited. `ManagedResult::evaluations` and the metrics of `track_stream_metrics_of` are spawned on the current Tokio runtime (falling back to running when awaited, or a helper thread, outside one).
- **Streaming.** `track_stream_metrics_of(creator, extractor)` gives the extractor `&stream` and expects a `'static` future, since Rust cannot share the stream object the way JavaScript does.
- **Graph traversal values.** `traverse`/`reverse_traverse` are generic over the value type `T` the callback returns; the execution context is an `IndexMap<String, T>` with JavaScript's key order.
- **Key order of flag maps.** The core Rust SDK returns flag JSON as `serde_json::Value`, whose maps are sorted unless `serde_json`'s `preserve_order` feature is on somewhere in your build. That only affects the order of graph nodes from `nodes()`/`terminal_nodes()`/`get_parent_nodes()`; traversal order follows the declared edge arrays either way. Enable this crate's `preserve-order` feature to match JavaScript exactly.
- **Numbers.** Durations are `f64` milliseconds; token counts are `u64`. Mustache renders numbers with JavaScript's `Number#toString` rules.
- **Logging.** `LdClientMin::logger()` mirrors JavaScript's optional `logger`; the core Rust client's implementation forwards to the `log` crate. Messages carry the same text.
- **Not ported.** Nothing from the 2.0.19 source is omitted. The JavaScript examples depend on the Node SDK and provider packages; see `examples/` for the Rust equivalent of `create-judge`.

## How faithful is it?

The JavaScript test suites (`__tests__/`, ~6,000 lines) were ported test by test;
[TESTS.md](TESTS.md) lists every JavaScript test file, the number ported, and each test
that was adapted or not ported and why. Mustache rendering is checked against a fixture
produced by running mustache.js 4.2.0 itself.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). This is a derivative of
`@launchdarkly/server-sdk-ai` (Copyright Catamorphic, Co., Apache-2.0) and contains a
port of mustache.js (MIT). It is not a LaunchDarkly product.
