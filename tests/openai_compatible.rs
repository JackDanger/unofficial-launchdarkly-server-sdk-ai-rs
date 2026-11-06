#![cfg(feature = "openai-compatible")]

mod common;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use unofficial_launchdarkly_server_sdk_ai::openai_compatible::{
    OpenAiCompatibleProvider, STRUCTURED_OUTPUT_NAME,
};
use unofficial_launchdarkly_server_sdk_ai::{
    judge, AiProvider, LdAiClient, LdAiCompletionConfig, LdAiConfigTracker,
    LdAiConfigTrackerParams, LdMessage, LdModelConfig, LdProviderConfig, ModelConfigRef,
    RunnerFactory, TrackerFactory,
};

use common::{test_context, FakeLdClient};

#[derive(Debug, Clone)]
struct Request {
    head: String,
    body: Value,
}

struct MockServer {
    base_url: String,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl MockServer {
    async fn start(responses: Vec<(u16, Value)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let recorded = Arc::clone(&requests);
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head, body_start) = loop {
                    let read = socket.read(&mut chunk).await.unwrap();
                    buffer.extend_from_slice(&chunk[..read]);
                    if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break (String::from_utf8_lossy(&buffer[..end]).to_string(), end + 4);
                    }
                };
                let length: usize = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while buffer.len() < body_start + length {
                    let read = socket.read(&mut chunk).await.unwrap();
                    buffer.extend_from_slice(&chunk[..read]);
                }
                let body = serde_json::from_slice(&buffer[body_start..body_start + length])
                    .unwrap_or(Value::Null);
                recorded.lock().unwrap().push(Request { head, body });
                let (status, response) = queue
                    .lock()
                    .unwrap()
                    .pop_front()
                    .unwrap_or((500, json!({"error": "no response queued"})));
                let payload = response.to_string();
                let reply = format!(
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                socket.write_all(reply.as_bytes()).await.unwrap();
                socket.shutdown().await.ok();
            }
        });
        Self { base_url, requests }
    }

    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

fn completion(content: &str) -> Value {
    json!({
        "id": "cmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    })
}

fn completion_config(
    client: &Arc<FakeLdClient>,
    model: Option<LdModelConfig>,
) -> LdAiCompletionConfig {
    let ld = client.dyn_client();
    LdAiCompletionConfig {
        key: "chat".to_string(),
        enabled: true,
        model,
        provider: Some(LdProviderConfig::named("openai")),
        messages: Some(vec![LdMessage::system("You are terse.")]),
        judge_configuration: None,
        tools: None,
        tracker_factory: TrackerFactory::new(move || {
            LdAiConfigTracker::new(
                Arc::clone(&ld),
                test_context(),
                LdAiConfigTrackerParams::new("run", "chat", "v1", 1),
            )
        }),
        evaluator: Default::default(),
    }
}

fn model_with_params() -> LdModelConfig {
    let mut model = LdModelConfig::named("qwen3-8b");
    model.parameters = json!({"temperature": 0.2, "max_tokens": 64, "tools": [{"name": "t"}]})
        .as_object()
        .cloned();
    model
}

#[tokio::test]
async fn sends_config_messages_parameters_and_auth_and_maps_the_response() {
    let server = MockServer::start(vec![(
        200,
        json!({
            "choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        }),
    )])
    .await;
    let client = FakeLdClient::new();
    let config = completion_config(&client, Some(model_with_params()));
    let provider = OpenAiCompatibleProvider::new(format!("{}/", server.base_url))
        .with_api_key(Some("sk-test".to_string()));

    let runner = provider
        .create_model(ModelConfigRef::Completion(&config), true)
        .await
        .unwrap();
    let result = runner.run("hello", None).await.unwrap();

    assert_eq!(result.content, "hi");
    assert!(result.metrics.success);
    let tokens = result.metrics.tokens.unwrap();
    assert_eq!((tokens.total, tokens.input, tokens.output), (18, 11, 7));
    assert_eq!(result.metrics.tool_calls, Some(vec!["lookup".to_string()]));
    assert!(result.parsed.is_none());

    let request = &server.requests()[0];
    assert!(request.head.starts_with("POST /v1/chat/completions "));
    assert!(request
        .head
        .lines()
        .any(|l| l.eq_ignore_ascii_case("authorization: Bearer sk-test")));
    assert_eq!(
        request.body,
        json!({
            "temperature": 0.2,
            "max_tokens": 64,
            "model": "qwen3-8b",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hello"}
            ]
        })
    );
}

#[tokio::test]
async fn requests_json_schema_structured_output_and_parses_it() {
    let server = MockServer::start(vec![(
        200,
        completion(r#"{"score": 0.75, "reasoning": "mostly relevant"}"#),
    )])
    .await;
    let client = FakeLdClient::new();
    let config = completion_config(&client, Some(LdModelConfig::named("m")));
    let runner = OpenAiCompatibleProvider::new(&server.base_url)
        .create_model(ModelConfigRef::Completion(&config), false)
        .await
        .unwrap();

    let result = runner
        .run("evaluate", Some(judge::evaluation_schema()))
        .await
        .unwrap();

    assert_eq!(
        result.parsed.unwrap(),
        json!({"score": 0.75, "reasoning": "mostly relevant"})
            .as_object()
            .unwrap()
            .clone()
    );
    assert_eq!(
        server.requests()[0].body["response_format"],
        json!({
            "type": "json_schema",
            "json_schema": {"name": STRUCTURED_OUTPUT_NAME, "schema": judge::evaluation_schema(), "strict": true}
        })
    );
}

#[tokio::test]
async fn multi_turn_runners_send_prior_turns_and_single_turn_runners_do_not() {
    let server = MockServer::start(vec![
        (200, completion("first")),
        (200, completion("second")),
        (200, completion("a")),
        (200, completion("b")),
    ])
    .await;
    let client = FakeLdClient::new();
    let config = completion_config(&client, Some(LdModelConfig::named("m")));
    let provider = OpenAiCompatibleProvider::new(&server.base_url);

    let chat = provider
        .create_model(ModelConfigRef::Completion(&config), true)
        .await
        .unwrap();
    chat.run("one", None).await.unwrap();
    chat.run("two", None).await.unwrap();
    let stateless = provider
        .create_model(ModelConfigRef::Completion(&config), false)
        .await
        .unwrap();
    stateless.run("x", None).await.unwrap();
    stateless.run("y", None).await.unwrap();

    let requests = server.requests();
    assert_eq!(
        requests[1].body["messages"],
        json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"}
        ])
    );
    assert_eq!(
        requests[3].body["messages"],
        json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "y"}
        ])
    );
}

#[tokio::test]
async fn non_success_status_is_an_error() {
    let server = MockServer::start(vec![(503, json!({"error": "overloaded"}))]).await;
    let client = FakeLdClient::new();
    let config = completion_config(&client, Some(LdModelConfig::named("m")));
    let runner = OpenAiCompatibleProvider::new(&server.base_url)
        .create_model(ModelConfigRef::Completion(&config), true)
        .await
        .unwrap();

    let err = runner.run("hello", None).await.unwrap_err();
    assert!(err.to_string().contains("503"), "{err}");
}

#[tokio::test]
async fn declines_configs_without_a_model_name() {
    let client = FakeLdClient::new();
    let provider = OpenAiCompatibleProvider::new("http://127.0.0.1:9/v1");
    for model in [None, Some(LdModelConfig::named(""))] {
        let config = completion_config(&client, model);
        assert!(provider
            .create_model(ModelConfigRef::Completion(&config), true)
            .await
            .is_none());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn judges_a_managed_model_end_to_end_against_a_self_hosted_endpoint() {
    let server = MockServer::start(vec![
        (200, completion("Paris.")),
        (200, completion(r#"{"score": 0.9, "reasoning": "correct"}"#)),
    ])
    .await;
    let ld = FakeLdClient::new();
    ld.set_flag(
        "chat",
        json!({
            "_ldMeta": {"variationKey": "v1", "enabled": true, "mode": "completion", "version": 3},
            "model": {"name": "qwen3-8b"},
            "provider": {"name": "openai"},
            "messages": [{"role": "system", "content": "Answer {{ldctx.key}} briefly."}],
            "judgeConfiguration": {"judges": [{"key": "relevance", "samplingRate": 1.0}]}
        }),
    );
    ld.set_flag(
        "relevance",
        json!({
            "_ldMeta": {"variationKey": "j1", "enabled": true, "mode": "judge"},
            "model": {"name": "qwen3-8b"},
            "provider": {"name": "openai"},
            "messages": [
                {"role": "system", "content": "Score relevance."},
                {"role": "user", "content": "{{response_to_evaluate}}"}
            ],
            "evaluationMetricKey": "$ld:ai:judge:relevance"
        }),
    );
    let factory = RunnerFactory::new().with_provider(
        "openai",
        Arc::new(OpenAiCompatibleProvider::new(&server.base_url)),
    );
    let ai = LdAiClient::with_runner_factory(ld.dyn_client(), factory);

    let model = ai
        .create_model("chat", &test_context(), None, None, None)
        .await
        .expect("model is enabled and the provider is registered");
    let result = model.run("Capital of France?").await.unwrap();
    assert_eq!(result.content, "Paris.");
    let evaluations = result.evaluations.await;

    assert_eq!(evaluations.len(), 1);
    assert_eq!(evaluations[0].score, Some(0.9));
    assert_eq!(
        evaluations[0].metric_key.as_deref(),
        Some("$ld:ai:judge:relevance")
    );

    let requests = server.requests();
    assert_eq!(
        requests[0].body["messages"][0]["content"],
        json!("Answer test-user briefly.")
    );
    assert_eq!(
        requests[1].body["messages"],
        json!([
            {"role": "system", "content": "Score relevance."},
            {"role": "user", "content": "MESSAGE HISTORY:\nCapital of France?\n\nRESPONSE TO EVALUATE:\nParis."}
        ])
    );

    let judged = ld.tracks_for("$ld:ai:judge:relevance");
    assert_eq!(judged.len(), 1);
    assert_eq!(judged[0].metric_value, Some(0.9));
    let data = judged[0].data.as_ref().unwrap();
    assert_eq!(data["configKey"], json!("chat"));
    assert_eq!(data["judgeConfigKey"], json!("relevance"));
    assert_eq!(data["version"], json!(3));
    assert_eq!(ld.tracks_for("$ld:ai:tokens:total").len(), 2);
}
