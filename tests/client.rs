mod common;

use std::sync::Arc;

use base64::Engine;
use common::{test_context, FakeLdClient, FakeProvider, FakeRunner};
use serde_json::{json, Value};
use unofficial_launchdarkly_server_sdk_ai::{
    init_track_context, LdAiAgentConfigDefault, LdAiAgentRequestConfig, LdAiClient,
    LdAiCompletionConfigDefault, LdAiJudgeConfigDefault, LdAiMetrics, LdMessage, LdModelConfig,
    LdProviderConfig, RunnerFactory, AI_SDK_LANGUAGE, AI_SDK_NAME, AI_SDK_VERSION,
};

fn new_client() -> (Arc<FakeLdClient>, LdAiClient) {
    let fake = FakeLdClient::new();
    let client = LdAiClient::new(fake.dyn_client());
    (fake, client)
}

fn model(value: Value) -> LdModelConfig {
    serde_json::from_value(value).unwrap()
}

fn disabled_default(mode: &str) -> Value {
    json!({ "_ldMeta": { "variationKey": "", "enabled": false, "mode": mode } })
}

fn only_variation_default(fake: &FakeLdClient, key: &str) -> Value {
    let calls = fake.variations();
    assert_eq!(calls.len(), 1, "expected one variation call, got {calls:?}");
    assert_eq!(calls[0].key, key);
    assert_eq!(calls[0].context, test_context());
    calls[0].default_value.clone()
}

fn assert_usage_tracked(fake: &FakeLdClient, event: &str, data: Value, metric: f64) {
    let matching: Vec<_> = fake
        .tracks_for(event)
        .into_iter()
        .filter(|t| {
            t.context == test_context()
                && t.data.as_ref() == Some(&data)
                && t.metric_value == Some(metric)
        })
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "expected one {event} event with {data}, got {:?}",
        fake.tracks()
    );
}

fn token(payload: Value) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload.to_string())
}

mod init_tracking {
    use super::*;

    #[tokio::test]
    async fn tracks_init_in_constructor_with_sdk_name_version_and_language_in_data() {
        let (fake, _client) = new_client();
        let first = fake.tracks().into_iter().next().expect("an init event");
        assert_eq!(first.key, "$ld:ai:sdk:info");
        assert_eq!(first.context, init_track_context());
        assert_eq!(
            first.data,
            Some(json!({
                "aiSdkName": AI_SDK_NAME,
                "aiSdkVersion": AI_SDK_VERSION,
                "aiSdkLanguage": AI_SDK_LANGUAGE,
            }))
        );
        assert_eq!(first.metric_value, Some(1.0));
        assert_eq!(AI_SDK_LANGUAGE, "rust");
    }
}

mod config_evaluation {
    use super::*;

    fn judge_flag(extra: Value) -> Value {
        let mut flag = json!({
            "enabled": true,
            "model": { "name": "gpt-4" },
            "provider": { "name": "openai" },
            "messages": [{ "role": "system", "content": "You are a judge." }],
            "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "judge" },
        });
        flag.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        flag
    }

    async fn evaluate_judge(flag: Value) -> (Arc<FakeLdClient>, Option<String>, bool) {
        let (fake, client) = new_client();
        fake.set_flag("test-judge", flag);
        let default = LdAiJudgeConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };
        let result = client
            .judge_config("test-judge", &test_context(), Some(default), None)
            .await;
        assert_eq!(
            only_variation_default(&fake, "test-judge"),
            disabled_default("judge")
        );
        let _ = result.create_tracker();
        (fake, result.evaluation_metric_key, result.enabled)
    }

    #[tokio::test]
    async fn evaluates_completion_config_successfully_with_variable_interpolation() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({
                "model": {
                    "name": "example-model",
                    "parameters": { "name": "imagination", "temperature": 0.7, "maxTokens": 4096 },
                },
                "provider": { "name": "example-provider" },
                "messages": [
                    { "role": "system", "content": "Hello {{name}}" },
                    { "role": "user", "content": "Score: {{score}}" },
                ],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
        );
        let default = LdAiCompletionConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };
        let variables = json!({ "name": "John", "score": 42 });

        let result = client
            .completion_config(
                "test-flag",
                &test_context(),
                Some(default),
                variables.as_object(),
                None,
            )
            .await;

        assert_eq!(
            only_variation_default(&fake, "test-flag"),
            disabled_default("completion")
        );
        assert_eq!(
            result.messages,
            Some(vec![
                LdMessage::system("Hello John"),
                LdMessage::user("Score: 42"),
            ])
        );
        let _ = result.create_tracker();
        assert!(result.enabled);
    }

    #[tokio::test]
    async fn includes_context_ldctx_in_variables_for_message_interpolation() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({
                "messages": [{ "role": "system", "content": "User key: {{ldctx.key}}" }],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
        );
        let default = LdAiCompletionConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };

        let result = client
            .completion_config("test-flag", &test_context(), Some(default), None, None)
            .await;

        assert_eq!(result.messages.unwrap()[0].content, "User key: test-user");
    }

    #[tokio::test]
    async fn evaluates_agent_config_successfully_with_instruction_interpolation() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-agent",
            json!({
                "model": {
                    "name": "example-model",
                    "parameters": { "temperature": 0.7, "maxTokens": 4096 },
                },
                "provider": { "name": "example-provider" },
                "instructions": "You are a helpful assistant. Your name is {{name}} and your score is {{score}}",
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "agent" },
            }),
        );
        let default = LdAiAgentConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };
        let variables = json!({ "name": "John", "score": 42 });

        let result = client
            .agent_config(
                "test-agent",
                &test_context(),
                Some(default),
                variables.as_object(),
                None,
            )
            .await;

        assert_eq!(
            only_variation_default(&fake, "test-agent"),
            disabled_default("agent")
        );
        assert_eq!(
            result.instructions.as_deref(),
            Some("You are a helpful assistant. Your name is John and your score is 42")
        );
        let _ = result.create_tracker();
        assert!(result.enabled);
    }

    #[tokio::test]
    async fn evaluates_judge_config_successfully_with_evaluation_metric_keys_legacy() {
        let (_fake, metric_key, enabled) = evaluate_judge(judge_flag(
            json!({ "evaluationMetricKeys": ["relevance", "accuracy"] }),
        ))
        .await;
        assert_eq!(metric_key.as_deref(), Some("relevance"));
        assert!(enabled);
    }

    #[tokio::test]
    async fn evaluates_judge_config_successfully_with_evaluation_metric_key() {
        let (_fake, metric_key, enabled) =
            evaluate_judge(judge_flag(json!({ "evaluationMetricKey": "relevance" }))).await;
        assert_eq!(metric_key.as_deref(), Some("relevance"));
        assert!(enabled);
    }

    #[tokio::test]
    async fn prioritizes_evaluation_metric_key_over_evaluation_metric_keys_when_both_are_provided()
    {
        let (_fake, metric_key, enabled) = evaluate_judge(judge_flag(json!({
            "evaluationMetricKey": "helpfulness",
            "evaluationMetricKeys": ["relevance", "accuracy"],
        })))
        .await;
        assert_eq!(metric_key.as_deref(), Some("helpfulness"));
        assert!(enabled);
    }

    #[tokio::test]
    async fn treats_empty_string_evaluation_metric_key_as_invalid_and_falls_back_to_evaluation_metric_keys(
    ) {
        let (_fake, metric_key, enabled) = evaluate_judge(judge_flag(json!({
            "evaluationMetricKey": "",
            "evaluationMetricKeys": ["relevance", "accuracy"],
        })))
        .await;
        assert_eq!(metric_key.as_deref(), Some("relevance"));
        assert!(enabled);
    }

    #[tokio::test]
    async fn skips_empty_and_whitespace_only_strings_in_evaluation_metric_keys_array() {
        let (_fake, metric_key, enabled) = evaluate_judge(judge_flag(json!({
            "evaluationMetricKeys": ["", "   ", "relevance", "accuracy"],
        })))
        .await;
        assert_eq!(metric_key.as_deref(), Some("relevance"));
        assert!(enabled);
    }

    #[tokio::test]
    async fn handles_mode_mismatch_by_returning_disabled_config() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({
                "model": { "name": "example-provider", "parameters": { "name": "imagination" } },
                "messages": [{ "role": "system", "content": "Hello" }],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "agent" },
            }),
        );
        let default = LdAiCompletionConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };

        let result = client
            .completion_config("test-flag", &test_context(), Some(default), None, None)
            .await;

        assert!(!result.enabled);
        let _ = result.create_tracker();
    }

    #[tokio::test]
    async fn handles_missing_metadata_mode_by_defaulting_to_completion_mode() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({
                "model": { "name": "example-provider", "parameters": { "name": "imagination" } },
                "messages": [{ "role": "system", "content": "Hello" }],
            }),
        );
        let default = LdAiCompletionConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };

        let result = client
            .completion_config("test-flag", &test_context(), Some(default), None, None)
            .await;

        assert!(!result.enabled);
        let _ = result.create_tracker();
        assert_eq!(result.messages, Some(vec![LdMessage::system("Hello")]));
        assert_eq!(
            result.model,
            Some(model(
                json!({ "name": "example-provider", "parameters": { "name": "imagination" } })
            ))
        );
    }

    #[tokio::test]
    async fn uses_default_value_when_flag_does_not_exist() {
        let (fake, client) = new_client();
        let default = LdAiCompletionConfigDefault {
            model: Some(model(
                json!({ "name": "default-model", "parameters": { "name": "default" } }),
            )),
            provider: Some(LdProviderConfig::named("default-provider")),
            messages: Some(vec![LdMessage::system("Default messages")]),
            enabled: Some(true),
            ..Default::default()
        };

        let result = client
            .completion_config(
                "non-existent-flag",
                &test_context(),
                Some(default.clone()),
                None,
                None,
            )
            .await;

        assert_eq!(result.model, default.model);
        assert_eq!(result.messages, default.messages);
        assert_eq!(result.provider, default.provider);
        let _ = result.create_tracker();
        assert!(result.enabled);
        let passed = only_variation_default(&fake, "non-existent-flag");
        assert_eq!(
            passed["model"],
            json!({ "name": "default-model", "parameters": { "name": "default" } })
        );
        assert_eq!(passed["provider"], json!({ "name": "default-provider" }));
    }
}

mod completion_config_method {
    use super::*;

    #[tokio::test]
    async fn calls_evaluate_with_correct_parameters_and_tracks_usage() {
        let (fake, client) = new_client();
        let default = LdAiCompletionConfigDefault {
            model: Some(model(
                json!({ "name": "test", "parameters": { "name": "test-model" } }),
            )),
            messages: Some(vec![]),
            enabled: Some(true),
            ..Default::default()
        };
        let variables = json!({ "var1": "value1" });

        let result = client
            .completion_config(
                "test-flag",
                &test_context(),
                Some(default),
                variables.as_object(),
                None,
            )
            .await;

        assert_usage_tracked(
            &fake,
            "$ld:ai:usage:completion-config",
            json!("test-flag"),
            1.0,
        );
        assert_eq!(
            only_variation_default(&fake, "test-flag"),
            json!({
                "_ldMeta": { "variationKey": "", "enabled": true, "mode": "completion" },
                "model": { "name": "test", "parameters": { "name": "test-model" } },
                "messages": [],
            })
        );
        assert!(result.enabled);
    }
}

mod agent_config_method {
    use super::*;

    #[tokio::test]
    async fn calls_evaluate_with_correct_parameters_and_tracks_usage() {
        let (fake, client) = new_client();
        let default = LdAiAgentConfigDefault {
            model: Some(model(
                json!({ "name": "test", "parameters": { "name": "test-model" } }),
            )),
            instructions: Some("You are a helpful assistant.".to_string()),
            enabled: Some(true),
            ..Default::default()
        };
        let variables = json!({ "var1": "value1" });

        let result = client
            .agent_config(
                "test-agent",
                &test_context(),
                Some(default),
                variables.as_object(),
                None,
            )
            .await;

        assert_usage_tracked(&fake, "$ld:ai:usage:agent-config", json!("test-agent"), 1.0);
        assert_eq!(
            only_variation_default(&fake, "test-agent"),
            json!({
                "_ldMeta": { "variationKey": "", "enabled": true, "mode": "agent" },
                "model": { "name": "test", "parameters": { "name": "test-model" } },
                "instructions": "You are a helpful assistant.",
            })
        );
        assert!(result.enabled);
        assert_eq!(
            result.instructions.as_deref(),
            Some("You are a helpful assistant.")
        );
        assert!(result.evaluator.judges().is_empty());
    }
}

mod agents_method {
    use super::*;

    #[tokio::test]
    async fn retrieves_multiple_agent_configs_with_interpolated_instructions() {
        let (fake, client) = new_client();
        fake.set_flag(
            "research-agent",
            json!({
                "model": { "name": "research-model", "parameters": { "temperature": 0.3, "maxTokens": 2048 } },
                "provider": { "name": "openai" },
                "instructions": "You are a research assistant specializing in {{topic}}.",
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "agent" },
            }),
        );
        fake.set_flag(
            "writing-agent",
            json!({
                "model": { "name": "writing-model", "parameters": { "temperature": 0.7, "maxTokens": 1024 } },
                "provider": { "name": "anthropic" },
                "instructions": "You are a writing assistant with {{style}} style.",
                "_ldMeta": { "variationKey": "v2", "enabled": true, "mode": "agent" },
            }),
        );
        let disabled = LdAiAgentConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };
        let requests = [
            LdAiAgentRequestConfig {
                key: "research-agent".to_string(),
                default_value: Some(disabled.clone()),
                variables: json!({ "topic": "climate change" }).as_object().cloned(),
            },
            LdAiAgentRequestConfig {
                key: "writing-agent".to_string(),
                default_value: Some(disabled),
                variables: json!({ "style": "academic" }).as_object().cloned(),
            },
        ];

        let result = client.agent_configs(&requests, &test_context()).await;

        assert_eq!(
            result.keys().collect::<Vec<_>>(),
            ["research-agent", "writing-agent"]
        );
        let expected = [
            (
                "research-agent",
                json!({ "name": "research-model", "parameters": { "temperature": 0.3, "maxTokens": 2048 } }),
                "openai",
                "You are a research assistant specializing in climate change.",
            ),
            (
                "writing-agent",
                json!({ "name": "writing-model", "parameters": { "temperature": 0.7, "maxTokens": 1024 } }),
                "anthropic",
                "You are a writing assistant with academic style.",
            ),
        ];
        for (key, expected_model, provider, instructions) in expected {
            let config = &result[key];
            assert_eq!(config.key, key);
            assert_eq!(config.model, Some(model(expected_model)));
            assert_eq!(config.provider, Some(LdProviderConfig::named(provider)));
            assert_eq!(config.instructions.as_deref(), Some(instructions));
            assert!(config.enabled);
            assert!(config.evaluator.judges().is_empty());
            assert!(config.tools.is_none());
            assert!(config.judge_configuration.is_none());
            let _ = config.create_tracker();
        }

        assert_usage_tracked(&fake, "$ld:ai:usage:agent-configs", json!(2), 2.0);
    }

    #[tokio::test]
    async fn handles_empty_agent_configs_array() {
        let (fake, client) = new_client();

        let result = client.agent_configs(&[], &test_context()).await;

        assert!(result.is_empty());
        assert_usage_tracked(&fake, "$ld:ai:usage:agent-configs", json!(0), 0.0);
    }
}

mod judge_config_method {
    use super::*;

    #[tokio::test]
    async fn calls_evaluate_with_correct_parameters_and_tracks_usage() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-judge",
            json!({
                "model": { "name": "gpt-4" },
                "provider": { "name": "openai" },
                "evaluationMetricKey": "relevance",
                "messages": [{ "role": "system", "content": "You are a judge for {{metric}}." }],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "judge" },
            }),
        );
        let default = LdAiJudgeConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };
        let variables = json!({ "metric": "relevance" });

        let result = client
            .judge_config(
                "test-judge",
                &test_context(),
                Some(default),
                variables.as_object(),
            )
            .await;

        assert_usage_tracked(&fake, "$ld:ai:usage:judge-config", json!("test-judge"), 1.0);
        assert_eq!(
            only_variation_default(&fake, "test-judge"),
            disabled_default("judge")
        );
        assert!(result.enabled);
        assert_eq!(result.model, Some(LdModelConfig::named("gpt-4")));
        assert_eq!(result.provider, Some(LdProviderConfig::named("openai")));
        assert_eq!(result.evaluation_metric_key.as_deref(), Some("relevance"));
        assert_eq!(
            result.messages,
            Some(vec![LdMessage::system("You are a judge for relevance.")])
        );
    }

    #[tokio::test]
    async fn strips_legacy_judge_template_messages_from_the_returned_config() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-judge",
            json!({
                "model": { "name": "gpt-4" },
                "provider": { "name": "openai" },
                "evaluationMetricKey": "relevance",
                "messages": [
                    { "role": "system", "content": "You are a judge." },
                    { "role": "assistant", "content": "{{message_history}}" },
                    { "role": "user", "content": "Evaluate: {{response_to_evaluate}}" },
                ],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "judge" },
            }),
        );
        let default = LdAiJudgeConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };

        let result = client
            .judge_config("test-judge", &test_context(), Some(default), None)
            .await;

        assert_eq!(
            result.messages,
            Some(vec![LdMessage::system("You are a judge.")])
        );
    }
}

mod create_judge_method {
    use super::*;

    fn judge_flag(enabled: bool) -> Value {
        json!({
            "model": { "name": "gpt-4" },
            "provider": { "name": "openai" },
            "evaluationMetricKey": "relevance",
            "messages": [{ "role": "system", "content": "You are a judge." }],
            "_ldMeta": { "variationKey": "v1", "enabled": enabled, "mode": "judge" },
        })
    }

    fn client_with(provider: Arc<FakeProvider>) -> (Arc<FakeLdClient>, LdAiClient) {
        let fake = FakeLdClient::new();
        let client = LdAiClient::with_runner_factory(
            fake.dyn_client(),
            RunnerFactory::new().with_provider("openai", provider),
        );
        (fake, client)
    }

    fn disabled() -> Option<LdAiJudgeConfigDefault> {
        Some(LdAiJudgeConfigDefault {
            enabled: Some(false),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn initializes_judge_successfully() {
        let provider = FakeProvider::with_model(FakeRunner::returning("", LdAiMetrics::new(true)));
        let (fake, client) = client_with(Arc::clone(&provider));
        fake.set_flag("test-judge", judge_flag(true));

        let judge = client
            .create_judge("test-judge", &test_context(), disabled(), None, None, 1.0)
            .await
            .expect("a judge");

        assert_usage_tracked(&fake, "$ld:ai:usage:create-judge", json!("test-judge"), 1.0);
        assert_eq!(
            only_variation_default(&fake, "test-judge"),
            disabled_default("judge")
        );
        assert_eq!(
            *provider.model_calls.lock().unwrap(),
            vec![("test-judge".to_string(), false)]
        );
        let config = judge.get_ai_config();
        assert_eq!(config.key, "test-judge");
        assert!(config.enabled);
        assert_eq!(config.model, Some(LdModelConfig::named("gpt-4")));
        assert_eq!(config.provider, Some(LdProviderConfig::named("openai")));
        assert_eq!(config.evaluation_metric_key.as_deref(), Some("relevance"));
        assert_eq!(
            config.messages,
            Some(vec![LdMessage::system("You are a judge.")])
        );
        assert_eq!(judge.sample_rate(), 1.0);
    }

    #[tokio::test]
    async fn returns_undefined_when_judge_configuration_is_disabled() {
        let provider = FakeProvider::with_model(FakeRunner::returning("", LdAiMetrics::new(true)));
        let (fake, client) = client_with(Arc::clone(&provider));
        fake.set_flag("test-judge", judge_flag(false));

        let judge = client
            .create_judge("test-judge", &test_context(), disabled(), None, None, 1.0)
            .await;

        assert!(judge.is_none());
        assert!(provider.model_calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn returns_undefined_when_runner_factory_create_model_returns_undefined() {
        let provider = FakeProvider::empty();
        let (fake, client) = client_with(Arc::clone(&provider));
        fake.set_flag("test-judge", judge_flag(true));

        let judge = client
            .create_judge("test-judge", &test_context(), disabled(), None, None, 1.0)
            .await;

        assert!(judge.is_none());
        assert_eq!(
            *provider.model_calls.lock().unwrap(),
            vec![("test-judge".to_string(), false)]
        );
    }

    #[tokio::test]
    async fn handles_errors_gracefully() {
        let provider = FakeProvider::with_model(FakeRunner::returning("", LdAiMetrics::new(true)));
        let (fake, client) = client_with(Arc::clone(&provider));
        fake.set_flag("test-judge", json!("Judge configuration error"));

        let judge = client
            .create_judge("test-judge", &test_context(), disabled(), None, None, 1.0)
            .await;

        assert!(judge.is_none());
        assert!(provider.model_calls.lock().unwrap().is_empty());
    }
}

mod create_tracker_method {
    use super::*;

    #[tokio::test]
    async fn reconstructs_a_tracker_from_a_resumption_token() {
        let (_fake, client) = new_client();
        let token = token(json!({
            "runId": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            "configKey": "my-config",
            "variationKey": "v1",
            "version": 3,
        }));

        let tracker = client.create_tracker(&token, &test_context()).unwrap();

        let data = tracker.get_track_data();
        assert_eq!(data.run_id, "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11");
        assert_eq!(data.config_key, "my-config");
        assert_eq!(data.variation_key, "v1");
        assert_eq!(data.version, 3);
        assert_eq!(data.model_version, 1);
        let serialized = serde_json::to_value(&data).unwrap();
        assert!(serialized.get("modelKey").is_none());
    }
}

mod model_key_and_model_version_tracking {
    use super::*;

    async fn completion(
        meta: Value,
    ) -> unofficial_launchdarkly_server_sdk_ai::LdAiCompletionConfig {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({
                "model": { "name": "gpt-4" },
                "provider": { "name": "openai" },
                "messages": [],
                "_ldMeta": meta,
            }),
        );
        client
            .completion_config("test-flag", &test_context(), None, None, None)
            .await
    }

    #[tokio::test]
    async fn stamps_model_key_and_model_version_on_tracker_from_flag_payload() {
        let result = completion(json!({
            "variationKey": "v1",
            "enabled": true,
            "mode": "completion",
            "version": 7,
            "modelKey": "my-model",
            "modelVersion": 2,
        }))
        .await;

        let data = result.create_tracker().get_track_data();
        assert_eq!(data.model_key.as_deref(), Some("my-model"));
        assert_eq!(data.model_version, 2);
    }

    #[tokio::test]
    async fn defaults_model_version_to_1_and_omits_model_key_when_absent_from_flag_payload() {
        let result = completion(json!({
            "variationKey": "v1",
            "enabled": true,
            "mode": "completion",
            "version": 7,
        }))
        .await;

        let data = result.create_tracker().get_track_data();
        let serialized = serde_json::to_value(&data).unwrap();
        assert!(serialized.get("modelKey").is_none());
        assert_eq!(data.model_version, 1);
    }

    #[tokio::test]
    async fn does_not_expose_model_key_or_model_version_on_config_model() {
        let result = completion(json!({
            "variationKey": "v1",
            "enabled": true,
            "mode": "completion",
            "modelKey": "my-model",
            "modelVersion": 2,
        }))
        .await;

        assert_eq!(result.model, Some(LdModelConfig::named("gpt-4")));
        assert_eq!(
            serde_json::to_value(result.model.unwrap()).unwrap(),
            json!({ "name": "gpt-4" })
        );
    }
}

mod optional_default_values {
    use super::*;

    fn assert_disabled_default_passed(fake: &FakeLdClient, key: &str) {
        let passed = only_variation_default(fake, key);
        assert_eq!(passed["_ldMeta"]["enabled"], json!(false));
    }

    #[tokio::test]
    async fn uses_a_disabled_completion_config_when_no_default_is_provided() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({ "_ldMeta": { "variationKey": "", "enabled": false, "version": 1, "mode": "completion" } }),
        );

        let result = client
            .completion_config("test-flag", &test_context(), None, None, None)
            .await;

        assert_disabled_default_passed(&fake, "test-flag");
        assert!(!result.enabled);
    }

    #[tokio::test]
    async fn uses_a_disabled_agent_config_when_no_default_is_provided() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-agent",
            json!({
                "_ldMeta": { "variationKey": "", "enabled": false, "version": 1, "mode": "agent" },
                "instructions": "",
            }),
        );

        let result = client
            .agent_config("test-agent", &test_context(), None, None, None)
            .await;

        assert_disabled_default_passed(&fake, "test-agent");
        assert!(!result.enabled);
    }

    #[tokio::test]
    async fn uses_a_disabled_judge_config_when_no_default_is_provided() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-judge",
            json!({
                "_ldMeta": { "variationKey": "", "enabled": false, "version": 1, "mode": "judge" },
                "messages": [],
                "evaluationMetricKeys": [],
            }),
        );

        let result = client
            .judge_config("test-judge", &test_context(), None, None)
            .await;

        assert_disabled_default_passed(&fake, "test-judge");
        assert!(!result.enabled);
    }
}

mod tools_map_support {
    use super::*;

    fn tools_json<T: serde::Serialize>(tools: &T) -> Value {
        serde_json::to_value(tools).unwrap()
    }

    async fn completion_with(
        flag: Value,
        default: LdAiCompletionConfigDefault,
    ) -> unofficial_launchdarkly_server_sdk_ai::LdAiCompletionConfig {
        let (fake, client) = new_client();
        fake.set_flag("test-flag", flag);
        client
            .completion_config("test-flag", &test_context(), Some(default), None, None)
            .await
    }

    fn disabled() -> LdAiCompletionConfigDefault {
        LdAiCompletionConfigDefault {
            enabled: Some(false),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn includes_tools_map_in_completion_config_from_flag_variation() {
        let tools = json!({
            "web-search-tool": {
                "name": "web-search-tool",
                "type": "function",
                "parameters": { "type": "object", "properties": {}, "required": [] },
                "customParameters": { "some-custom-parameter": "some-custom-value" },
            },
        });
        let result = completion_with(
            json!({
                "model": { "name": "example-model" },
                "tools": tools,
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
            disabled(),
        )
        .await;

        assert_eq!(tools_json(&result.tools.unwrap()), tools);
    }

    #[tokio::test]
    async fn includes_tools_map_in_agent_config_from_flag_variation() {
        let (fake, client) = new_client();
        let tools = json!({
            "search-tool": {
                "name": "search-tool",
                "type": "function",
                "customParameters": { "maxResults": 10 },
            },
        });
        fake.set_flag(
            "test-agent",
            json!({
                "model": { "name": "example-model" },
                "instructions": "You are a helpful agent.",
                "tools": tools,
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "agent" },
            }),
        );
        let default = LdAiAgentConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };

        let result = client
            .agent_config("test-agent", &test_context(), Some(default), None, None)
            .await;

        assert_eq!(tools_json(&result.tools.unwrap()), tools);
    }

    #[tokio::test]
    async fn returns_undefined_tools_when_variation_has_no_tools() {
        let default_tools = serde_json::from_value(json!({
            "default-tool": {
                "name": "default-tool",
                "type": "function",
                "customParameters": { "priority": "high" },
            },
        }))
        .unwrap();
        let default = LdAiCompletionConfigDefault {
            enabled: Some(true),
            tools: Some(default_tools),
            ..Default::default()
        };
        let result = completion_with(
            json!({ "_ldMeta": { "enabled": true, "mode": "completion", "variationKey": "" } }),
            default,
        )
        .await;

        assert!(result.tools.is_none());
    }

    #[tokio::test]
    async fn returns_undefined_tools_when_no_tools_are_configured() {
        let result = completion_with(
            json!({
                "model": { "name": "example-model" },
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
            disabled(),
        )
        .await;

        assert!(result.tools.is_none());
    }

    #[tokio::test]
    async fn converts_model_parameters_tools_array_to_map_when_root_tools_is_absent() {
        let result = completion_with(
            json!({
                "model": {
                    "name": "example-model",
                    "parameters": {
                        "tools": [
                            {
                                "name": "search",
                                "type": "function",
                                "description": "Search the web",
                                "parameters": { "type": "object", "properties": {}, "required": [] },
                            },
                            {
                                "name": "get_weather",
                                "type": "function",
                                "description": "Get weather for a location",
                            },
                        ],
                    },
                },
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
            disabled(),
        )
        .await;

        let tools = result.tools.expect("tools");
        let search = &tools["search"];
        assert_eq!(search.name, "search");
        assert_eq!(search.tool_type.as_deref(), Some("function"));
        assert_eq!(search.description.as_deref(), Some("Search the web"));
        assert_eq!(
            search.parameters.clone().map(Value::Object),
            Some(json!({ "type": "object", "properties": {}, "required": [] }))
        );
        let weather = &tools["get_weather"];
        assert_eq!(weather.name, "get_weather");
        assert_eq!(
            weather.description.as_deref(),
            Some("Get weather for a location")
        );
    }

    #[tokio::test]
    async fn uses_root_tools_map_over_model_parameters_tools_array_when_both_are_present() {
        let root_tools = json!({ "root-tool": { "name": "root-tool", "type": "function" } });
        let result = completion_with(
            json!({
                "model": {
                    "name": "example-model",
                    "parameters": { "tools": [{ "name": "params-tool", "type": "function" }] },
                },
                "tools": root_tools,
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
            disabled(),
        )
        .await;

        assert_eq!(tools_json(&result.tools.unwrap()), root_tools);
    }

    #[tokio::test]
    async fn skips_model_parameters_tools_array_entries_without_a_name() {
        let result = completion_with(
            json!({
                "model": {
                    "name": "example-model",
                    "parameters": {
                        "tools": [
                            { "name": "valid-tool", "type": "function" },
                            { "type": "function", "description": "no name on this one" },
                        ],
                    },
                },
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
            disabled(),
        )
        .await;

        let tools = result.tools.expect("tools");
        assert_eq!(tools["valid-tool"].name, "valid-tool");
        assert_eq!(tools.len(), 1);
    }

    #[tokio::test]
    async fn returns_undefined_when_model_parameters_tools_is_not_an_array() {
        let result = completion_with(
            json!({
                "model": { "name": "example-model", "parameters": { "tools": "not-an-array" } },
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
            disabled(),
        )
        .await;

        assert!(result.tools.is_none());
    }
}

mod template_methods {
    use super::*;

    #[tokio::test]
    async fn completion_config_template_preserves_mustache_placeholders_in_messages() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({
                "model": { "name": "gpt-4" },
                "provider": { "name": "openai" },
                "messages": [
                    { "role": "system", "content": "Hello {{name}}" },
                    { "role": "user", "content": "Score: {{score}}" },
                ],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
        );
        let default = LdAiCompletionConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };

        let result = client
            .completion_config_template("test-flag", &test_context(), Some(default), None)
            .await;

        assert_eq!(
            result.messages,
            Some(vec![
                LdMessage::system("Hello {{name}}"),
                LdMessage::user("Score: {{score}}"),
            ])
        );
        assert!(result.enabled);
        let _ = result.create_tracker();
    }

    #[tokio::test]
    async fn completion_config_template_emits_the_correct_usage_tracking_event() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({
                "messages": [{ "role": "system", "content": "Hello {{name}}" }],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
        );

        client
            .completion_config_template("test-flag", &test_context(), None, None)
            .await;

        assert_usage_tracked(
            &fake,
            "$ld:ai:usage:completion-config-template",
            json!("test-flag"),
            1.0,
        );
    }

    #[tokio::test]
    async fn completion_config_template_uses_disabled_default_when_no_default_value_is_provided() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({ "_ldMeta": { "variationKey": "v1", "enabled": false, "mode": "completion" } }),
        );

        let result = client
            .completion_config_template("test-flag", &test_context(), None, None)
            .await;

        assert!(!result.enabled);
    }

    #[tokio::test]
    async fn agent_config_template_preserves_mustache_placeholders_in_instructions() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-agent",
            json!({
                "model": { "name": "gpt-4" },
                "provider": { "name": "openai" },
                "instructions": "You are a research assistant specializing in {{topic}}.",
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "agent" },
            }),
        );
        let default = LdAiAgentConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };

        let result = client
            .agent_config_template("test-agent", &test_context(), Some(default), None)
            .await;

        assert_eq!(
            result.instructions.as_deref(),
            Some("You are a research assistant specializing in {{topic}}.")
        );
        assert!(result.enabled);
        let _ = result.create_tracker();
    }

    #[tokio::test]
    async fn agent_config_template_emits_the_correct_usage_tracking_event() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-agent",
            json!({
                "instructions": "You are a {{role}}.",
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "agent" },
            }),
        );

        client
            .agent_config_template("test-agent", &test_context(), None, None)
            .await;

        assert_usage_tracked(
            &fake,
            "$ld:ai:usage:agent-config-template",
            json!("test-agent"),
            1.0,
        );
    }

    #[tokio::test]
    async fn agent_config_template_uses_disabled_default_when_no_default_value_is_provided() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-agent",
            json!({ "_ldMeta": { "variationKey": "v1", "enabled": false, "mode": "agent" } }),
        );

        let result = client
            .agent_config_template("test-agent", &test_context(), None, None)
            .await;

        assert!(!result.enabled);
    }

    #[tokio::test]
    async fn judge_config_template_preserves_mustache_placeholders_in_messages() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-judge",
            json!({
                "model": { "name": "gpt-4" },
                "provider": { "name": "openai" },
                "evaluationMetricKey": "relevance",
                "messages": [
                    { "role": "system", "content": "You are a judge evaluating {{criteria}}." },
                    { "role": "user", "content": "Score: {{score}}" },
                ],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "judge" },
            }),
        );
        let default = LdAiJudgeConfigDefault {
            enabled: Some(false),
            ..Default::default()
        };

        let result = client
            .judge_config_template("test-judge", &test_context(), Some(default))
            .await;

        assert_eq!(
            result.messages,
            Some(vec![
                LdMessage::system("You are a judge evaluating {{criteria}}."),
                LdMessage::user("Score: {{score}}"),
            ])
        );
        assert!(result.enabled);
        let _ = result.create_tracker();
    }

    #[tokio::test]
    async fn judge_config_template_emits_the_correct_usage_tracking_event() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-judge",
            json!({
                "evaluationMetricKey": "relevance",
                "messages": [{ "role": "system", "content": "You are a {{role}}." }],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "judge" },
            }),
        );

        client
            .judge_config_template("test-judge", &test_context(), None)
            .await;

        assert_usage_tracked(
            &fake,
            "$ld:ai:usage:judge-config-template",
            json!("test-judge"),
            1.0,
        );
    }

    #[tokio::test]
    async fn judge_config_template_uses_disabled_default_when_no_default_value_is_provided() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-judge",
            json!({ "_ldMeta": { "variationKey": "v1", "enabled": false, "mode": "judge" } }),
        );

        let result = client
            .judge_config_template("test-judge", &test_context(), None)
            .await;

        assert!(!result.enabled);
    }

    #[tokio::test]
    async fn completion_config_template_does_not_apply_ldctx_interpolation() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-flag",
            json!({
                "messages": [{ "role": "system", "content": "User: {{ldctx.key}}" }],
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "completion" },
            }),
        );

        let result = client
            .completion_config_template("test-flag", &test_context(), None, None)
            .await;

        assert_eq!(result.messages.unwrap()[0].content, "User: {{ldctx.key}}");
    }

    #[tokio::test]
    async fn agent_config_template_does_not_apply_ldctx_interpolation() {
        let (fake, client) = new_client();
        fake.set_flag(
            "test-agent",
            json!({
                "instructions": "Context key: {{ldctx.key}}",
                "_ldMeta": { "variationKey": "v1", "enabled": true, "mode": "agent" },
            }),
        );

        let result = client
            .agent_config_template("test-agent", &test_context(), None, None)
            .await;

        assert_eq!(
            result.instructions.as_deref(),
            Some("Context key: {{ldctx.key}}")
        );
    }
}
