mod common;

use std::sync::Arc;

use common::{test_context, FakeLdClient, FakeRunner, RecordingLogger, TEST_RUN_ID};
use serde_json::{json, Value};
use unofficial_launchdarkly_server_sdk_ai::judge::{
    build_evaluation_input, evaluation_schema, parse_evaluation_response,
    strip_legacy_judge_messages,
};
use unofficial_launchdarkly_server_sdk_ai::{
    Judge, LdAiConfigTracker, LdAiConfigTrackerParams, LdAiJudgeConfig, LdAiMetrics, LdJudgeResult,
    LdLogger, LdMessage, LdMessageRole, LdModelConfig, LdProviderConfig, LdTokenUsage, Runner,
    RunnerResult, TrackerFactory,
};

mod strip_legacy_judge_messages {
    use super::*;

    #[test]
    fn strips_assistant_messages_containing_message_history() {
        let messages = vec![
            LdMessage::system("You are a judge."),
            LdMessage::assistant("Here is the history: {{message_history}}"),
        ];
        let result = strip_legacy_judge_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, LdMessageRole::System);
    }

    #[test]
    fn strips_user_messages_containing_response_to_evaluate() {
        let messages = vec![
            LdMessage::system("You are a judge."),
            LdMessage::user("Evaluate: {{response_to_evaluate}}"),
        ];
        let result = strip_legacy_judge_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, LdMessageRole::System);
    }

    #[test]
    fn strips_all_legacy_template_messages_from_a_typical_legacy_config() {
        let messages = vec![
            LdMessage::system("You are a judge."),
            LdMessage::assistant("{{message_history}}"),
            LdMessage::user("{{response_to_evaluate}}"),
        ];
        let result = strip_legacy_judge_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, LdMessageRole::System);
    }

    #[test]
    fn does_not_strip_system_messages_even_when_they_contain_template_variables() {
        let messages = vec![LdMessage::system(
            "Judge using {{message_history}} and {{response_to_evaluate}}.",
        )];
        let result = strip_legacy_judge_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, LdMessageRole::System);
    }

    #[test]
    fn leaves_non_system_messages_without_template_variables_untouched() {
        let messages = vec![
            LdMessage::system("You are a judge."),
            LdMessage::user("This is a regular message."),
        ];
        assert_eq!(strip_legacy_judge_messages(&messages).len(), 2);
    }

    #[test]
    fn returns_an_empty_array_for_an_empty_input() {
        assert_eq!(strip_legacy_judge_messages(&[]), Vec::<LdMessage>::new());
    }

    #[test]
    fn passes_a_new_style_system_only_config_through_unchanged() {
        let messages = vec![LdMessage::system("You are a judge.")];
        assert_eq!(strip_legacy_judge_messages(&messages), messages);
    }
}

mod judge {
    use super::*;

    pub struct Fixture {
        pub client: Arc<FakeLdClient>,
        pub logger: Arc<RecordingLogger>,
        pub config: LdAiJudgeConfig,
    }

    impl Fixture {
        pub fn new() -> Self {
            let client = FakeLdClient::new();
            let ld_client = client.dyn_client();
            let config = LdAiJudgeConfig {
                key: "test-judge".to_string(),
                enabled: true,
                model: Some(LdModelConfig::named("gpt-4")),
                provider: Some(LdProviderConfig::named("openai")),
                messages: Some(vec![LdMessage::system(
                    "You are a helpful judge that evaluates AI responses.",
                )]),
                evaluation_metric_key: Some("relevance".to_string()),
                tracker_factory: TrackerFactory::new(move || {
                    LdAiConfigTracker::new(
                        Arc::clone(&ld_client),
                        test_context(),
                        LdAiConfigTrackerParams::new(
                            TEST_RUN_ID,
                            "test-config",
                            "test-variation",
                            1,
                        ),
                    )
                }),
            };
            Self {
                client,
                logger: Arc::new(RecordingLogger::default()),
                config,
            }
        }

        pub fn logger(&self) -> Option<Arc<dyn LdLogger>> {
            Some(Arc::clone(&self.logger) as Arc<dyn LdLogger>)
        }

        pub fn judge(&self, runner: &Arc<FakeRunner>, sample_rate: f64) -> Judge {
            self.judge_with(self.config.clone(), runner, sample_rate)
        }

        pub fn judge_with(
            &self,
            config: LdAiJudgeConfig,
            runner: &Arc<FakeRunner>,
            sample_rate: f64,
        ) -> Judge {
            Judge::new(
                config,
                Arc::clone(runner) as Arc<dyn Runner>,
                sample_rate,
                self.logger(),
            )
        }

        pub fn track_data_json(&self) -> String {
            serde_json::to_string(&self.config.create_tracker().get_track_data()).unwrap()
        }
    }

    pub fn tokens_100() -> Option<LdTokenUsage> {
        Some(LdTokenUsage {
            total: 100,
            input: 50,
            output: 50,
        })
    }

    pub fn runner_returning(parsed: Option<Value>, metrics: LdAiMetrics) -> Arc<FakeRunner> {
        FakeRunner::new(move |_, _| {
            Ok(RunnerResult {
                content: String::new(),
                metrics: metrics.clone(),
                raw: None,
                parsed: parsed.as_ref().and_then(Value::as_object).cloned(),
            })
        })
    }

    pub fn scoring_runner(score: f64, reasoning: &str) -> Arc<FakeRunner> {
        runner_returning(
            Some(json!({ "score": score, "reasoning": reasoning })),
            LdAiMetrics::new(true),
        )
    }

    pub fn unsampled() -> LdJudgeResult {
        LdJudgeResult {
            judge_config_key: Some("test-judge".to_string()),
            success: false,
            sampled: false,
            ..LdJudgeResult::default()
        }
    }

    pub fn sampled_failure() -> LdJudgeResult {
        LdJudgeResult {
            sampled: true,
            ..unsampled()
        }
    }

    mod constructor {
        use super::*;

        #[test]
        fn initializes_with_proper_configuration() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.5, "ok");
            let judge = fixture.judge(&runner, 1.0);
            assert_eq!(judge.get_ai_config().key, "test-judge");
            assert!(Arc::ptr_eq(
                judge.get_runner(),
                &(Arc::clone(&runner) as Arc<dyn Runner>)
            ));
        }

        #[test]
        fn defaults_sample_rate_to_1_0_when_omitted() {
            let fixture = Fixture::new();
            let judge = Judge::new(
                fixture.config.clone(),
                scoring_runner(0.5, "ok") as Arc<dyn Runner>,
                1.0,
                None,
            );
            assert_eq!(judge.sample_rate(), 1.0);
        }

        #[test]
        fn exposes_the_sample_rate_provided_to_the_constructor() {
            let fixture = Fixture::new();
            assert_eq!(
                fixture
                    .judge(&scoring_runner(0.5, "ok"), 0.25)
                    .sample_rate(),
                0.25
            );
        }

        #[test]
        fn honors_a_sample_rate_of_0() {
            let fixture = Fixture::new();
            assert_eq!(
                fixture.judge(&scoring_runner(0.5, "ok"), 0.0).sample_rate(),
                0.0
            );
        }
    }

    mod sampling_fallback_in_evaluate {
        use super::*;

        #[tokio::test]
        async fn uses_the_constructor_sample_rate_when_no_per_call_rate_is_supplied() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.5, "ok");
            let judge = fixture
                .judge(&runner, 0.5)
                .with_random_source(Arc::new(|| 0.6));

            let result = judge.evaluate("input", "output", None).await;

            assert!(!result.sampled);
            assert!(runner.inputs().is_empty());
        }

        #[tokio::test]
        async fn honors_an_explicit_per_call_sampling_rate_of_0_over_the_constructor_default() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.5, "ok");
            let judge = fixture
                .judge(&runner, 1.0)
                .with_random_source(Arc::new(|| 0.5));

            let result = judge.evaluate("input", "output", Some(0.0)).await;

            assert!(!result.sampled);
            assert!(runner.inputs().is_empty());
        }

        #[tokio::test]
        async fn per_call_sampling_rate_of_undefined_falls_through_to_the_constructor_default() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.5, "ok");
            let judge = fixture
                .judge(&runner, 0.0)
                .with_random_source(Arc::new(|| 0.5));

            let result = judge.evaluate("input", "output", None).await;

            assert!(!result.sampled);
            assert!(runner.inputs().is_empty());
        }
    }

    mod evaluate {
        use super::*;

        #[tokio::test]
        async fn evaluates_ai_response_successfully() {
            let fixture = Fixture::new();
            let runner = runner_returning(
                Some(json!({
                    "score": 0.8,
                    "reasoning": "The response is relevant to the question",
                })),
                LdAiMetrics {
                    tokens: tokens_100(),
                    ..LdAiMetrics::new(true)
                },
            );
            let judge = fixture.judge(&runner, 1.0);

            let result = judge
                .evaluate(
                    "What is the capital of France?",
                    "Paris is the capital of France.",
                    None,
                )
                .await;

            assert_eq!(
                result,
                LdJudgeResult {
                    score: Some(0.8),
                    reasoning: Some("The response is relevant to the question".to_string()),
                    metric_key: Some("relevance".to_string()),
                    success: true,
                    sampled: true,
                    judge_config_key: Some("test-judge".to_string()),
                    error_message: None,
                }
            );
            assert_eq!(
                *runner.calls.lock().unwrap(),
                vec![(
                    "MESSAGE HISTORY:\nWhat is the capital of France?\n\nRESPONSE TO EVALUATE:\nParis is the capital of France.".to_string(),
                    Some(evaluation_schema().clone()),
                )]
            );
            assert_eq!(
                fixture.client.track_keys(),
                vec![
                    "$ld:ai:duration:total",
                    "$ld:ai:generation:success",
                    "$ld:ai:tokens:total",
                    "$ld:ai:tokens:input",
                    "$ld:ai:tokens:output",
                ]
            );
        }

        #[tokio::test]
        async fn passes_a_string_input_to_the_runner_not_a_message_list() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.85, "Good answer.");
            let judge = fixture.judge(&runner, 1.0);

            judge
                .evaluate("What is AI?", "AI is artificial intelligence.", None)
                .await;

            let inputs = runner.inputs();
            assert_eq!(inputs.len(), 1);
            assert!(inputs[0].contains("MESSAGE HISTORY:\nWhat is AI?"));
            assert!(inputs[0].contains("RESPONSE TO EVALUATE:\nAI is artificial intelligence."));
        }

        #[tokio::test]
        async fn returns_evaluation_result_with_correct_evaluation_metric_key_for_tracker_integration(
        ) {
            let fixture = Fixture::new();
            let runner = runner_returning(
                Some(json!({ "score": 0.85, "reasoning": "Highly relevant response" })),
                LdAiMetrics {
                    tokens: tokens_100(),
                    ..LdAiMetrics::new(true)
                },
            );
            let judge = fixture.judge(&runner, 1.0);

            let result = judge.evaluate("test input", "test output", None).await;

            assert_eq!(result.score, Some(0.85));
            assert_eq!(result.metric_key.as_deref(), Some("relevance"));
            assert_eq!(result.judge_config_key.as_deref(), Some("test-judge"));
            assert!(result.success);
            assert!(result.sampled);
        }

        #[tokio::test]
        async fn handles_sampling_rate_correctly() {
            let fixture = Fixture::new();
            let runner = runner_returning(
                Some(json!({ "score": 0.8, "reasoning": "Good" })),
                LdAiMetrics {
                    tokens: tokens_100(),
                    ..LdAiMetrics::new(true)
                },
            );
            let judge = fixture
                .judge(&runner, 1.0)
                .with_random_source(Arc::new(|| 0.3));

            let result = judge.evaluate("test input", "test output", Some(0.5)).await;

            assert!(result.sampled);
            assert_eq!(runner.inputs().len(), 1);
        }

        #[tokio::test]
        async fn returns_unsampled_result_when_skipped_by_sampling() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.8, "Good");
            let judge = fixture
                .judge(&runner, 1.0)
                .with_random_source(Arc::new(|| 0.8));

            let result = judge.evaluate("test input", "test output", Some(0.5)).await;

            assert_eq!(result, unsampled());
            assert!(runner.inputs().is_empty());
            assert!(fixture
                .logger
                .debugs()
                .contains(&"Judge evaluation skipped due to sampling rate: 0.5".to_string()));
            assert!(fixture.client.tracks().is_empty());
        }

        #[tokio::test]
        async fn returns_error_result_when_evaluation_metric_key_is_missing() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.8, "Good");
            let judge = fixture.judge_with(
                LdAiJudgeConfig {
                    evaluation_metric_key: None,
                    ..fixture.config.clone()
                },
                &runner,
                1.0,
            );

            let result = judge.evaluate("test input", "test output", None).await;

            assert_eq!(
                result,
                LdJudgeResult {
                    error_message: Some(
                        "Judge configuration is missing required evaluation metric key".to_string()
                    ),
                    ..sampled_failure()
                }
            );
            assert_eq!(
                fixture.logger.warnings(),
                vec![format!(
                    "Judge configuration is missing required evaluation metric key {}",
                    fixture.track_data_json()
                )]
            );
            assert!(runner.inputs().is_empty());
        }

        #[tokio::test]
        async fn uses_evaluation_metric_key_when_provided() {
            let fixture = Fixture::new();
            let runner = runner_returning(
                Some(json!({ "score": 0.8, "reasoning": "The response is relevant" })),
                LdAiMetrics {
                    tokens: tokens_100(),
                    ..LdAiMetrics::new(true)
                },
            );
            let judge = fixture.judge_with(
                LdAiJudgeConfig {
                    evaluation_metric_key: Some("relevance".to_string()),
                    ..fixture.config.clone()
                },
                &runner,
                1.0,
            );

            let result = judge.evaluate("test input", "test output", None).await;

            assert_eq!(
                result,
                LdJudgeResult {
                    score: Some(0.8),
                    reasoning: Some("The response is relevant".to_string()),
                    metric_key: Some("relevance".to_string()),
                    success: true,
                    sampled: true,
                    judge_config_key: Some("test-judge".to_string()),
                    error_message: None,
                }
            );
        }

        #[tokio::test]
        async fn proceeds_does_not_error_early_when_messages_is_undefined() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.7, "Acceptable response.");
            let judge = fixture.judge_with(
                LdAiJudgeConfig {
                    messages: None,
                    ..fixture.config.clone()
                },
                &runner,
                1.0,
            );

            let result = judge.evaluate("test input", "test output", None).await;

            assert!(result.sampled);
            assert!(result.success);
            assert_eq!(runner.inputs().len(), 1);
        }

        #[tokio::test]
        async fn returns_result_with_success_false_when_parsed_is_undefined_or_has_no_score_reasoning(
        ) {
            let fixture = Fixture::new();
            let runner = runner_returning(
                None,
                LdAiMetrics {
                    tokens: tokens_100(),
                    ..LdAiMetrics::new(true)
                },
            );
            let judge = fixture.judge(&runner, 1.0);

            let result = judge.evaluate("test input", "test output", None).await;

            assert_eq!(result, sampled_failure());
            assert_eq!(
                fixture.logger.warnings(),
                vec![format!(
                    "Could not parse evaluation response: undefined {}",
                    fixture.track_data_json()
                )]
            );
        }

        #[tokio::test]
        async fn returns_result_with_success_false_when_parsed_is_an_empty_object() {
            let fixture = Fixture::new();
            let runner = runner_returning(
                Some(json!({})),
                LdAiMetrics {
                    tokens: tokens_100(),
                    ..LdAiMetrics::new(true)
                },
            );
            let judge = fixture.judge(&runner, 1.0);

            let result = judge.evaluate("test input", "test output", None).await;

            assert_eq!(result, sampled_failure());
            assert_eq!(
                fixture.logger.warnings(),
                vec![format!(
                    "Could not parse evaluation response: {{}} {}",
                    fixture.track_data_json()
                )]
            );
        }

        #[tokio::test]
        async fn returns_result_with_success_false_when_response_structure_is_malformed() {
            let fixture = Fixture::new();
            let runner = runner_returning(
                Some(json!({
                    "evaluations": { "relevance": { "score": 0.8, "reasoning": "Good" } },
                })),
                LdAiMetrics {
                    tokens: tokens_100(),
                    ..LdAiMetrics::new(true)
                },
            );
            let judge = fixture.judge(&runner, 1.0);

            let result = judge.evaluate("test input", "test output", None).await;

            assert_eq!(result, sampled_failure());
            let warnings = fixture.logger.warnings();
            assert_eq!(warnings.len(), 1);
            assert!(warnings[0].starts_with("Could not parse evaluation response:"));
            assert!(warnings[0].ends_with(&fixture.track_data_json()));
        }

        #[tokio::test]
        async fn handles_runner_errors_gracefully() {
            let fixture = Fixture::new();
            let runner = FakeRunner::failing("Provider error");
            let judge = fixture.judge(&runner, 1.0);

            let result = judge.evaluate("test input", "test output", None).await;

            assert_eq!(
                result,
                LdJudgeResult {
                    error_message: Some("Provider error".to_string()),
                    ..sampled_failure()
                }
            );
            assert_eq!(
                fixture.logger.errors(),
                vec!["Judge evaluation failed: Provider error".to_string()]
            );
            assert_eq!(
                fixture.client.track_keys(),
                vec!["$ld:ai:duration:total", "$ld:ai:generation:error"]
            );
        }
    }

    mod evaluate_messages {
        use super::*;

        fn response(content: &str) -> RunnerResult {
            RunnerResult {
                content: content.to_string(),
                metrics: LdAiMetrics::new(true),
                ..RunnerResult::default()
            }
        }

        #[tokio::test]
        async fn evaluates_messages_and_response_successfully() {
            let fixture = Fixture::new();
            let runner = runner_returning(
                Some(json!({
                    "score": 0.8,
                    "reasoning": "The response is relevant to the question",
                })),
                LdAiMetrics {
                    tokens: tokens_100(),
                    ..LdAiMetrics::new(true)
                },
            );
            let judge = fixture.judge(&runner, 1.0);
            let messages = vec![
                LdMessage::user("What is the capital of France?"),
                LdMessage::assistant("Paris is the capital of France."),
            ];

            let result = judge
                .evaluate_messages(
                    &messages,
                    &response("Paris is the capital of France."),
                    None,
                )
                .await;

            assert_eq!(
                result,
                LdJudgeResult {
                    score: Some(0.8),
                    reasoning: Some("The response is relevant to the question".to_string()),
                    metric_key: Some("relevance".to_string()),
                    success: true,
                    sampled: true,
                    judge_config_key: Some("test-judge".to_string()),
                    error_message: None,
                }
            );
            assert_eq!(
                *runner.calls.lock().unwrap(),
                vec![(
                    "MESSAGE HISTORY:\nuser: What is the capital of France?\nassistant: Paris is the capital of France.\n\nRESPONSE TO EVALUATE:\nParis is the capital of France.".to_string(),
                    Some(evaluation_schema().clone()),
                )]
            );
        }

        #[tokio::test]
        async fn renders_each_message_as_role_content_joined_by_newlines() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.5, "ok");
            let judge = fixture.judge(&runner, 1.0);
            let messages = vec![LdMessage::user("hi"), LdMessage::assistant("hello")];

            judge
                .evaluate_messages(&messages, &response("hello"), None)
                .await;

            assert!(runner.inputs()[0].contains("MESSAGE HISTORY:\nuser: hi\nassistant: hello"));
        }

        #[tokio::test]
        async fn produces_an_empty_message_history_section_when_messages_is_empty() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.5, "ok");
            let judge = fixture.judge(&runner, 1.0);

            judge
                .evaluate_messages(&[], &response("output"), None)
                .await;

            assert_eq!(
                runner.inputs()[0],
                "MESSAGE HISTORY:\n\n\nRESPONSE TO EVALUATE:\noutput"
            );
        }

        #[tokio::test]
        async fn does_not_contaminate_the_runner_input_across_successive_evaluations() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.5, "ok");
            let judge = fixture.judge(&runner, 1.0);

            judge.evaluate("Q1", "A1", None).await;
            judge.evaluate("Q2", "A2", None).await;

            let inputs = runner.inputs();
            assert_eq!(
                inputs[0],
                "MESSAGE HISTORY:\nQ1\n\nRESPONSE TO EVALUATE:\nA1"
            );
            assert_eq!(
                inputs[1],
                "MESSAGE HISTORY:\nQ2\n\nRESPONSE TO EVALUATE:\nA2"
            );
            assert!(!inputs[1].contains("Q1"));
            assert!(!inputs[1].contains("A1"));
        }

        #[tokio::test]
        async fn handles_sampling_rate_correctly() {
            let fixture = Fixture::new();
            let runner = scoring_runner(0.5, "ok");
            let judge = fixture
                .judge(&runner, 1.0)
                .with_random_source(Arc::new(|| 0.8));

            let result = judge
                .evaluate_messages(
                    &[LdMessage::user("test")],
                    &response("test response"),
                    Some(0.5),
                )
                .await;

            assert_eq!(result, unsampled());
            assert!(runner.inputs().is_empty());
        }
    }

    mod build_evaluation_input {
        use super::*;

        #[test]
        fn builds_the_evaluation_string_in_the_expected_format() {
            assert_eq!(
                build_evaluation_input("hello", "world"),
                "MESSAGE HISTORY:\nhello\n\nRESPONSE TO EVALUATE:\nworld"
            );
        }
    }

    mod parse_evaluation_response {
        use super::*;

        fn parse(data: Value) -> Option<(f64, String)> {
            parse_evaluation_response(data.as_object())
        }

        #[test]
        fn parses_valid_evaluation_response_correctly() {
            assert_eq!(
                parse(json!({ "score": 0.8, "reasoning": "Good" })),
                Some((0.8, "Good".to_string()))
            );
        }

        #[test]
        fn returns_undefined_for_empty_response_data() {
            assert_eq!(parse(json!({})), None);
        }

        #[test]
        fn handles_missing_reasoning_field() {
            assert_eq!(parse(json!({ "score": 0.8 })), None);
        }

        #[test]
        fn handles_invalid_score_values_out_of_range() {
            assert_eq!(parse(json!({ "score": 1.5, "reasoning": "Good" })), None);
        }

        #[test]
        fn handles_negative_score_values() {
            assert_eq!(parse(json!({ "score": -0.1, "reasoning": "Good" })), None);
        }

        #[test]
        fn handles_invalid_reasoning_type() {
            assert_eq!(parse(json!({ "score": 0.8, "reasoning": 123 })), None);
        }
    }
}
