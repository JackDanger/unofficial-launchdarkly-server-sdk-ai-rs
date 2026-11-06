mod common;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use common::{test_context, FakeLdClient, FakeProvider, FakeRunner, RecordingLogger, TEST_RUN_ID};
use indexmap::IndexMap;
use serde_json::{json, Value};
use unofficial_launchdarkly_server_sdk_ai::{
    AgentGraphDefinition, AgentGraphRunner, AgentGraphRunnerResult, AiProvider, BoxError,
    Evaluator, LdAgentGraphFlagValue, LdAiAgentConfig, LdAiCompletionConfig, LdAiConfigTracker,
    LdAiConfigTrackerParams, LdAiGraphMetrics, LdAiMetrics, LdGraphTracker, LdLogger,
    LdProviderConfig, ModelConfigRef, Runner, RunnerFactory, RunnerResult, ToolFn, ToolRegistry,
    TrackerFactory,
};

fn tracker_factory(key: &str) -> TrackerFactory {
    let ld_client = FakeLdClient::new().dyn_client();
    let key = key.to_string();
    TrackerFactory::new(move || {
        LdAiConfigTracker::new(
            Arc::clone(&ld_client),
            test_context(),
            LdAiConfigTrackerParams::new(TEST_RUN_ID, key.clone(), "v1", 1),
        )
    })
}

fn make_config(provider_name: &str) -> LdAiCompletionConfig {
    LdAiCompletionConfig {
        key: "test-config".to_string(),
        enabled: true,
        model: None,
        provider: Some(LdProviderConfig::named(provider_name)),
        messages: None,
        judge_configuration: None,
        tools: None,
        tracker_factory: tracker_factory("test-config"),
        evaluator: Evaluator::noop(),
    }
}

fn make_agent_config(provider_name: &str) -> LdAiAgentConfig {
    LdAiAgentConfig {
        key: "test-agent-config".to_string(),
        enabled: true,
        model: None,
        provider: Some(LdProviderConfig::named(provider_name)),
        instructions: None,
        judge_configuration: None,
        tools: None,
        tracker_factory: tracker_factory("test-agent-config"),
        evaluator: Evaluator::noop(),
    }
}

fn make_graph_def() -> AgentGraphDefinition {
    let ld_client = FakeLdClient::new().dyn_client();
    AgentGraphDefinition::new(
        LdAgentGraphFlagValue::default(),
        IndexMap::new(),
        true,
        move || {
            LdGraphTracker::new(
                Arc::clone(&ld_client),
                TEST_RUN_ID,
                "test-graph",
                None,
                1,
                test_context(),
            )
        },
    )
}

fn make_runner() -> Arc<dyn Runner> {
    FakeRunner::returning("", LdAiMetrics::new(true))
}

fn make_tools(name: &str) -> ToolRegistry {
    let tool: ToolFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    ToolRegistry::from([(name.to_string(), tool)])
}

fn same_runner(a: &Option<Arc<dyn Runner>>, b: &Arc<dyn Runner>) -> bool {
    a.as_ref().is_some_and(|a| Arc::ptr_eq(a, b))
}

fn attempts(logger: &RecordingLogger) -> Vec<String> {
    logger
        .debugs()
        .into_iter()
        .filter_map(|line| {
            line.strip_prefix("Attempting to create runner with provider: ")
                .map(str::to_string)
        })
        .collect()
}

fn as_logger(logger: &Arc<RecordingLogger>) -> Option<&dyn LdLogger> {
    Some(logger.as_ref() as &dyn LdLogger)
}

struct GraphRunner;

#[async_trait]
impl AgentGraphRunner for GraphRunner {
    async fn run(&self, _input: &str) -> Result<AgentGraphRunnerResult, BoxError> {
        Ok(AgentGraphRunnerResult::default())
    }
}

mod runner_factory_create_model {
    use super::*;

    #[tokio::test]
    async fn returns_undefined_and_logs_a_warning_when_no_provider_package_is_installed() {
        let logger = Arc::new(RecordingLogger::default());
        let config = make_config("openai");

        let result = RunnerFactory::new()
            .create_model(
                ModelConfigRef::Completion(&config),
                as_logger(&logger),
                None,
                true,
            )
            .await;

        assert!(result.is_none());
        assert!(logger
            .warnings()
            .iter()
            .any(|w| w.contains("not supported")));
    }

    #[tokio::test]
    async fn returns_a_runner_from_the_first_provider_that_succeeds() {
        let runner = make_runner();
        let provider = FakeProvider::with_model(Arc::clone(&runner));
        let factory = RunnerFactory::new().with_provider("openai", provider.clone());
        let config = make_config("openai");

        let result = factory
            .create_model(ModelConfigRef::Completion(&config), None, None, true)
            .await;

        assert!(same_runner(&result, &runner));
        assert_eq!(
            *provider.model_calls.lock().unwrap(),
            vec![("test-config".to_string(), true)]
        );
    }

    #[tokio::test]
    async fn uses_only_the_default_ai_provider_when_one_is_specified() {
        let openai = FakeProvider::with_model(make_runner());
        let langchain = FakeProvider::with_model(make_runner());
        let factory = RunnerFactory::new()
            .with_provider("openai", openai.clone())
            .with_multi_provider("langchain", langchain.clone());
        let logger = Arc::new(RecordingLogger::default());
        let config = make_config("langchain");

        factory
            .create_model(
                ModelConfigRef::Completion(&config),
                as_logger(&logger),
                Some("openai"),
                true,
            )
            .await;

        assert_eq!(
            factory.providers_to_try(Some("openai"), Some("langchain")),
            vec!["openai"]
        );
        assert_eq!(attempts(&logger), vec!["openai"]);
        assert_eq!(openai.model_calls.lock().unwrap().len(), 1);
        assert!(langchain.model_calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn defaults_multi_turn_to_true_when_forwarding_to_the_provider_factory() {
        let provider = FakeProvider::with_model(make_runner());
        let factory = RunnerFactory::new().with_provider("openai", provider.clone());
        let config = make_config("openai");

        factory
            .create_model(ModelConfigRef::Completion(&config), None, None, true)
            .await;

        assert_eq!(
            *provider.model_calls.lock().unwrap(),
            vec![("test-config".to_string(), true)]
        );
    }

    #[tokio::test]
    async fn forwards_multi_turn_false_to_the_provider_factory_when_specified() {
        let provider = FakeProvider::with_model(make_runner());
        let factory = RunnerFactory::new().with_provider("openai", provider.clone());
        let config = make_config("openai");

        factory
            .create_model(ModelConfigRef::Completion(&config), None, None, false)
            .await;

        assert_eq!(
            *provider.model_calls.lock().unwrap(),
            vec![("test-config".to_string(), false)]
        );
    }

    #[tokio::test]
    async fn falls_through_to_multi_provider_packages_when_specific_provider_returns_undefined() {
        let runner = make_runner();
        let openai = FakeProvider::empty();
        let langchain = FakeProvider::with_model(Arc::clone(&runner));
        let vercel = FakeProvider::with_model(make_runner());
        let factory = RunnerFactory::new()
            .with_provider("openai", openai.clone())
            .with_multi_provider("langchain", langchain.clone())
            .with_multi_provider("vercel", vercel.clone());
        let logger = Arc::new(RecordingLogger::default());
        let config = make_config("openai");

        let result = factory
            .create_model(
                ModelConfigRef::Completion(&config),
                as_logger(&logger),
                None,
                true,
            )
            .await;

        assert!(same_runner(&result, &runner));
        assert_eq!(attempts(&logger), vec!["openai", "langchain"]);
        assert_eq!(openai.model_calls.lock().unwrap().len(), 1);
        assert_eq!(langchain.model_calls.lock().unwrap().len(), 1);
        assert!(vercel.model_calls.lock().unwrap().is_empty());
    }
}

mod runner_factory_with_fallback {
    use super::*;

    #[tokio::test]
    async fn returns_the_first_truthy_result_and_does_not_call_remaining_factories() {
        let runner = make_runner();
        let factory_a = FakeProvider::with_model(Arc::clone(&runner));
        let factory_b = FakeProvider::with_model(make_runner());
        let factory = RunnerFactory::new()
            .with_provider("openai", factory_a)
            .with_multi_provider("langchain", factory_b.clone());
        let config = make_config("openai");

        let result = factory
            .create_model(ModelConfigRef::Completion(&config), None, None, true)
            .await;

        assert!(same_runner(&result, &runner));
        assert!(factory_b.model_calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn returns_undefined_when_all_factories_return_undefined() {
        let factory = RunnerFactory::new()
            .with_provider("openai", FakeProvider::empty())
            .with_multi_provider("langchain", FakeProvider::empty())
            .with_multi_provider("vercel", FakeProvider::empty());
        let config = make_config("openai");

        let result = factory
            .create_model(ModelConfigRef::Completion(&config), None, None, true)
            .await;

        assert!(result.is_none());
    }
}

mod runner_factory_create_agent {
    use super::*;

    struct ToolRecordingProvider {
        runner: Arc<dyn Runner>,
        calls: Mutex<Vec<(String, Option<Vec<String>>)>>,
    }

    #[async_trait]
    impl AiProvider for ToolRecordingProvider {
        async fn create_agent(
            &self,
            config: &LdAiAgentConfig,
            tools: Option<&ToolRegistry>,
        ) -> Option<Arc<dyn Runner>> {
            self.calls.lock().unwrap().push((
                config.key.clone(),
                tools.map(|t| t.keys().cloned().collect()),
            ));
            Some(Arc::clone(&self.runner))
        }
    }

    #[tokio::test]
    async fn delegates_to_factory_create_agent_with_config_and_tools() {
        let runner = make_runner();
        let provider = Arc::new(ToolRecordingProvider {
            runner: Arc::clone(&runner),
            calls: Mutex::new(Vec::new()),
        });
        let factory = RunnerFactory::new().with_provider("openai", provider.clone());
        let tools = make_tools("myTool");

        let result = factory
            .create_agent(&make_agent_config("openai"), Some(&tools), None, None)
            .await;

        assert!(same_runner(&result, &runner));
        assert_eq!(
            *provider.calls.lock().unwrap(),
            vec![(
                "test-agent-config".to_string(),
                Some(vec!["myTool".to_string()])
            )]
        );
    }

    #[tokio::test]
    async fn returns_undefined_and_warns_when_no_provider_supports_create_agent() {
        let logger = Arc::new(RecordingLogger::default());

        let result = RunnerFactory::new()
            .create_agent(&make_agent_config("openai"), None, as_logger(&logger), None)
            .await;

        assert!(result.is_none());
        assert!(logger
            .warnings()
            .iter()
            .any(|w| w.contains("not supported")));
    }
}

mod runner_factory_create_agent_graph {
    use super::*;

    #[tokio::test]
    async fn delegates_to_factory_create_agent_graph_with_graph_def_and_tools() {
        let graph_runner: Arc<dyn AgentGraphRunner> = Arc::new(GraphRunner);
        let provider = Arc::new(FakeProvider {
            graph_runner: Some(Arc::clone(&graph_runner)),
            ..FakeProvider::default()
        });
        let factory = RunnerFactory::new().with_multi_provider("langchain", provider.clone());
        let tools = make_tools("search");

        let result = factory
            .create_agent_graph(&make_graph_def(), Some(&tools), None, None)
            .await;

        assert!(result.is_some_and(|r| Arc::ptr_eq(&r, &graph_runner)));
        assert_eq!(*provider.graph_calls.lock().unwrap(), vec![true]);
    }

    #[tokio::test]
    async fn returns_undefined_and_warns_when_no_provider_supports_create_agent_graph() {
        let logger = Arc::new(RecordingLogger::default());

        let result = RunnerFactory::new()
            .create_agent_graph(&make_graph_def(), None, as_logger(&logger), None)
            .await;

        assert!(result.is_none());
        assert!(logger
            .warnings()
            .iter()
            .any(|w| w.contains("AgentGraphRunner")));
    }
}

mod ai_provider_default_factory_methods {
    use super::*;

    struct ConcreteProvider;

    impl AiProvider for ConcreteProvider {}

    #[tokio::test]
    async fn create_model_returns_undefined_by_default() {
        let config = make_config("openai");
        let result = ConcreteProvider
            .create_model(ModelConfigRef::Completion(&config), true)
            .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn create_agent_returns_undefined_by_default() {
        let result = ConcreteProvider
            .create_agent(&make_agent_config("openai"), None)
            .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn create_agent_graph_returns_undefined_by_default() {
        let result = ConcreteProvider
            .create_agent_graph(&make_graph_def(), None)
            .await;
        assert!(result.is_none());
    }
}

mod runner_protocol {
    use super::*;

    struct PlainRunner {
        result: RunnerResult,
        calls: Mutex<Vec<(String, Option<Value>)>>,
    }

    #[async_trait]
    impl Runner for PlainRunner {
        async fn run(
            &self,
            input: &str,
            output_type: Option<&Value>,
        ) -> Result<RunnerResult, BoxError> {
            self.calls
                .lock()
                .unwrap()
                .push((input.to_string(), output_type.cloned()));
            Ok(self.result.clone())
        }
    }

    struct PlainGraphRunner {
        result: AgentGraphRunnerResult,
    }

    #[async_trait]
    impl AgentGraphRunner for PlainGraphRunner {
        async fn run(&self, _input: &str) -> Result<AgentGraphRunnerResult, BoxError> {
            Ok(self.result.clone())
        }
    }

    #[tokio::test]
    async fn can_be_implemented_as_a_plain_object_no_class_extension_required() {
        let my_runner = PlainRunner {
            result: RunnerResult {
                content: "Hello from runner".to_string(),
                metrics: LdAiMetrics::new(true),
                ..RunnerResult::default()
            },
            calls: Mutex::new(Vec::new()),
        };

        let result = my_runner.run("Hello", None).await.unwrap();

        assert_eq!(result.content, "Hello from runner");
        assert!(result.metrics.success);
    }

    #[tokio::test]
    async fn runner_run_accepts_optional_output_type_for_structured_output() {
        let my_runner = PlainRunner {
            result: RunnerResult {
                content: String::new(),
                metrics: LdAiMetrics::new(true),
                parsed: json!({ "score": 0.9, "reasoning": "good" })
                    .as_object()
                    .cloned(),
                raw: None,
            },
            calls: Mutex::new(Vec::new()),
        };
        let schema = json!({ "type": "object", "properties": { "score": { "type": "number" } } });

        let result = my_runner.run("Evaluate", Some(&schema)).await.unwrap();

        assert_eq!(
            result.parsed,
            json!({ "score": 0.9, "reasoning": "good" })
                .as_object()
                .cloned()
        );
        assert_eq!(
            *my_runner.calls.lock().unwrap(),
            vec![("Evaluate".to_string(), Some(schema))]
        );
    }

    #[tokio::test]
    async fn agent_graph_runner_can_be_implemented_as_a_plain_object() {
        let my_graph_runner = PlainGraphRunner {
            result: AgentGraphRunnerResult {
                content: "Graph output".to_string(),
                metrics: LdAiGraphMetrics {
                    success: true,
                    path: vec!["node-a".to_string()],
                    node_metrics: IndexMap::from([("node-a".to_string(), LdAiMetrics::new(true))]),
                    ..LdAiGraphMetrics::default()
                },
                raw: None,
            },
        };

        let result = my_graph_runner.run("user input").await.unwrap();

        assert_eq!(result.content, "Graph output");
        assert_eq!(result.metrics.path, vec!["node-a"]);
    }

    #[test]
    fn runner_result_does_not_include_evaluations_field() {
        let RunnerResult {
            content,
            metrics,
            raw,
            parsed,
        } = RunnerResult {
            content: "test".to_string(),
            metrics: LdAiMetrics::new(true),
            ..RunnerResult::default()
        };
        assert_eq!(content, "test");
        assert!(metrics.success);
        assert!(raw.is_none() && parsed.is_none());
    }
}
