//! Rust version of the JavaScript SDK's `examples/features/create-judge`, judging with
//! a self-hosted OpenAI-compatible model.
//!
//! ```sh
//! LAUNCHDARKLY_SDK_KEY=sdk-... LAUNCHDARKLY_JUDGE_KEY=sample-judge \
//! OPENAI_BASE_URL=http://localhost:8000/v1 \
//!   cargo run --example create_judge --features openai-compatible
//! ```

use std::sync::Arc;
use std::time::Duration;

use launchdarkly_server_sdk::{Client, ConfigBuilder};
use unofficial_launchdarkly_server_sdk_ai::openai_compatible::OpenAiCompatibleProvider;
use unofficial_launchdarkly_server_sdk_ai::{ContextBuilder, LdAiClient, RunnerFactory};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdk_key = std::env::var("LAUNCHDARKLY_SDK_KEY")
        .map_err(|_| "*** Please set the LAUNCHDARKLY_SDK_KEY env first")?;
    let judge_key =
        std::env::var("LAUNCHDARKLY_JUDGE_KEY").unwrap_or_else(|_| "sample-judge".to_string());
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let ld_client = Arc::new(Client::build(ConfigBuilder::new(&sdk_key).build()?)?);
    ld_client.start_with_default_executor();
    if ld_client
        .wait_for_initialization(Duration::from_secs(10))
        .await
        != Some(true)
    {
        return Err("*** SDK failed to initialize".into());
    }
    println!("*** SDK successfully initialized");

    let provider =
        OpenAiCompatibleProvider::new(base_url).with_api_key(std::env::var("OPENAI_API_KEY").ok());
    let factory = RunnerFactory::new().with_provider("openai", Arc::new(provider));
    let ai_client = LdAiClient::with_runner_factory(ld_client.clone(), factory);

    let context = ContextBuilder::new("example-user-key")
        .name("Sandy")
        .build()?;

    let Some(judge) = ai_client
        .create_judge(&judge_key, &context, None, None, None, 1.0)
        .await
    else {
        println!("AI config '{judge_key}' is disabled or its provider is not registered.");
        return Ok(());
    };

    let input = "You are a helpful assistant for the company LaunchDarkly. How can you help me?";
    let output =
        "I can answer any question you have except for questions about the company LaunchDarkly.";
    let result = judge.evaluate(input, output, None).await;

    println!("\nJudge result:");
    println!("- judge_config_key: {judge_key}");
    println!("  sampled: {}", result.sampled);
    if result.sampled {
        println!("  success: {}", result.success);
        println!("  error_message: {:?}", result.error_message);
        println!("  metric_key: {:?}", result.metric_key);
        println!("  score: {:?}", result.score);
        println!("  reasoning: {:?}", result.reasoning);
    }

    ld_client.flush();
    ld_client.close();
    Ok(())
}
