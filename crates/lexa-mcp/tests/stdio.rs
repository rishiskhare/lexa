use std::path::PathBuf;

use rmcp::{
    model::CallToolRequestParams,
    transport::{ConfigureCommandExt, TokioChildProcess},
    ServiceExt,
};
use serde_json::{json, Map, Value};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/sample")
}

#[tokio::test]
async fn mcp_lists_tools_and_searches_files() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let db_path = temp.path().join("lexa.sqlite");
    let fixture = fixture_path();

    let mut db = lexa_core::open(
        &db_path,
        lexa_core::EmbeddingConfig {
            backend: lexa_core::EmbeddingBackend::Hash,
            show_download_progress: false,
        },
    )?;
    db.index_path(&fixture)?;

    let transport = TokioChildProcess::new(
        tokio::process::Command::new(env!("CARGO_BIN_EXE_lexa-mcp")).configure(|cmd| {
            cmd.env("LEXA_DB", &db_path);
            cmd.env("LEXA_EMBEDDER", "hash");
        }),
    )?;

    let client = ().serve(transport).await?;
    let tools = client.list_all_tools().await?;
    assert!(tools.iter().any(|tool| tool.name == "search_files"));

    let mut args = Map::new();
    args.insert("query".to_string(), json!("config validation function"));
    args.insert("tier".to_string(), json!("fast"));
    args.insert("limit".to_string(), json!(3));
    let result = client
        .call_tool(CallToolRequestParams::new("search_files").with_arguments(args))
        .await?;
    let text = result
        .content
        .first()
        .and_then(|content| content.as_text())
        .map(|text| text.text.as_str())
        .unwrap_or_default();
    let hits: Value = serde_json::from_str(text)?;
    assert!(hits[0]["path"].as_str().unwrap().ends_with("src/config.rs"));

    client.cancel().await?;
    Ok(())
}
