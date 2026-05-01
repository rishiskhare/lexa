use std::path::PathBuf;
use std::process::Command;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/sample")
}

#[test]
fn cli_indexes_searches_and_purges_fixture() {
    let temp = tempfile::tempdir().unwrap();
    let db = temp.path().join("lexa.sqlite");
    let fixture = fixture_path();

    let index = Command::new(env!("CARGO_BIN_EXE_lexa"))
        .args(["--hash-embeddings", "--db"])
        .arg(&db)
        .arg("index")
        .arg(&fixture)
        .output()
        .unwrap();
    assert!(
        index.status.success(),
        "{}",
        String::from_utf8_lossy(&index.stderr)
    );
    assert!(String::from_utf8_lossy(&index.stdout).contains("indexed 4 file(s)"));

    for tier in ["instant", "fast", "deep"] {
        let search = Command::new(env!("CARGO_BIN_EXE_lexa"))
            .args(["--hash-embeddings", "--db"])
            .arg(&db)
            .args([
                "search",
                "config validation function",
                "--tier",
                tier,
                "--json",
            ])
            .output()
            .unwrap();
        assert!(
            search.status.success(),
            "{}",
            String::from_utf8_lossy(&search.stderr)
        );
        let json: serde_json::Value = serde_json::from_slice(&search.stdout).unwrap();
        let first = json.as_array().unwrap().first().unwrap();
        assert!(first["path"].as_str().unwrap().ends_with("src/config.rs"));
        assert!(first["line_start"].as_i64().unwrap() > 0);
        assert!(first["breakdown"].is_object());
    }

    let purge = Command::new(env!("CARGO_BIN_EXE_lexa"))
        .args(["--hash-embeddings", "--db"])
        .arg(&db)
        .arg("purge")
        .arg(&fixture)
        .output()
        .unwrap();
    assert!(
        purge.status.success(),
        "{}",
        String::from_utf8_lossy(&purge.stderr)
    );
    assert!(String::from_utf8_lossy(&purge.stdout).contains("purged 4 file(s)"));
}
