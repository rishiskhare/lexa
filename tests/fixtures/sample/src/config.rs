pub struct Config {
    pub project: String,
    pub max_results: usize,
}

/// Config validation function used before search starts.
pub fn validate_config(config: &Config) -> Result<(), String> {
    if config.project.trim().is_empty() {
        return Err("project name is required".to_string());
    }
    if config.max_results == 0 {
        return Err("max_results must be positive".to_string());
    }
    Ok(())
}
