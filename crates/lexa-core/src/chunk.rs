use std::path::Path;

use tree_sitter::{Language, Node, Parser};

#[derive(Debug, Clone)]
pub struct RawChunk {
    pub byte_start: usize,
    pub byte_end: usize,
    pub line_start: usize,
    pub line_end: usize,
    pub kind: &'static str,
    pub text: String,
    pub context: Option<String>,
}

pub fn supported_kind(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "go" | "java" | "c" | "cc" | "cpp" | "h"
        | "hpp" => Some("code"),
        "md" | "mdx" => Some("markdown"),
        "txt" | "log" | "toml" | "yaml" | "yml" | "json" | "csv" => Some("text"),
        "pdf" => Some("pdf"),
        _ => None,
    }
}

#[cfg(test)]
pub fn chunk_text(text: &str, kind: &'static str) -> Vec<RawChunk> {
    chunk_text_for_path(text, kind, None)
}

pub fn chunk_text_for_path(text: &str, kind: &'static str, path: Option<&Path>) -> Vec<RawChunk> {
    match kind {
        "code" => chunk_code(text, path),
        "markdown" => chunk_markdown(text),
        _ => chunk_plain(text),
    }
}

fn chunk_markdown(text: &str) -> Vec<RawChunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut starts = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            starts.push(idx);
        }
    }
    if starts.is_empty() {
        return chunk_plain(text);
    }
    if starts.first().copied() != Some(0) {
        starts.insert(0, 0);
    }
    starts.push(lines.len());

    let mut out = Vec::new();
    for pair in starts.windows(2) {
        let context = nearest_heading(&lines, pair[0]);
        push_window(
            text,
            &lines,
            pair[0],
            pair[1],
            WindowSpec::new(90, "markdown", context),
            &mut out,
        );
    }
    out
}

fn chunk_code(text: &str, path: Option<&Path>) -> Vec<RawChunk> {
    if let Some(chunks) = chunk_code_with_tree_sitter(text, path) {
        if !chunks.is_empty() {
            return chunks;
        }
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut starts = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if is_code_boundary(line.trim_start()) {
            starts.push(idx);
        }
    }
    if starts.is_empty() {
        return chunk_plain(text)
            .into_iter()
            .map(|mut chunk| {
                chunk.kind = "code";
                chunk
            })
            .collect();
    }
    if starts.first().copied() != Some(0) {
        starts.insert(0, 0);
    }
    starts.push(lines.len());

    let mut out = Vec::new();
    for pair in starts.windows(2) {
        let context = lines.get(pair[0]).map(|line| line.trim().to_string());
        push_window(
            text,
            &lines,
            pair[0],
            pair[1],
            WindowSpec::new(120, "code", context),
            &mut out,
        );
    }
    out
}

fn chunk_code_with_tree_sitter(text: &str, path: Option<&Path>) -> Option<Vec<RawChunk>> {
    let mut parser = Parser::new();
    let language = language_for_path(path).or_else(|| language_for_source(text))?;
    parser.set_language(&language).ok()?;
    let tree = parser.parse(text, None)?;
    let mut ranges = Vec::new();
    collect_boundary_nodes(tree.root_node(), &mut ranges);
    if ranges.is_empty() {
        return None;
    }
    ranges.sort_by_key(|node| node.start_byte());
    ranges.dedup_by_key(|node| (node.start_byte(), node.end_byte()));
    Some(
        ranges
            .into_iter()
            .filter_map(|node| raw_chunk_from_node(text, node))
            .collect(),
    )
}

fn language_for_path(path: Option<&Path>) -> Option<Language> {
    let ext = path?.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "c" | "h" => Some(tree_sitter_c::LANGUAGE.into()),
        "cc" | "cpp" | "hpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        _ => None,
    }
}

fn language_for_source(text: &str) -> Option<Language> {
    if text.contains("pub fn ") || text.contains("fn ") || text.contains("impl ") {
        Some(tree_sitter_rust::LANGUAGE.into())
    } else if text.contains("def ") || text.contains("class ") {
        Some(tree_sitter_python::LANGUAGE.into())
    } else if text.contains("package ") && text.contains("func ") {
        Some(tree_sitter_go::LANGUAGE.into())
    } else if text.contains("public class ") || text.contains("class ") && text.contains(';') {
        Some(tree_sitter_java::LANGUAGE.into())
    } else if text.contains("#include") {
        Some(tree_sitter_c::LANGUAGE.into())
    } else {
        Some(tree_sitter_javascript::LANGUAGE.into())
    }
}

fn collect_boundary_nodes<'a>(node: Node<'a>, out: &mut Vec<Node<'a>>) {
    if is_tree_boundary(node.kind()) {
        out.push(node);
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_boundary_nodes(child, out);
    }
}

fn is_tree_boundary(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "function_definition"
            | "function_declaration"
            | "method_definition"
            | "method_declaration"
            | "class_definition"
            | "class_declaration"
            | "class_specifier"
            | "struct_item"
            | "struct_specifier"
            | "enum_item"
            | "impl_item"
            | "trait_item"
            | "interface_declaration"
            | "constructor_declaration"
            | "type_declaration"
    )
}

fn raw_chunk_from_node(text: &str, node: Node<'_>) -> Option<RawChunk> {
    let byte_start = extend_start_to_leading_comments(text, node.start_byte());
    let byte_end = node.end_byte();
    let body = text.get(byte_start..byte_end)?.trim().to_string();
    if body.is_empty() {
        return None;
    }
    Some(RawChunk {
        byte_start,
        byte_end,
        line_start: line_number_for_byte(text, byte_start),
        line_end: node.end_position().row + 1,
        kind: "code",
        context: body.lines().next().map(|line| line.trim().to_string()),
        text: body,
    })
}

fn extend_start_to_leading_comments(text: &str, start_byte: usize) -> usize {
    let mut current = line_start_for_byte(text, start_byte);
    let mut best = current;
    let mut saw_comment = false;

    while current > 0 {
        let prev_end = current.saturating_sub(1);
        let prev_start = line_start_for_byte(text, prev_end);
        let line = &text[prev_start..prev_end];
        let trimmed = line.trim_start();
        if is_leading_comment(trimmed) {
            saw_comment = true;
            best = prev_start;
            current = prev_start;
        } else if saw_comment && trimmed.trim().is_empty() {
            best = prev_start;
            current = prev_start;
        } else {
            break;
        }
    }

    best
}

fn is_leading_comment(line: &str) -> bool {
    line.starts_with("//")
        || line.starts_with("///")
        || line.starts_with("/*")
        || line.starts_with('*')
        || line.starts_with('#')
}

fn line_start_for_byte(text: &str, byte: usize) -> usize {
    text[..byte.min(text.len())]
        .rfind('\n')
        .map(|idx| idx + 1)
        .unwrap_or(0)
}

fn line_number_for_byte(text: &str, byte: usize) -> usize {
    text[..byte.min(text.len())]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1
}

fn chunk_plain(text: &str) -> Vec<RawChunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    push_window(
        text,
        &lines,
        0,
        lines.len(),
        WindowSpec::new(80, "text", None),
        &mut out,
    );
    out
}

struct WindowSpec {
    max_lines: usize,
    kind: &'static str,
    context: Option<String>,
}

impl WindowSpec {
    fn new(max_lines: usize, kind: &'static str, context: Option<String>) -> Self {
        Self {
            max_lines,
            kind,
            context,
        }
    }
}

fn push_window(
    text: &str,
    lines: &[&str],
    start: usize,
    end: usize,
    spec: WindowSpec,
    out: &mut Vec<RawChunk>,
) {
    let mut cursor = start;
    while cursor < end {
        let next = usize::min(cursor + spec.max_lines, end);
        let body = lines[cursor..next].join("\n").trim().to_string();
        if !body.is_empty() {
            out.push(RawChunk {
                byte_start: byte_offset_for_line(text, cursor),
                byte_end: byte_offset_for_line(text, next),
                line_start: cursor + 1,
                line_end: next,
                kind: spec.kind,
                text: body,
                context: spec.context.clone(),
            });
        }
        if next == end {
            break;
        }
        cursor = next.saturating_sub(8);
    }
}

fn is_code_boundary(line: &str) -> bool {
    line.starts_with("fn ")
        || line.starts_with("pub fn ")
        || line.starts_with("async fn ")
        || line.starts_with("pub async fn ")
        || line.starts_with("def ")
        || line.starts_with("class ")
        || line.starts_with("function ")
        || line.starts_with("export function ")
        || line.starts_with("func ")
        || line.starts_with("type ")
        || line.starts_with("impl ")
        || line.starts_with("struct ")
        || line.starts_with("enum ")
}

fn nearest_heading(lines: &[&str], start: usize) -> Option<String> {
    for line in lines[..=start.min(lines.len().saturating_sub(1))]
        .iter()
        .rev()
    {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            let heading = trimmed.trim_start_matches('#').trim();
            if !heading.is_empty() {
                return Some(heading.to_string());
            }
        }
    }
    None
}

fn byte_offset_for_line(text: &str, line: usize) -> usize {
    if line == 0 {
        return 0;
    }
    let mut count = 0;
    for (idx, byte) in text.bytes().enumerate() {
        if byte == b'\n' {
            count += 1;
            if count == line {
                return idx + 1;
            }
        }
    }
    text.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_chunks_keep_offsets_and_context() {
        let chunks = chunk_text("# Lexa\n\nBody\n\n## Search\n\nFind files", "markdown");
        assert!(chunks
            .iter()
            .any(|chunk| chunk.context.as_deref() == Some("Search")));
        assert!(chunks
            .iter()
            .all(|chunk| chunk.byte_end >= chunk.byte_start));
        assert!(chunks
            .iter()
            .all(|chunk| chunk.line_end >= chunk.line_start));
    }

    #[test]
    fn code_chunks_follow_symbol_boundaries() {
        let chunks = chunk_text(
            "use std::fs;\n\npub fn validate_config() {}\n\nfn search() {}",
            "code",
        );
        assert!(chunks
            .iter()
            .any(|chunk| chunk.text.contains("validate_config")));
        assert!(chunks.iter().any(|chunk| chunk.text.contains("fn search")));
    }
}
