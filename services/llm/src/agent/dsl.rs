//! Mini-DSL parser for planner output.
//!
//! Line-oriented, regex-driven, prose-tolerant. Spec §3.

use anyhow::{anyhow, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub id: String,        // "step1"
    pub tool: String,      // "classify"
    pub args: Vec<(String, ArgValue)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArgValue {
    Literal(Value),                              // string, int, float, bool
    Ref { step_id: String, field: String },      // {{step1.label}}
}

impl Step {
    pub fn args_as_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        for (k, v) in &self.args {
            m.insert(k.clone(), match v {
                ArgValue::Literal(val) => val.clone(),
                ArgValue::Ref { step_id, field } => Value::String(format!("{{{{{}.{}}}}}", step_id, field)),
            });
        }
        Value::Object(m)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("no steps recognized in planner output")]
    NoSteps,
    #[error("malformed step on line {line_no}: {detail}")]
    MalformedStep { line_no: usize, detail: String },
    #[error("malformed argument in {tool}: {detail}")]
    MalformedArg { tool: String, detail: String },
    #[error("unknown tool: {0}")]
    UnknownTool(String),
}

pub const KNOWN_TOOLS: &[&str] = &[
    "classify", "detect", "vision", "reflect", "tts", "stt", "http_fetch", "final",
];

static STEP_RE: OnceLock<Regex> = OnceLock::new();
static REF_RE: OnceLock<Regex> = OnceLock::new();

fn step_re() -> &'static Regex {
    STEP_RE.get_or_init(|| {
        Regex::new(r"^\s*(step\d+)\.\s*([a-z_]+)\s*\((.*)\)\s*$").unwrap()
    })
}

fn ref_re() -> &'static Regex {
    REF_RE.get_or_init(|| {
        Regex::new(r"^\{\{(step\d+)\.([a-z_][a-z0-9_]*)\}\}$").unwrap()
    })
}

/// Parse planner output into a list of Steps. Prose lines and blank lines are
/// silently skipped. Lines that LOOK like a step (start with `stepN.`) but
/// fail to parse return MalformedStep.
pub fn parse(text: &str) -> Result<Vec<Step>, ParseError> {
    let mut steps = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() { continue; }

        // Skip prose: any line that doesn't start with `step<digit>.`
        let looks_like_step = line.starts_with("step")
            && line.bytes().nth(4).map(|b| b.is_ascii_digit()).unwrap_or(false);
        if !looks_like_step { continue; }

        let caps = step_re().captures(line)
            .ok_or_else(|| ParseError::MalformedStep {
                line_no: i + 1,
                detail: format!("does not match `stepN. tool(args)`: {}", raw),
            })?;

        let id   = caps.get(1).unwrap().as_str().to_string();
        let tool = caps.get(2).unwrap().as_str().to_string();
        let args_str = caps.get(3).unwrap().as_str();

        if !KNOWN_TOOLS.contains(&tool.as_str()) {
            return Err(ParseError::UnknownTool(tool));
        }

        let args = parse_args(&tool, args_str)?;
        steps.push(Step { id, tool, args });
    }

    if steps.is_empty() { return Err(ParseError::NoSteps); }
    Ok(steps)
}

fn parse_args(tool: &str, s: &str) -> Result<Vec<(String, ArgValue)>, ParseError> {
    let s = s.trim();
    if s.is_empty() { return Ok(Vec::new()); }

    let mut out = Vec::new();
    for raw_pair in split_top_level(s) {
        let pair = raw_pair.trim();
        if pair.is_empty() { continue; }
        let (k, v) = pair.split_once('=').ok_or_else(|| ParseError::MalformedArg {
            tool: tool.to_string(),
            detail: format!("expected `key=value`, got `{}`", pair),
        })?;
        let key = k.trim().to_string();
        let val = parse_value(tool, v.trim())?;
        out.push((key, val));
    }
    Ok(out)
}

/// Split args on commas, respecting double-quoted strings and {{}} blocks.
fn split_top_level(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    let mut in_str = false;
    let mut brace_depth = 0i32;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' if brace_depth == 0 => { in_str = !in_str; buf.push(c); }
            '{' if !in_str && i + 1 < chars.len() && chars[i+1] == '{' => {
                brace_depth += 1; buf.push_str("{{"); i += 1;
            }
            '}' if !in_str && i + 1 < chars.len() && chars[i+1] == '}' => {
                brace_depth = brace_depth.saturating_sub(1); buf.push_str("}}"); i += 1;
            }
            ',' if !in_str && brace_depth == 0 => {
                parts.push(std::mem::take(&mut buf));
            }
            _ => buf.push(c),
        }
        i += 1;
    }
    if !buf.is_empty() { parts.push(buf); }
    parts
}

fn parse_value(tool: &str, v: &str) -> Result<ArgValue, ParseError> {
    // Ref?
    if let Some(caps) = ref_re().captures(v) {
        return Ok(ArgValue::Ref {
            step_id: caps.get(1).unwrap().as_str().to_string(),
            field:   caps.get(2).unwrap().as_str().to_string(),
        });
    }
    // Quoted string?
    if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
        return Ok(ArgValue::Literal(Value::String(v[1..v.len()-1].to_string())));
    }
    // Bool?
    match v {
        "true"  | "1" => return Ok(ArgValue::Literal(Value::Bool(true))),
        "false" | "0" => return Ok(ArgValue::Literal(Value::Bool(false))),
        _ => {}
    }
    // int / float
    if let Ok(n) = v.parse::<i64>()  { return Ok(ArgValue::Literal(json!(n))); }
    if let Ok(n) = v.parse::<f64>()  { return Ok(ArgValue::Literal(json!(n))); }
    // Bare word — accept as a string literal (e.g. `image=input`)
    if v.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !v.is_empty() {
        return Ok(ArgValue::Literal(Value::String(v.to_string())));
    }
    Err(ParseError::MalformedArg {
        tool: tool.to_string(),
        detail: format!("could not parse value `{}`", v),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_classify_step() {
        let p = parse("step1. classify(image=input, top_k=3)").unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].id, "step1");
        assert_eq!(p[0].tool, "classify");
        assert_eq!(p[0].args.len(), 2);
    }

    #[test]
    fn parses_ref_arg() {
        let p = parse("step2. tts(text={{step1.output}})").unwrap();
        match &p[0].args[0].1 {
            ArgValue::Ref { step_id, field } => {
                assert_eq!(step_id, "step1"); assert_eq!(field, "output");
            }
            other => panic!("expected ref, got {:?}", other),
        }
    }

    #[test]
    fn coerces_string_int_bool() {
        let p = parse(r#"step1. classify(image="input", top_k=3, raw=true)"#).unwrap();
        assert!(matches!(p[0].args[0].1, ArgValue::Literal(Value::String(ref s)) if s == "input"));
        assert!(matches!(p[0].args[1].1, ArgValue::Literal(Value::Number(_))));
        assert!(matches!(p[0].args[2].1, ArgValue::Literal(Value::Bool(true))));
    }

    #[test]
    fn skips_prose_lines() {
        let text = "Now I will analyze this.\n\nstep1. final(answer=\"42\")\n";
        let p = parse(text).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].tool, "final");
    }

    #[test]
    fn handles_commas_inside_strings() {
        let p = parse(r#"step1. reflect(prompt="hello, world", max_tokens=8)"#).unwrap();
        assert_eq!(p[0].args.len(), 2);
        assert!(matches!(p[0].args[0].1,
            ArgValue::Literal(Value::String(ref s)) if s == "hello, world"));
    }

    #[test]
    fn rejects_unknown_tool() {
        let err = parse("step1. magic(x=1)").unwrap_err();
        assert!(matches!(err, ParseError::UnknownTool(t) if t == "magic"));
    }

    #[test]
    fn rejects_no_steps() {
        let err = parse("just prose, nothing else").unwrap_err();
        assert!(matches!(err, ParseError::NoSteps));
    }

    #[test]
    fn rejects_malformed_step_with_step_prefix() {
        let err = parse("step1 missing dot").unwrap_err();
        assert!(matches!(err, ParseError::MalformedStep { .. }));
    }

    #[test]
    fn rejects_json_planner_output() {
        // Regression guard from spec §9.5 — JSON-shaped output must not parse.
        let err = parse(r#"[{"tool":"classify","args":{}}]"#).unwrap_err();
        // The line starts with `[`, not `stepN.`, so it's prose-skipped → NoSteps.
        assert!(matches!(err, ParseError::NoSteps));
    }

    #[test]
    fn parses_multi_step_plan() {
        let text = r#"
step1. classify(image=input, top_k=3)
step2. detect(image=input)
step3. reflect(prompt="A {{step1.label}} with {{step2.count}} things.", max_tokens=64)
step4. final(answer={{step3.output}})
"#;
        let p = parse(text).unwrap();
        assert_eq!(p.len(), 4);
        assert_eq!(p[3].tool, "final");
    }
}
