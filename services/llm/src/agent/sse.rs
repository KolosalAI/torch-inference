//! SSE event payload types for /llm/v1/agent/run.
//!
//! All variants serialize to a single JSON object suitable for the SSE `data:`
//! line. The `event:` line name is the variant's snake_case form, produced by
//! `AgentEvent::event_name()`.

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum AgentEvent {
    RunStart {
        run_id: String,
        model: String,
        deadline_ms: u64,
        max_steps: usize,
    },
    Plan {
        steps: Vec<PlanStep>,
        retries: u8,
    },
    StepStart {
        idx: usize,
        id: String,
        tool: String,
        args: Value,
    },
    StepResult {
        idx: usize,
        id: String,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        duration_ms: u64,
    },
    Error {
        kind: String,
        message: String,
    },
    Final {
        answer: String,
        steps_executed: usize,
        total_ms: u64,
        completed: bool,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanStep {
    pub id: String,
    pub tool: String,
    pub args: Value,
}

impl AgentEvent {
    pub fn event_name(&self) -> &'static str {
        match self {
            AgentEvent::RunStart { .. }   => "run_start",
            AgentEvent::Plan { .. }       => "plan",
            AgentEvent::StepStart { .. }  => "step_start",
            AgentEvent::StepResult { .. } => "step_result",
            AgentEvent::Error { .. }      => "error",
            AgentEvent::Final { .. }      => "final",
        }
    }

    /// Render as a complete SSE frame: `event: <name>\ndata: <json>\n\n`.
    pub fn to_sse_frame(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string());
        format!("event: {}\ndata: {}\n\n", self.event_name(), json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn run_start_serializes_with_event_name() {
        let ev = AgentEvent::RunStart {
            run_id: "01HZ".into(),
            model: "hrm-text-1b".into(),
            deadline_ms: 60_000,
            max_steps: 8,
        };
        assert_eq!(ev.event_name(), "run_start");
        let frame = ev.to_sse_frame();
        assert!(frame.starts_with("event: run_start\n"));
        assert!(frame.contains("\"run_id\":\"01HZ\""));
        assert!(frame.ends_with("\n\n"));
    }

    #[test]
    fn step_result_omits_value_when_none() {
        let ev = AgentEvent::StepResult {
            idx: 1, id: "step1".into(), ok: false,
            value: None, error: Some("timeout".into()), duration_ms: 5000,
        };
        let json: Value = serde_json::from_str(
            ev.to_sse_frame().split("data: ").nth(1).unwrap().trim()
        ).unwrap();
        assert_eq!(json["ok"], false);
        assert_eq!(json["error"], "timeout");
        assert!(json.get("value").is_none());
    }

    #[test]
    fn step_result_omits_error_when_some_value() {
        let ev = AgentEvent::StepResult {
            idx: 1, id: "step1".into(), ok: true,
            value: Some(json!({"label":"cat"})), error: None, duration_ms: 200,
        };
        let frame = ev.to_sse_frame();
        assert!(frame.contains("\"ok\":true"));
        assert!(frame.contains("\"label\":\"cat\""));
        assert!(!frame.contains("\"error\""));
    }
}
