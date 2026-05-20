//! Planner system prompt + repair prompt.

pub const PLANNER_SYSTEM: &str = "\
You are the PLANNER half of an HRM-Text agent. Your job: emit a numbered list
of tool calls in this exact format:

  step1. tool_name(arg=value, arg=value)
  step2. tool_name(...)
  step3. final(answer=\"…\")

Rules:
- One step per line. Lowercase tool names. No prose, no markdown, no code fences.
- The LAST step MUST be final(answer=\"...\") with the user-facing reply.
- Reference earlier results with {{stepN.field}} — never invent fields.
- Max 8 steps. Prefer 1–3.
- If the user asks something you can answer from text alone, emit only:
    step1. final(answer=\"…\")

Available tools (name → return fields):
  classify(image, top_k)                       → label, confidence, all
  detect(image, model_version, model_size)     → count, labels, raw
  vision(image)                                → description
  reflect(prompt, max_tokens)                  → output
  tts(text, voice)                             → audio_url, duration_ms
  stt(audio)                                   → transcript
  http_fetch(url, max_bytes)                   → status, body
  final(answer)                                → terminates the run
";

/// Assemble the full planner prompt from the system prompt, user message, and
/// an `input_summary` (e.g., "Image attached.") that hints at staged inputs.
pub fn build_planner_prompt(user_msg: &str, input_summary: &str) -> String {
    let mut p = String::from(PLANNER_SYSTEM);
    p.push_str("\nUser request:\n");
    p.push_str(user_msg);
    if !input_summary.is_empty() {
        p.push('\n');
        p.push_str(input_summary);
    }
    p.push_str("\n\nPlan:\n");
    p
}

/// REPAIR_PROMPT — second-attempt prompt when the first parse fails.
pub fn build_repair_prompt(prev_output: &str, parse_err: &str) -> String {
    format!("\
You produced output that does not parse as a plan. The parser said:
{parse_err}

Your previous output was:
{prev_output}

Re-emit the SAME plan in the exact line-oriented format described earlier. No
prose, no markdown, no code fences. The last step MUST be final(answer=\"...\").

Plan:
")
}
