//! HRM-Text agentic orchestration layer.
//!
//! Planner/executor loop mirroring HRM-Text's hierarchical recurrent
//! structure. See docs/superpowers/specs/2026-05-20-hrm-agentic-orchestration-design.md.

pub mod sse;
pub mod dsl;
pub mod tool;
pub mod tools;
pub mod planner;
pub mod prompt;
pub mod executor;
