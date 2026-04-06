#[derive(Debug, serde::Serialize)]
pub struct Envelope<T> {
    pub data: T,
    pub meta: ResponseMeta,
}

#[derive(Debug, serde::Serialize)]
pub struct ResponseMeta {
    pub latency_ms: f64,
    pub model_id: String,
    pub postprocessing_applied: bool,
    pub postprocess_steps: Vec<String>,
    pub warnings: Vec<String>,
    pub version: &'static str,
    pub request_id: String,
}

impl<T> Envelope<T> {
    pub fn new(data: T, meta: ResponseMeta) -> Self {
        Self { data, meta }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[test]
    fn test_envelope_serializes_data_and_meta() {
        #[derive(Serialize)]
        struct Payload {
            value: u32,
        }

        let meta = ResponseMeta {
            latency_ms: 42.0,
            model_id: "test-model".into(),
            postprocessing_applied: true,
            postprocess_steps: vec!["normalize".into()],
            warnings: vec![],
            version: "1.0.0",
            request_id: "req-123".into(),
        };
        let envelope = Envelope::new(Payload { value: 7 }, meta);
        let json = serde_json::to_string(&envelope).unwrap();
        assert!(json.contains("\"value\":7"));
        assert!(json.contains("\"latency_ms\":42.0"));
        assert!(json.contains("\"model_id\":\"test-model\""));
        assert!(json.contains("\"postprocessing_applied\":true"));
        assert!(json.contains("\"normalize\""));
    }

    #[test]
    fn test_envelope_postprocessing_false_when_steps_empty() {
        #[derive(Serialize)]
        struct Payload {
            ok: bool,
        }

        let meta = ResponseMeta {
            latency_ms: 1.0,
            model_id: "m".into(),
            postprocessing_applied: false,
            postprocess_steps: vec![],
            warnings: vec![],
            version: "1.0.0",
            request_id: "r".into(),
        };
        let env = Envelope::new(Payload { ok: true }, meta);
        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains("\"postprocessing_applied\":false"));
    }

    #[test]
    fn test_envelope_warnings_propagated() {
        #[derive(Serialize)]
        struct Payload {}

        let meta = ResponseMeta {
            latency_ms: 0.0,
            model_id: "m".into(),
            postprocessing_applied: true,
            postprocess_steps: vec![],
            warnings: vec!["clipping_detected".into()],
            version: "1.0.0",
            request_id: "r".into(),
        };
        let env = Envelope::new(Payload {}, meta);
        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains("clipping_detected"));
    }
}
