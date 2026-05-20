use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct HrmTokenizer {
    inner: Tokenizer,
}

impl HrmTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("load tokenizer at {}: {}", path.display(), e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let enc = self.inner.encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().iter().map(|&x| x as i64).collect())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }

    pub fn decode_single(&self, id: u32) -> Result<String> {
        self.decode(&[id])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/hrm-text-1b")
    }

    fn skip_if_no_model() -> Option<std::path::PathBuf> {
        let d = fixture_dir();
        if d.join("tokenizer.json").exists() { Some(d) } else { None }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let Some(dir) = skip_if_no_model() else {
            eprintln!("skipping: run `make hrm-download` to enable tokenizer tests");
            return;
        };
        let tok = HrmTokenizer::load(&dir).unwrap();
        let ids = tok.encode("hello world", true).unwrap();
        assert!(!ids.is_empty());
        let id_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        let text = tok.decode(&id_u32).unwrap();
        assert!(text.to_lowercase().contains("hello"));
    }
}
