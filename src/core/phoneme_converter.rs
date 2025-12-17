/// Phoneme Converter - Text to IPA phonemes for TTS
/// Implements grapheme-to-phoneme conversion for Kokoro TTS
use anyhow::{Result, Context};
use std::collections::HashMap;
use lazy_static::lazy_static;

/// Simple English G2P rules (basic implementation)
/// For production, this should use espeak-ng or a proper G2P library
pub struct PhonemeConverter {
    vocab: HashMap<String, i64>,
}

impl PhonemeConverter {
    pub fn new() -> Result<Self> {
        // Load vocab from Kokoro config
        let vocab = Self::build_vocab();
        Ok(Self { vocab })
    }
    
    /// Build vocabulary from Kokoro's phoneme set
    fn build_vocab() -> HashMap<String, i64> {
        let mut vocab = HashMap::new();
        
        // Punctuation and special characters
        vocab.insert(";".to_string(), 1);
        vocab.insert(":".to_string(), 2);
        vocab.insert(",".to_string(), 3);
        vocab.insert(".".to_string(), 4);
        vocab.insert("!".to_string(), 5);
        vocab.insert("?".to_string(), 6);
        vocab.insert(" ".to_string(), 16);
        
        // Common English phonemes (IPA)
        vocab.insert("a".to_string(), 43);
        vocab.insert("b".to_string(), 44);
        vocab.insert("d".to_string(), 46);
        vocab.insert("e".to_string(), 47);
        vocab.insert("f".to_string(), 48);
        vocab.insert("h".to_string(), 50);
        vocab.insert("i".to_string(), 51);
        vocab.insert("k".to_string(), 53);
        vocab.insert("l".to_string(), 54);
        vocab.insert("m".to_string(), 55);
        vocab.insert("n".to_string(), 56);
        vocab.insert("o".to_string(), 57);
        vocab.insert("p".to_string(), 58);
        vocab.insert("r".to_string(), 60);
        vocab.insert("s".to_string(), 61);
        vocab.insert("t".to_string(), 62);
        vocab.insert("u".to_string(), 63);
        vocab.insert("v".to_string(), 64);
        vocab.insert("w".to_string(), 65);
        vocab.insert("z".to_string(), 68);
        
        // IPA vowels
        vocab.insert("ə".to_string(), 83); // schwa
        vocab.insert("ɛ".to_string(), 86); // DRESS
        vocab.insert("ɪ".to_string(), 102); // KIT
        vocab.insert("ɑ".to_string(), 69); // PALM
        vocab.insert("ɔ".to_string(), 76); // THOUGHT
        vocab.insert("ʊ".to_string(), 135); // FOOT
        vocab.insert("æ".to_string(), 72); // TRAP
        
        // IPA consonants
        vocab.insert("ð".to_string(), 81); // this
        vocab.insert("θ".to_string(), 119); // think
        vocab.insert("ʃ".to_string(), 131); // ship
        vocab.insert("ʒ".to_string(), 147); // measure
        vocab.insert("ŋ".to_string(), 112); // sing
        vocab.insert("ʤ".to_string(), 82); // judge
        vocab.insert("ʧ".to_string(), 133); // church
        
        // Stress markers
        vocab.insert("ˈ".to_string(), 156); // primary stress
        vocab.insert("ˌ".to_string(), 157); // secondary stress
        vocab.insert("ː".to_string(), 158); // long vowel
        
        vocab
    }
    
    /// Convert text to phoneme tokens
    /// This is a simplified implementation - proper G2P requires espeak-ng
    pub fn text_to_phonemes(&self, text: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();
        
        // Simple word-by-word conversion
        let words = text.split_whitespace();
        
        for (i, word) in words.enumerate() {
            if i > 0 {
                // Add space token
                if let Some(&token) = self.vocab.get(" ") {
                    tokens.push(token);
                }
            }
            
            // Convert word to phonemes (simplified)
            let phonemes = self.word_to_phonemes(word)?;
            tokens.extend(phonemes);
        }
        
        // Add end marker
        if let Some(&token) = self.vocab.get(".") {
            tokens.push(token);
        }
        
        Ok(tokens)
    }
    
    /// Convert a single word to phonemes
    /// This is a VERY simplified implementation
    /// Real implementation should use espeak-ng or similar
    fn word_to_phonemes(&self, word: &str) -> Result<Vec<i64>> {
        let word_lower = word.to_lowercase();
        let mut phonemes = Vec::new();
        
        // Handle punctuation
        if let Some(&token) = self.vocab.get(&word_lower) {
            return Ok(vec![token]);
        }
        
        // Simple letter-to-phoneme mapping (very basic)
        for ch in word_lower.chars() {
            let phoneme = match ch {
                'a' => "æ",  // TRAP vowel
                'e' => "ɛ",  // DRESS vowel
                'i' => "ɪ",  // KIT vowel
                'o' => "ɔ",  // THOUGHT vowel
                'u' => "ʊ",  // FOOT vowel
                'h' => "h",
                'l' => "l",
                'w' => "w",
                'r' => "r",
                'd' => "d",
                't' => "t",
                's' => "s",
                'n' => "n",
                'p' => "p",
                'b' => "b",
                'k' => "k",
                'g' => "g",
                'f' => "f",
                'v' => "v",
                'm' => "m",
                'y' => "j",
                'z' => "z",
                _ => continue,
            };
            
            if let Some(&token) = self.vocab.get(phoneme) {
                phonemes.push(token);
            }
        }
        
        Ok(phonemes)
    }
    
    /// Get phoneme from token ID
    pub fn token_to_phoneme(&self, token: i64) -> Option<String> {
        for (phoneme, &id) in &self.vocab {
            if id == token {
                return Some(phoneme.clone());
            }
        }
        None
    }
}

/// Enhanced G2P using simple dictionary
pub struct EnhancedG2P {
    converter: PhonemeConverter,
    dictionary: HashMap<String, Vec<i64>>,
}

impl EnhancedG2P {
    pub fn new() -> Result<Self> {
        let converter = PhonemeConverter::new()?;
        let dictionary = Self::build_dictionary(&converter);
        
        Ok(Self {
            converter,
            dictionary,
        })
    }
    
    /// Build a simple pronunciation dictionary
    fn build_dictionary(converter: &PhonemeConverter) -> HashMap<String, Vec<i64>> {
        let mut dict = HashMap::new();
        
        // Common words with their IPA representations
        // Format: word -> phoneme string
        let pronunciations = vec![
            ("hello", "hɛloʊ"),
            ("world", "wɜrld"),
            ("the", "ðə"),
            ("quick", "kwɪk"),
            ("brown", "braʊn"),
            ("fox", "fɑks"),
            ("jumps", "ʤəmps"),
            ("over", "oʊvər"),
            ("lazy", "leɪzi"),
            ("dog", "dɔg"),
            ("test", "tɛst"),
            ("testing", "tɛstɪŋ"),
            ("speech", "spiʧ"),
            ("synthesis", "sɪnθəsɪs"),
            ("one", "wən"),
            ("two", "tu"),
            ("three", "θri"),
            ("four", "fɔr"),
            ("five", "faɪv"),
            ("how", "haʊ"),
            ("are", "ɑr"),
            ("you", "ju"),
            ("today", "tədeɪ"),
        ];
        
        for (word, ipa) in pronunciations {
            let tokens = Self::ipa_to_tokens(ipa, converter);
            dict.insert(word.to_string(), tokens);
        }
        
        dict
    }
    
    /// Convert IPA string to token IDs
    fn ipa_to_tokens(ipa: &str, converter: &PhonemeConverter) -> Vec<i64> {
        let mut tokens = Vec::new();
        let mut chars = ipa.chars().peekable();
        
        while let Some(ch) = chars.next() {
            // Try to match multi-character IPA symbols
            let phoneme = if ch == 'ʤ' || ch == 'ʧ' || ch == 'ʃ' || ch == 'ʒ' || 
                             ch == 'θ' || ch == 'ð' || ch == 'ŋ' {
                ch.to_string()
            } else {
                ch.to_string()
            };
            
            if let Some(&token) = converter.vocab.get(&phoneme) {
                tokens.push(token);
            }
        }
        
        tokens
    }
    
    /// Convert text to phonemes with dictionary lookup
    pub fn text_to_phonemes(&self, text: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            if i > 0 {
                // Add space
                if let Some(&space_token) = self.converter.vocab.get(" ") {
                    tokens.push(space_token);
                }
            }
            
            // Clean word (remove punctuation)
            let clean_word = word.to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_string();
            
            // Look up in dictionary first
            if let Some(phonemes) = self.dictionary.get(&clean_word) {
                tokens.extend(phonemes);
            } else {
                // Fallback to basic converter
                let phonemes = self.converter.word_to_phonemes(&clean_word)?;
                tokens.extend(phonemes);
            }
            
            // Add punctuation at end
            for ch in word.chars().rev() {
                if !ch.is_alphabetic() {
                    if let Some(&token) = self.converter.vocab.get(&ch.to_string()) {
                        tokens.push(token);
                        break;
                    }
                }
            }
        }
        
        // Add final period
        if let Some(&period) = self.converter.vocab.get(".") {
            tokens.push(period);
        }
        
        log::debug!("Text: '{}' -> {} phoneme tokens", text, tokens.len());
        
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phoneme_converter() {
        let converter = PhonemeConverter::new().unwrap();
        let tokens = converter.text_to_phonemes("hello").unwrap();
        assert!(tokens.len() > 0);
    }
    
    #[test]
    fn test_enhanced_g2p() {
        let g2p = EnhancedG2P::new().unwrap();
        let tokens = g2p.text_to_phonemes("hello world").unwrap();
        assert!(tokens.len() > 0);
        println!("Tokens: {:?}", tokens);
    }
}
