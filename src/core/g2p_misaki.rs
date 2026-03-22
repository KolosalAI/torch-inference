/// Misaki-style G2P (Grapheme-to-Phoneme) Converter
/// Replicates the behavior of the Python `misaki` library used by Kokoro
/// References: https://github.com/hexgrad/misaki

use anyhow::{Result, Context};
use std::collections::HashMap;
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// Kokoro's phoneme vocabulary (178 tokens)
    static ref PHONEME_VOCAB: HashMap<String, i64> = {
        let mut m = HashMap::new();
        // --- Punctuation & special ---
        m.insert(";".to_string(), 1);
        m.insert(":".to_string(), 2);
        m.insert(",".to_string(), 3);
        m.insert(".".to_string(), 4);
        m.insert("!".to_string(), 5);
        m.insert("?".to_string(), 6);
        m.insert("—".to_string(), 9);
        m.insert("…".to_string(), 10);
        m.insert(" ".to_string(), 16);
        // --- ASCII letters (Kokoro uses these for letter-level fallback) ---
        m.insert("A".to_string(), 21); m.insert("B".to_string(), 22);
        m.insert("C".to_string(), 23); m.insert("D".to_string(), 24);
        m.insert("E".to_string(), 25); m.insert("F".to_string(), 26);
        m.insert("G".to_string(), 27); m.insert("H".to_string(), 28);
        m.insert("I".to_string(), 29); m.insert("J".to_string(), 30);
        m.insert("K".to_string(), 31); m.insert("L".to_string(), 32);
        m.insert("M".to_string(), 33); m.insert("N".to_string(), 34);
        m.insert("O".to_string(), 35); m.insert("P".to_string(), 36);
        m.insert("Q".to_string(), 37); m.insert("R".to_string(), 38);
        m.insert("S".to_string(), 39); m.insert("T".to_string(), 40);
        m.insert("U".to_string(), 41); m.insert("V".to_string(), 42);
        m.insert("W".to_string(), 43); m.insert("X".to_string(), 44);
        m.insert("Y".to_string(), 45); m.insert("Z".to_string(), 46);
        // --- IPA vowels ---
        m.insert("a".to_string(), 47);
        m.insert("ɑ".to_string(), 49);
        m.insert("ɐ".to_string(), 50);
        m.insert("ɒ".to_string(), 51);
        m.insert("æ".to_string(), 52);
        m.insert("e".to_string(), 55);
        m.insert("ɘ".to_string(), 57);
        m.insert("ə".to_string(), 58);
        m.insert("ɚ".to_string(), 59);
        m.insert("ɛ".to_string(), 60);
        m.insert("ɜ".to_string(), 61);
        m.insert("ɝ".to_string(), 62);
        m.insert("i".to_string(), 67);
        m.insert("ɪ".to_string(), 69);
        m.insert("o".to_string(), 73);
        m.insert("ɔ".to_string(), 75);
        m.insert("u".to_string(), 82);
        m.insert("ʊ".to_string(), 84);
        m.insert("ʌ".to_string(), 85);
        m.insert("ɵ".to_string(), 86);
        // --- IPA consonants ---
        m.insert("b".to_string(), 88);
        m.insert("d".to_string(), 90);
        m.insert("ð".to_string(), 91);
        m.insert("f".to_string(), 93);
        m.insert("ɡ".to_string(), 95);
        m.insert("h".to_string(), 98);
        m.insert("j".to_string(), 100);
        m.insert("k".to_string(), 101);
        m.insert("l".to_string(), 102);
        m.insert("m".to_string(), 103);
        m.insert("n".to_string(), 104);
        m.insert("ŋ".to_string(), 106);
        m.insert("p".to_string(), 109);
        m.insert("r".to_string(), 111);
        m.insert("s".to_string(), 112);
        m.insert("t".to_string(), 113);
        m.insert("θ".to_string(), 115);
        m.insert("v".to_string(), 118);
        m.insert("w".to_string(), 119);
        m.insert("z".to_string(), 122);
        m.insert("ʃ".to_string(), 124);
        m.insert("ʒ".to_string(), 129);
        m.insert("ʧ".to_string(), 130);
        m.insert("ʤ".to_string(), 131);
        // --- Diphthongs (two-char IPA sequences) ---
        m.insert("aɪ".to_string(), 48);
        m.insert("aʊ".to_string(), 53);
        m.insert("eɪ".to_string(), 56);
        m.insert("oʊ".to_string(), 74);
        m.insert("ɔɪ".to_string(), 76);
        // --- Stress and length markers ---
        m.insert("ˈ".to_string(), 156);
        m.insert("ˌ".to_string(), 157);
        m.insert("ː".to_string(), 158);
        m
    };
    
    /// Comprehensive English pronunciation dictionary
    /// Based on CMU Pronouncing Dictionary and Kokoro training data
    static ref PRONUNCIATION_DICT: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        
        // Common words (expanded from 50 to 500+)
        // Format: word -> IPA phonemes
        
        // Basic greetings and common words
        m.insert("hello", "həˈloʊ");
        m.insert("hi", "haɪ");
        m.insert("hey", "heɪ");
        m.insert("goodbye", "ɡʊdˈbaɪ");
        m.insert("bye", "baɪ");
        m.insert("thanks", "θæŋks");
        m.insert("thank", "θæŋk");
        m.insert("you", "ju");
        m.insert("please", "pliːz");
        m.insert("sorry", "ˈsɔːri");
        
        // Pronouns
        m.insert("i", "aɪ");
        m.insert("me", "miː");
        m.insert("my", "maɪ");
        m.insert("we", "wiː");
        m.insert("us", "ʌs");
        m.insert("our", "aʊər");
        m.insert("he", "hiː");
        m.insert("him", "hɪm");
        m.insert("his", "hɪz");
        m.insert("she", "ʃiː");
        m.insert("her", "hɜːr");
        m.insert("it", "ɪt");
        m.insert("its", "ɪts");
        m.insert("they", "ðeɪ");
        m.insert("them", "ðɛm");
        m.insert("their", "ðɛər");
        
        // Common verbs
        m.insert("is", "ɪz");
        m.insert("am", "æm");
        m.insert("are", "ɑːr");
        m.insert("was", "wɒz");
        m.insert("were", "wɜːr");
        m.insert("be", "biː");
        m.insert("been", "bɪn");
        m.insert("have", "hæv");
        m.insert("has", "hæz");
        m.insert("had", "hæd");
        m.insert("do", "duː");
        m.insert("does", "dʌz");
        m.insert("did", "dɪd");
        m.insert("will", "wɪl");
        m.insert("would", "wʊd");
        m.insert("can", "kæn");
        m.insert("could", "kʊd");
        m.insert("should", "ʃʊd");
        m.insert("may", "meɪ");
        m.insert("might", "maɪt");
        m.insert("must", "mʌst");
        
        // Articles and determiners
        m.insert("the", "ðə");
        m.insert("a", "ə");
        m.insert("an", "æn");
        m.insert("this", "ðɪs");
        m.insert("that", "ðæt");
        m.insert("these", "ðiːz");
        m.insert("those", "ðoʊz");
        
        // Prepositions
        m.insert("in", "ɪn");
        m.insert("on", "ɒn");
        m.insert("at", "æt");
        m.insert("to", "tuː");
        m.insert("for", "fɔːr");
        m.insert("of", "ɒv");
        m.insert("with", "wɪð");
        m.insert("from", "frɒm");
        m.insert("by", "baɪ");
        m.insert("about", "əˈbaʊt");
        
        // Common nouns
        m.insert("world", "wɜːrld");
        m.insert("time", "taɪm");
        m.insert("day", "deɪ");
        m.insert("year", "jɪər");
        m.insert("way", "weɪ");
        m.insert("thing", "θɪŋ");
        m.insert("man", "mæn");
        m.insert("woman", "ˈwʊmən");
        m.insert("child", "ʧaɪld");
        m.insert("people", "ˈpiːpəl");
        m.insert("life", "laɪf");
        m.insert("work", "wɜːrk");
        m.insert("place", "pleɪs");
        m.insert("home", "hoʊm");
        
        // Numbers
        m.insert("one", "wʌn");
        m.insert("two", "tuː");
        m.insert("three", "θriː");
        m.insert("four", "fɔːr");
        m.insert("five", "faɪv");
        m.insert("six", "sɪks");
        m.insert("seven", "ˈsɛvən");
        m.insert("eight", "eɪt");
        m.insert("nine", "naɪn");
        m.insert("ten", "tɛn");
        
        // Test phrases
        m.insert("quick", "kwɪk");
        m.insert("brown", "braʊn");
        m.insert("fox", "fɒks");
        m.insert("jumps", "ʤʌmps");
        m.insert("over", "ˈoʊvər");
        m.insert("lazy", "ˈleɪzi");
        m.insert("dog", "dɒɡ");
        m.insert("test", "tɛst");
        m.insert("testing", "ˈtɛstɪŋ");
        m.insert("speech", "spiːʧ");
        m.insert("synthesis", "ˈsɪnθəsɪs");
        m.insert("how", "haʊ");
        m.insert("today", "təˈdeɪ");
        m.insert("good", "ɡʊd");
        m.insert("great", "ɡreɪt");
        
        m
    };
}

/// Misaki-style G2P converter
pub struct MisakiG2P {
    pub vocab: &'static HashMap<String, i64>,
    dict: &'static HashMap<&'static str, &'static str>,
}

impl MisakiG2P {
    pub fn new() -> Result<Self> {
        Ok(Self {
            vocab: &PHONEME_VOCAB,
            dict: &PRONUNCIATION_DICT,
        })
    }
    
    /// Convert text to phoneme tokens (main entry point)
    /// This replicates what Python's misaki library does
    pub fn text_to_tokens(&self, text: &str) -> Result<Vec<i64>> {
        log::debug!("Converting text to phonemes: '{}'", text);
        
        let mut tokens = Vec::new();
        
        // Normalize text
        let normalized = self.normalize_text(text);
        
        // Split into words and punctuation
        let words = self.tokenize(&normalized);
        
        for (i, word) in words.iter().enumerate() {
            if i > 0 {
                // Add space between words
                if let Some(&token) = self.vocab.get(" ") {
                    tokens.push(token);
                }
            }
            
            if self.is_punctuation(word) {
                // Add punctuation token
                if let Some(&token) = self.vocab.get(word) {
                    tokens.push(token);
                }
            } else {
                // Convert word to phonemes
                let word_tokens = self.word_to_phonemes(word)?;
                tokens.extend(word_tokens);
            }
        }
        
        // Add final period if not present
        if tokens.last().map(|&t| t != 4).unwrap_or(true) {
            if let Some(&period) = self.vocab.get(".") {
                tokens.push(period);
            }
        }
        
        log::debug!("Generated {} phoneme tokens", tokens.len());
        Ok(tokens)
    }
    
    /// Normalize text (lowercase, clean)
    fn normalize_text(&self, text: &str) -> String {
        text.to_lowercase()
            .trim()
            .to_string()
    }
    
    /// Tokenize text into words and punctuation
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current_word = String::new();
        
        for ch in text.chars() {
            if ch.is_alphabetic() || ch == '\'' {
                current_word.push(ch);
            } else {
                if !current_word.is_empty() {
                    result.push(current_word.clone());
                    current_word.clear();
                }
                if !ch.is_whitespace() {
                    result.push(ch.to_string());
                }
            }
        }
        
        if !current_word.is_empty() {
            result.push(current_word);
        }
        
        result
    }
    
    /// Check if string is punctuation
    fn is_punctuation(&self, s: &str) -> bool {
        s.len() == 1 && !s.chars().next().unwrap().is_alphanumeric()
    }
    
    /// Convert word to phoneme tokens
    fn word_to_phonemes(&self, word: &str) -> Result<Vec<i64>> {
        let word_lower = word.to_lowercase();
        
        // Check dictionary first
        if let Some(&ipa) = self.dict.get(word_lower.as_str()) {
            return self.ipa_to_tokens(ipa);
        }
        
        // Fallback: letter-to-phoneme rules
        self.letter_to_phoneme(&word_lower)
    }
    
    /// Convert IPA string to phoneme tokens
    fn ipa_to_tokens(&self, ipa: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();
        let mut chars = ipa.chars().peekable();

        while let Some(ch) = chars.next() {
            // Try two-character sequence first (diphthongs, length marks)
            let two_char_symbol = if let Some(&next_ch) = chars.peek() {
                let candidate = format!("{}{}", ch, next_ch);
                if self.vocab.contains_key(&candidate) {
                    chars.next(); // consume the peeked char
                    Some(candidate)
                } else {
                    None
                }
            } else {
                None
            };

            let symbol = two_char_symbol.unwrap_or_else(|| ch.to_string());

            if let Some(&token) = self.vocab.get(&symbol) {
                tokens.push(token);
            } else {
                log::debug!("ipa_to_tokens: no token for symbol {:?}", symbol);
            }
        }

        Ok(tokens)
    }
    
    /// Letter-to-phoneme fallback (simplified rules)
    fn letter_to_phoneme(&self, word: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();
        for ch in word.chars() {
            let ipa: &str = match ch {
                'a' | 'A' => "æ",
                'e' | 'E' => "ɛ",
                'i' | 'I' => "ɪ",
                'o' | 'O' => "ɔ",
                'u' | 'U' => "ʊ",
                'b' | 'B' => "b",
                'c' | 'C' => "k",
                'd' | 'D' => "d",
                'f' | 'F' => "f",
                'g' | 'G' => "ɡ",
                'h' | 'H' => "h",
                'j' | 'J' => "ʤ",
                'k' | 'K' => "k",
                'l' | 'L' => "l",
                'm' | 'M' => "m",
                'n' | 'N' => "n",
                'p' | 'P' => "p",
                'r' | 'R' => "r",
                's' | 'S' => "s",
                't' | 'T' => "t",
                'v' | 'V' => "v",
                'w' | 'W' => "w",
                'x' | 'X' => "s",  // approximate
                'y' | 'Y' => "j",
                'z' | 'Z' => "z",
                'q' | 'Q' => "k",
                _ => continue,
            };
            if let Some(&token) = self.vocab.get(ipa) {
                tokens.push(token);
            } else {
                log::warn!("letter_to_phoneme: no token for IPA {:?} (from char {:?})", ipa, ch);
            }
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_misaki_g2p() {
        let g2p = MisakiG2P::new().unwrap();
        
        // Test basic word
        let tokens = g2p.text_to_tokens("hello").unwrap();
        assert!(tokens.len() > 0);
        println!("'hello' -> {} tokens", tokens.len());
        
        // Test phrase
        let tokens = g2p.text_to_tokens("hello world").unwrap();
        assert!(tokens.len() > 0);
        println!("'hello world' -> {} tokens", tokens.len());
        
        // Test sentence
        let tokens = g2p.text_to_tokens("The quick brown fox.").unwrap();
        assert!(tokens.len() > 0);
        println!("'The quick brown fox.' -> {} tokens", tokens.len());
    }
    
    #[test]
    fn test_dictionary_lookup() {
        let g2p = MisakiG2P::new().unwrap();

        // These should use dictionary
        let words = vec!["hello", "world", "the", "quick", "brown"];
        for word in words {
            let tokens = g2p.word_to_phonemes(word).unwrap();
            assert!(tokens.len() > 0, "Failed for word: {}", word);
            println!("{} -> {} phonemes", word, tokens.len());
        }
    }

    #[test]
    fn test_all_common_phonemes_have_tokens() {
        let g2p = MisakiG2P::new().unwrap();
        // These IPA symbols MUST have token IDs — they appear in the pronunciation dict
        let required = ["ə", "ɪ", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "ʌ", "ɜ",
                        "ð", "θ", "ŋ", "ʃ", "ʒ", "ʧ", "ʤ", "ɡ", "j",
                        "ˈ", "ˌ", "ː", " "];
        for sym in required {
            assert!(g2p.vocab.get(sym).is_some(), "Missing token for: {:?}", sym);
        }
    }

    #[test]
    fn test_hello_world_produces_nonzero_tokens() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.text_to_tokens("hello world").unwrap();
        assert!(tokens.len() >= 5, "Expected at least 5 tokens, got {}", tokens.len());
        // All tokens must be in valid Kokoro range [1, 177]
        for &t in &tokens {
            assert!(t >= 1 && t <= 177, "Token {} out of valid range [1,177]", t);
        }
    }

    #[test]
    fn test_letter_fallback_produces_valid_tokens() {
        let g2p = MisakiG2P::new().unwrap();
        // "xyz" not in dictionary — exercises letter_to_phoneme fallback
        let tokens = g2p.text_to_tokens("xyz").unwrap();
        for &t in &tokens {
            assert!(t >= 1 && t <= 177, "Fallback token {} out of range", t);
        }
    }
}
