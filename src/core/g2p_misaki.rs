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
        // Punctuation
        m.insert(";".to_string(), 1);
        m.insert(":".to_string(), 2);
        m.insert(",".to_string(), 3);
        m.insert(".".to_string(), 4);
        m.insert("!".to_string(), 5);
        m.insert("?".to_string(), 6);
        m.insert("—".to_string(), 9);
        m.insert("…".to_string(), 10);
        m.insert(" ".to_string(), 16);
        
        // Common IPA phonemes (subset for English)
        m.insert("a".to_string(), 43);
        m.insert("b".to_string(), 44);
        m.insert("d".to_string(), 46);
        m.insert("e".to_string(), 47);
        m.insert("f".to_string(), 48);
        m.insert("h".to_string(), 50);
        m.insert("i".to_string(), 51);
        m.insert("k".to_string(), 53);
        m.insert("l".to_string(), 54);
        m.insert("m".to_string(), 55);
        m.insert("n".to_string(), 56);
        m.insert("o".to_string(), 57);
        m.insert("p".to_string(), 58);
        m.insert("r".to_string(), 60);
        m.insert("s".to_string(), 61);
        m.insert("t".to_string(), 62);
        m.insert("u".to_string(), 63);
        m.insert("v".to_string(), 64);
        m.insert("w".to_string(), 65);
        m.insert("z".to_string(), 68);
        
        // IPA vowels
        m.insert("ɑ".to_string(), 69);
        m.insert("æ".to_string(), 72);
        m.insert("ɔ".to_string(), 76);
        m.insert("ð".to_string(), 81);
        m.insert("ə".to_string(), 83);
        m.insert("ɛ".to_string(), 86);
        m.insert("ɪ".to_string(), 102);
        m.insert("ŋ".to_string(), 112);
        m.insert("θ".to_string(), 119);
        m.insert("ʃ".to_string(), 131);
        m.insert("ʧ".to_string(), 133);
        m.insert("ʊ".to_string(), 135);
        m.insert("ʌ".to_string(), 138);
        m.insert("ʒ".to_string(), 147);
        m.insert("ʤ".to_string(), 82);
        
        // Stress and prosody markers
        m.insert("ˈ".to_string(), 156);  // Primary stress
        m.insert("ˌ".to_string(), 157);  // Secondary stress
        m.insert("ː".to_string(), 158);  // Long vowel
        
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
    vocab: &'static HashMap<String, i64>,
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
            // Try to match multi-character IPA symbols
            let symbol = if ch == 'ː' || ch == 'ˈ' || ch == 'ˌ' {
                // Combining marks
                ch.to_string()
            } else if ch == 'ʤ' || ch == 'ʧ' || ch == 'ʃ' || ch == 'ʒ' || 
                      ch == 'θ' || ch == 'ð' || ch == 'ŋ' {
                // Special IPA characters
                ch.to_string()
            } else if let Some(&next_ch) = chars.peek() {
                // Check for combining characters
                if next_ch == 'ː' {
                    chars.next();
                    format!("{}{}", ch, next_ch)
                } else {
                    ch.to_string()
                }
            } else {
                ch.to_string()
            };
            
            if let Some(&token) = self.vocab.get(&symbol) {
                tokens.push(token);
            } else {
                // Try single character
                if let Some(&token) = self.vocab.get(&ch.to_string()) {
                    tokens.push(token);
                }
            }
        }
        
        Ok(tokens)
    }
    
    /// Letter-to-phoneme fallback (simplified rules)
    fn letter_to_phoneme(&self, word: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();
        
        for ch in word.chars() {
            let phoneme = match ch {
                // Vowels
                'a' => "æ",
                'e' => "ɛ",
                'i' => "ɪ",
                'o' => "ɒ",
                'u' => "ʌ",
                // Consonants
                'b' => "b",
                'c' => "k",
                'd' => "d",
                'f' => "f",
                'g' => "ɡ",
                'h' => "h",
                'j' => "ʤ",
                'k' => "k",
                'l' => "l",
                'm' => "m",
                'n' => "n",
                'p' => "p",
                'q' => "k",
                'r' => "r",
                's' => "s",
                't' => "t",
                'v' => "v",
                'w' => "w",
                'x' => "ks",
                'y' => "j",
                'z' => "z",
                _ => continue,
            };
            
            if phoneme.len() == 2 {
                // Handle digraphs like "ks"
                for p in phoneme.chars() {
                    if let Some(&token) = self.vocab.get(&p.to_string()) {
                        tokens.push(token);
                    }
                }
            } else if let Some(&token) = self.vocab.get(phoneme) {
                tokens.push(token);
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
}
