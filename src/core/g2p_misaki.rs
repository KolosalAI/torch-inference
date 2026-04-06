/// Misaki-style G2P (Grapheme-to-Phoneme) Converter
/// Uses the actual Kokoro vocabulary from config.json (114 symbols, 178-token space)
/// Phoneme strings are processed character-by-character matching the Python KModel.forward()
/// References: https://github.com/hexgrad/misaki, https://github.com/hexgrad/kokoro
use anyhow::Result;
use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    /// Exact Kokoro vocabulary from config.json (hexgrad/Kokoro-82M)
    /// Maps each Unicode phoneme character to its integer token ID
    static ref PHONEME_VOCAB: HashMap<String, i64> = {
        let mut m = HashMap::new();
        // Punctuation & special
        m.insert(";".to_string(), 1);
        m.insert(":".to_string(), 2);
        m.insert(",".to_string(), 3);
        m.insert(".".to_string(), 4);
        m.insert("!".to_string(), 5);
        m.insert("?".to_string(), 6);
        m.insert("\u{2014}".to_string(), 9);   // — em dash
        m.insert("\u{2026}".to_string(), 10);  // … ellipsis
        m.insert("\u{201C}".to_string(), 11);  // " left double quote
        m.insert("(".to_string(), 12);
        m.insert(")".to_string(), 13);
        m.insert("\u{201D}".to_string(), 14);  // " right double quote
        m.insert("\u{201E}".to_string(), 15);  // „ low double quote
        m.insert(" ".to_string(), 16);
        m.insert("\u{0303}".to_string(), 17);  // ̃ combining tilde
        m.insert("\u{02A3}".to_string(), 18);  // ʣ
        m.insert("\u{02A5}".to_string(), 19);  // ʥ
        m.insert("\u{02A6}".to_string(), 20);  // ʦ
        m.insert("\u{02A8}".to_string(), 21);  // ʨ
        m.insert("\u{1D5D}".to_string(), 22);  // ᵝ
        m.insert("\u{AB67}".to_string(), 23);  // ꭧ
        // Capital letters used as Kokoro diphthong encodings
        m.insert("A".to_string(), 24);   // /eɪ/ diphthong
        m.insert("I".to_string(), 25);   // /aɪ/ diphthong
        m.insert("O".to_string(), 31);   // /oʊ/ diphthong
        m.insert("Q".to_string(), 33);   // rare
        m.insert("S".to_string(), 35);   // rare
        m.insert("T".to_string(), 36);   // flap/tap (replaces ɾ)
        m.insert("W".to_string(), 39);   // /aʊ/ diphthong
        m.insert("Y".to_string(), 41);   // /ɔɪ/ diphthong
        m.insert("\u{1D4A}".to_string(), 42);  // ᵊ reduced vowel
        // Lowercase IPA (used as phoneme characters)
        m.insert("a".to_string(), 43);
        m.insert("b".to_string(), 44);
        m.insert("c".to_string(), 45);
        m.insert("d".to_string(), 46);
        m.insert("e".to_string(), 47);
        m.insert("f".to_string(), 48);
        m.insert("h".to_string(), 50);
        m.insert("i".to_string(), 51);
        m.insert("j".to_string(), 52);
        m.insert("k".to_string(), 53);
        m.insert("l".to_string(), 54);
        m.insert("m".to_string(), 55);
        m.insert("n".to_string(), 56);
        m.insert("o".to_string(), 57);
        m.insert("p".to_string(), 58);
        m.insert("q".to_string(), 59);
        m.insert("r".to_string(), 60);
        m.insert("s".to_string(), 61);
        m.insert("t".to_string(), 62);
        m.insert("u".to_string(), 63);
        m.insert("v".to_string(), 64);
        m.insert("w".to_string(), 65);
        m.insert("x".to_string(), 66);
        m.insert("y".to_string(), 67);
        m.insert("z".to_string(), 68);
        // Extended IPA
        m.insert("\u{0251}".to_string(), 69);  // ɑ
        m.insert("\u{0250}".to_string(), 70);  // ɐ
        m.insert("\u{0252}".to_string(), 71);  // ɒ
        m.insert("\u{00E6}".to_string(), 72);  // æ
        m.insert("\u{03B2}".to_string(), 75);  // β
        m.insert("\u{0254}".to_string(), 76);  // ɔ
        m.insert("\u{0255}".to_string(), 77);  // ɕ
        m.insert("\u{00E7}".to_string(), 78);  // ç
        m.insert("\u{0256}".to_string(), 80);  // ɖ
        m.insert("\u{00F0}".to_string(), 81);  // ð
        m.insert("\u{02A4}".to_string(), 82);  // ʤ
        m.insert("\u{0259}".to_string(), 83);  // ə
        m.insert("\u{025A}".to_string(), 85);  // ɚ
        m.insert("\u{025B}".to_string(), 86);  // ɛ
        m.insert("\u{025C}".to_string(), 87);  // ɜ
        m.insert("\u{025F}".to_string(), 90);  // ɟ
        m.insert("\u{0261}".to_string(), 92);  // ɡ
        m.insert("\u{0265}".to_string(), 99);  // ɥ
        m.insert("\u{0268}".to_string(), 101); // ɨ
        m.insert("\u{026A}".to_string(), 102); // ɪ
        m.insert("\u{029D}".to_string(), 103); // ʝ
        m.insert("\u{026F}".to_string(), 110); // ɯ
        m.insert("\u{0270}".to_string(), 111); // ɰ
        m.insert("\u{014B}".to_string(), 112); // ŋ
        m.insert("\u{0273}".to_string(), 113); // ɳ
        m.insert("\u{0272}".to_string(), 114); // ɲ
        m.insert("\u{0274}".to_string(), 115); // ɴ
        m.insert("\u{00F8}".to_string(), 116); // ø
        m.insert("\u{0278}".to_string(), 118); // ɸ
        m.insert("\u{03B8}".to_string(), 119); // θ
        m.insert("\u{0153}".to_string(), 120); // œ
        m.insert("\u{0279}".to_string(), 123); // ɹ (American English r)
        m.insert("\u{027E}".to_string(), 125); // ɾ (flap, usually replaced by T)
        m.insert("\u{027B}".to_string(), 126); // ɻ
        m.insert("\u{0281}".to_string(), 128); // ʁ
        m.insert("\u{027D}".to_string(), 129); // ɽ
        m.insert("\u{0282}".to_string(), 130); // ʂ
        m.insert("\u{0283}".to_string(), 131); // ʃ
        m.insert("\u{0288}".to_string(), 132); // ʈ
        m.insert("\u{02A7}".to_string(), 133); // ʧ
        m.insert("\u{028A}".to_string(), 135); // ʊ
        m.insert("\u{028B}".to_string(), 136); // ʋ
        m.insert("\u{028C}".to_string(), 138); // ʌ
        m.insert("\u{0263}".to_string(), 139); // ɣ
        m.insert("\u{0264}".to_string(), 140); // ɤ
        m.insert("\u{03C7}".to_string(), 142); // χ
        m.insert("\u{028E}".to_string(), 143); // ʎ
        m.insert("\u{0292}".to_string(), 147); // ʒ
        m.insert("\u{0294}".to_string(), 148); // ʔ
        // Stress and length markers
        m.insert("\u{02C8}".to_string(), 156); // ˈ primary stress
        m.insert("\u{02CC}".to_string(), 157); // ˌ secondary stress
        m.insert("\u{02D0}".to_string(), 158); // ː length mark
        m.insert("\u{02B0}".to_string(), 162); // ʰ aspirated
        m.insert("\u{02B2}".to_string(), 164); // ʲ palatalized
        // Tone/intonation
        m.insert("\u{2193}".to_string(), 169); // ↓
        m.insert("\u{2192}".to_string(), 171); // →
        m.insert("\u{2197}".to_string(), 172); // ↗
        m.insert("\u{2198}".to_string(), 173); // ↘
        m.insert("\u{1D7B}".to_string(), 177); // ᵻ
        m
    };

    /// English pronunciation dictionary using actual misaki/Kokoro phoneme strings.
    /// These are direct outputs of the misaki G2P library (hexgrad/misaki), using
    /// Kokoro's single-character diphthong encodings:
    ///   A = /eɪ/,  I = /aɪ/,  O = /oʊ/,  W = /aʊ/,  Y = /ɔɪ/,  T = flap ɾ
    ///   ɹ = American English /r/
    static ref PRONUNCIATION_DICT: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        // Greetings
        m.insert("hello",    "həlˈO");
        m.insert("hi",       "hˈI");
        m.insert("hey",      "hˈA");
        m.insert("goodbye",  "ɡˌʊdbˈI");
        m.insert("bye",      "bˈI");
        m.insert("thanks",   "θˈæŋks");
        m.insert("thank",    "θˈæŋk");
        m.insert("please",   "plˈiz");
        m.insert("sorry",    "sˈɔɹi");
        m.insert("you",      "ju");

        // Pronouns
        m.insert("i",        "ˈI");
        m.insert("me",       "mˌi");
        m.insert("my",       "mI");
        m.insert("we",       "wi");
        m.insert("us",       "ˌʌs");
        m.insert("our",      "ˌWəɹ");
        m.insert("he",       "hi");
        m.insert("him",      "hˌɪm");
        m.insert("his",      "hɪz");
        m.insert("she",      "ʃi");
        m.insert("her",      "hɜɹ");
        m.insert("it",       "ɪt");
        m.insert("its",      "ɪts");
        m.insert("they",     "ðA");
        m.insert("them",     "ðˌɛm");
        m.insert("their",    "ðɛɹ");
        m.insert("who",      "hˌu");
        m.insert("what",     "wˌʌt");
        m.insert("which",    "wˌɪʧ");
        m.insert("where",    "wˌɛɹ");
        m.insert("when",     "wˌɛn");
        m.insert("why",      "wˌI");
        m.insert("how",      "hˌW");

        // Common verbs
        m.insert("is",       "ɪz");
        m.insert("am",       "æm");
        m.insert("are",      "ɑɹ");
        m.insert("was",      "wʌz");
        m.insert("were",     "wɜɹ");
        m.insert("be",       "bi");
        m.insert("been",     "bɪn");
        m.insert("have",     "hæv");
        m.insert("has",      "hæz");
        m.insert("had",      "hæd");
        m.insert("do",       "dˈu");
        m.insert("does",     "dˈʌz");
        m.insert("did",      "dˈɪd");
        m.insert("will",     "wɪl");
        m.insert("would",    "wʊd");
        m.insert("can",      "kæn");
        m.insert("could",    "kʊd");
        m.insert("should",   "ʃˌʊd");
        m.insert("may",      "mˈA");
        m.insert("might",    "mˌIt");
        m.insert("must",     "mˈʌst");
        m.insert("say",      "sˈA");
        m.insert("said",     "sˈɛd");
        m.insert("go",       "ɡˌO");
        m.insert("went",     "wˈɛnt");
        m.insert("come",     "kˈʌm");
        m.insert("came",     "kˈAm");
        m.insert("know",     "nˈO");
        m.insert("think",    "θˈɪŋk");
        m.insert("see",      "sˈi");
        m.insert("look",     "lˈʊk");
        m.insert("make",     "mˌAk");
        m.insert("take",     "tˈAk");
        m.insert("give",     "ɡˈɪv");
        m.insert("find",     "fˈInd");
        m.insert("tell",     "tˈɛl");
        m.insert("ask",      "ˈæsk");
        m.insert("keep",     "kˈip");
        m.insert("hold",     "hˈOld");
        m.insert("turn",     "tˈɜɹn");
        m.insert("show",     "ʃˈO");
        m.insert("start",    "stˈɑɹt");
        m.insert("stop",     "stˈɑp");
        m.insert("play",     "plˈA");
        m.insert("run",      "ɹˈʌn");
        m.insert("move",     "mˈuv");
        m.insert("get",      "ɡɛt");
        m.insert("put",      "pˌʊt");
        m.insert("set",      "sˈɛt");
        m.insert("write",    "ɹˈIt");
        m.insert("help",     "hˈɛlp");
        m.insert("need",     "nˈid");
        m.insert("want",     "wˈɑnt");
        m.insert("feel",     "fˈil");
        m.insert("try",      "tɹˈI");
        m.insert("leave",    "lˈiv");
        m.insert("call",     "kˈɔl");
        m.insert("let",      "lˈɛt");
        m.insert("begin",    "bəɡˈɪn");
        m.insert("live",     "lˈIv");
        m.insert("use",      "jˈuz");
        m.insert("listen",   "lˈɪsᵊn");
        m.insert("hear",     "hˈɪɹ");
        m.insert("speak",    "spˈik");

        // Articles / determiners
        m.insert("the",      "ði");
        m.insert("a",        "A");
        m.insert("an",       "æn");
        m.insert("this",     "ðɪs");
        m.insert("that",     "ðæt");
        m.insert("these",    "ðiz");
        m.insert("those",    "ðOz");

        // Prepositions
        m.insert("in",       "ɪn");
        m.insert("on",       "ˌɔn");
        m.insert("at",       "æt");
        m.insert("to",       "tu");
        m.insert("for",      "fɔɹ");
        m.insert("of",       "ʌv");
        m.insert("with",     "wɪð");
        m.insert("from",     "fɹʌm");
        m.insert("by",       "bI");
        m.insert("about",    "əbˈWt");
        m.insert("into",     "ˈɪntu");
        m.insert("through",  "θɹu");
        m.insert("during",   "dˈʊɹɪŋ");
        m.insert("without",  "wɪðˈWt");
        m.insert("under",    "ˈʌndəɹ");
        m.insert("within",   "wɪðˈɪn");
        m.insert("after",    "ˈæftəɹ");
        m.insert("before",   "bəfˈɔɹ");
        m.insert("between",  "bətwˈin");

        // Common nouns
        m.insert("world",    "wˈɜɹld");
        m.insert("time",     "tˈIm");
        m.insert("day",      "dˈA");
        m.insert("year",     "jˈɪɹ");
        m.insert("way",      "wˈA");
        m.insert("thing",    "θˈɪŋ");
        m.insert("man",      "mˈæn");
        m.insert("woman",    "wˈʊmən");
        m.insert("child",    "ʧˈIld");
        m.insert("people",   "pˈipᵊl");
        m.insert("life",     "lˈIf");
        m.insert("work",     "wˈɜɹk");
        m.insert("place",    "plˈAs");
        m.insert("home",     "hˈOm");
        m.insert("here",     "hˈɪɹ");
        m.insert("there",    "ðɛɹ");
        m.insert("now",      "nˈW");
        m.insert("all",      "ˈɔl");
        m.insert("then",     "ðˈɛn");
        m.insert("well",     "wˈɛl");
        m.insert("even",     "ˈivən");
        m.insert("back",     "bˈæk");
        m.insert("also",     "ˈɔlsO");
        m.insert("because",  "bəkˈʌz");

        // Adjectives
        m.insert("good",     "ɡˈʊd");
        m.insert("great",    "ɡɹˈAt");
        m.insert("new",      "nˈu");
        m.insert("first",    "fˈɜɹst");
        m.insert("last",     "lˈæst");
        m.insert("long",     "lˈɔŋ");
        m.insert("little",   "lˈɪTᵊl");
        m.insert("own",      "ˈOn");
        m.insert("right",    "ɹˈIt");
        m.insert("big",      "bˈɪɡ");
        m.insert("high",     "hˈI");
        m.insert("different","dˈɪfəɹənt");
        m.insert("real",     "ɹˈiᵊl");
        m.insert("small",    "smˈɔl");
        m.insert("large",    "lˈɑɹʤ");
        m.insert("very",     "vˈɛɹi");
        m.insert("more",     "mˈɔɹ");
        m.insert("most",     "mˈOst");
        m.insert("other",    "ˈʌðəɹ");
        m.insert("same",     "sˈAm");
        m.insert("another",  "ənˈʌðəɹ");
        m.insert("always",   "ˈɔlwˌAz");
        m.insert("never",    "nˈɛvəɹ");
        m.insert("often",    "ˈɔfᵊn");
        m.insert("usually",  "jˈuʒəwəli");
        m.insert("too",      "tˈu");

        // Numbers
        m.insert("one",      "wˈʌn");
        m.insert("two",      "tˈu");
        m.insert("three",    "θɹˈi");
        m.insert("four",     "fˈɔɹ");
        m.insert("five",     "fˈIv");
        m.insert("six",      "sˈɪks");
        m.insert("seven",    "sˈɛvən");
        m.insert("eight",    "ˈAt");
        m.insert("nine",     "nˈIn");
        m.insert("ten",      "tˈɛn");

        // Test / pangram words
        m.insert("quick",    "kwˈɪk");
        m.insert("brown",    "bɹˈWn");
        m.insert("fox",      "fˈɑks");
        m.insert("jumps",    "ʤˈʌmps");
        m.insert("over",     "ˈOvəɹ");
        m.insert("lazy",     "lˈAzi");
        m.insert("dog",      "dˈɔɡ");
        m.insert("today",    "tədˈA");
        m.insert("test",     "tˈɛst");
        m.insert("testing",  "tˈɛstɪŋ");
        m.insert("speech",   "spˈiʧ");
        m.insert("synthesis","sˈɪnθəsɪs");

        // TTS-related words
        m.insert("voice",    "vˈYs");
        m.insert("sound",    "sˈWnd");
        m.insert("music",    "mjˈuzɪk");
        m.insert("audio",    "ˈɔdiO");
        m.insert("text",     "tˈɛkst");
        m.insert("word",     "wˈɜɹd");
        m.insert("sentence", "sˈɛntᵊns");

        // Tech words
        m.insert("system",   "sˈɪstəm");
        m.insert("program",  "pɹˈOɡɹˌæm");
        m.insert("computer", "kəmpjˈuTəɹ");
        m.insert("language", "lˈæŋɡwɪʤ");
        m.insert("number",   "nˈʌmbəɹ");
        m.insert("service",  "sˈɜɹvəs");
        m.insert("hand",     "hˈænd");
        m.insert("point",    "pˈYnt");
        m.insert("part",     "pˈɑɹt");
        m.insert("case",     "kˈAs");
        m.insert("week",     "wˈik");
        m.insert("company",  "kˈʌmpᵊni");
        m.insert("fact",     "fˈækt");
        m.insert("group",    "ɡɹˈup");
        m.insert("problem",  "pɹˈɑbləm");

        m
    };
}

/// Misaki-compatible G2P converter
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

    /// Convert text to phoneme token IDs.
    /// Replicates what Python misaki + KModel.forward() does:
    ///   text -> IPA phoneme string -> character-by-character vocab lookup -> i64 IDs
    pub fn text_to_tokens(&self, text: &str) -> Result<Vec<i64>> {
        log::debug!("Converting text to phonemes: '{}'", text);

        let mut tokens = Vec::new();
        let normalized = text.trim().to_lowercase();
        let words = self.tokenize(&normalized);

        for (i, word) in words.iter().enumerate() {
            if i > 0 && !self.is_punctuation(word) {
                if let Some(&t) = self.vocab.get(" ") {
                    tokens.push(t);
                }
            }

            if self.is_punctuation(word) {
                if let Some(&t) = self.vocab.get(word.as_str()) {
                    tokens.push(t);
                }
            } else {
                let word_tokens = self.word_to_tokens(word)?;
                tokens.extend(word_tokens);
            }
        }

        // Add trailing period if sentence doesn't end with punctuation
        if tokens.last().map(|&t| t != 4).unwrap_or(true) {
            if let Some(&period) = self.vocab.get(".") {
                tokens.push(period);
            }
        }

        log::debug!("Generated {} phoneme tokens", tokens.len());
        Ok(tokens)
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_alphabetic() || ch == '\'' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    result.push(current.clone());
                    current.clear();
                }
                if !ch.is_whitespace() {
                    result.push(ch.to_string());
                }
            }
        }
        if !current.is_empty() {
            result.push(current);
        }
        result
    }

    fn is_punctuation(&self, s: &str) -> bool {
        let mut chars = s.chars();
        if let Some(ch) = chars.next() {
            chars.next().is_none() && !ch.is_alphanumeric()
        } else {
            false
        }
    }

    fn word_to_tokens(&self, word: &str) -> Result<Vec<i64>> {
        if let Some(&ipa) = self.dict.get(word) {
            return self.phoneme_string_to_tokens(ipa);
        }
        // Fallback: letter-level approximation
        self.letter_fallback(word)
    }

    /// Convert a Kokoro IPA phoneme string to token IDs.
    /// The string is processed character-by-character (Unicode scalar values),
    /// exactly matching KModel.forward(): vocab.get(p) for p in phonemes
    fn phoneme_string_to_tokens(&self, ipa: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();
        for ch in ipa.chars() {
            let s = ch.to_string();
            if let Some(&id) = self.vocab.get(&s) {
                tokens.push(id);
            } else {
                log::debug!(
                    "phoneme_string_to_tokens: no token for {:?} (U+{:04X})",
                    s,
                    ch as u32
                );
            }
        }
        Ok(tokens)
    }

    /// Letter-level fallback for unknown words.
    /// Uses simple English grapheme-to-phoneme rules with correct Kokoro IPA chars.
    fn letter_fallback(&self, word: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();
        for ch in word.chars() {
            // Map each letter to a plausible Kokoro phoneme character
            let ipa: &str = match ch {
                'a' => "æ", // ɑ/æ — use æ as default
                'e' => "ɛ",
                'i' => "ɪ",
                'o' => "ɔ",
                'u' => "ʌ",
                'b' => "b",
                'c' => "k",
                'd' => "d",
                'f' => "f",
                'g' => "ɡ", // ɡ (U+0261, not ASCII g)
                'h' => "h",
                'j' => "ʤ",
                'k' => "k",
                'l' => "l",
                'm' => "m",
                'n' => "n",
                'p' => "p",
                'q' => "k",
                'r' => "ɹ", // American English r
                's' => "s",
                't' => "t",
                'v' => "v",
                'w' => "w",
                'x' => "ks",
                'y' => "j",
                'z' => "z",
                _ => continue,
            };
            // Handle 'ks' for 'x' (two phonemes)
            for sub_ch in ipa.chars() {
                let s = sub_ch.to_string();
                if let Some(&id) = self.vocab.get(&s) {
                    tokens.push(id);
                }
            }
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_world_correct_tokens() {
        let g2p = MisakiG2P::new().unwrap();
        // "hello" -> "həlˈO": h=50 ə=83 l=54 ˈ=156 O=31
        // " " -> 16
        // "world" -> "wˈɜɹld": w=65 ˈ=156 ɜ=87 ɹ=123 l=54 d=46
        // "." -> 4
        let tokens = g2p.text_to_tokens("hello world").unwrap();
        assert_eq!(
            tokens,
            vec![50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46, 4]
        );
    }

    #[test]
    fn test_hello_tokens() {
        let g2p = MisakiG2P::new().unwrap();
        // "hello" -> "həlˈO" -> [50,83,54,156,31] then "." -> 4
        let tokens = g2p.text_to_tokens("hello").unwrap();
        assert_eq!(tokens, vec![50, 83, 54, 156, 31, 4]);
    }

    #[test]
    fn test_vocab_completeness() {
        let g2p = MisakiG2P::new().unwrap();
        // Key symbols that appear in common English phonemes
        let required = [
            ("ɹ", 123),
            ("ə", 83),
            ("ɪ", 102),
            ("ɛ", 86),
            ("æ", 72),
            ("ɑ", 69),
            ("ɔ", 76),
            ("ʊ", 135),
            ("ʌ", 138),
            ("ɜ", 87),
            ("ð", 81),
            ("θ", 119),
            ("ŋ", 112),
            ("ʃ", 131),
            ("ʒ", 147),
            ("ʧ", 133),
            ("ʤ", 82),
            ("ɡ", 92),
            ("ˈ", 156),
            ("ˌ", 157),
            (" ", 16),
            ("A", 24),
            ("I", 25),
            ("O", 31),
            ("W", 39),
            ("Y", 41),
            ("T", 36),
        ];
        for (sym, expected_id) in required {
            let id = g2p.vocab.get(sym).copied();
            assert_eq!(id, Some(expected_id), "Wrong/missing token for {:?}", sym);
        }
    }

    #[test]
    fn test_all_tokens_in_valid_range() {
        let g2p = MisakiG2P::new().unwrap();
        let sentences = [
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "testing speech synthesis",
            "how are you today",
        ];
        for s in sentences {
            let tokens = g2p.text_to_tokens(s).unwrap();
            assert!(!tokens.is_empty(), "Empty tokens for: {}", s);
            for &t in &tokens {
                assert!(
                    t >= 1 && t <= 177,
                    "Token {} out of range for sentence: {}",
                    t,
                    s
                );
            }
        }
    }

    #[test]
    fn test_letter_fallback_valid_tokens() {
        let g2p = MisakiG2P::new().unwrap();
        // Unknown word - exercises letter fallback
        let tokens = g2p.text_to_tokens("xylophone").unwrap();
        for &t in &tokens {
            assert!(t >= 1 && t <= 177, "Fallback token {} out of range", t);
        }
    }

    // ---- Additional coverage for uncovered paths ----------------------

    #[test]
    fn test_empty_string_produces_period() {
        let g2p = MisakiG2P::new().unwrap();
        // Empty text: no words → tokens is empty before trailing "."
        // The implementation adds a trailing "." when tokens.last() != Some(&4)
        let tokens = g2p.text_to_tokens("").unwrap();
        // period token (4) should be the last (and only) token
        assert_eq!(tokens, vec![4]);
    }

    #[test]
    fn test_sentence_ending_with_period_no_duplicate() {
        let g2p = MisakiG2P::new().unwrap();
        // A sentence that already ends with "." should not get a second period
        let tokens = g2p.text_to_tokens("hello.").unwrap();
        // Last two tokens must not both be 4
        if tokens.len() >= 2 {
            assert!(
                !(tokens[tokens.len() - 1] == 4 && tokens[tokens.len() - 2] == 4),
                "duplicate period found: {:?}",
                tokens
            );
        }
    }

    #[test]
    fn test_punctuation_only() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.text_to_tokens("!").unwrap();
        // "!" → token 5 in vocab; no space before first word
        assert!(tokens.contains(&5));
    }

    #[test]
    fn test_multiple_punctuation() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.text_to_tokens("hello, world!").unwrap();
        assert!(!tokens.is_empty());
        // comma (3) should appear
        assert!(tokens.contains(&3), "comma token missing: {:?}", tokens);
        // exclamation (5) should appear
        assert!(tokens.contains(&5), "! token missing: {:?}", tokens);
    }

    #[test]
    fn test_is_punctuation_via_text_to_tokens() {
        let g2p = MisakiG2P::new().unwrap();
        // Punctuation token must NOT get a space prepended (is_punctuation returns true)
        let tokens_word_punct = g2p.text_to_tokens("hi,").unwrap();
        // If comma got a space before it, space token 16 would appear between "hi" phonemes and ","
        // Just verify no crash and tokens are valid
        for &t in &tokens_word_punct {
            assert!(t >= 1 && t <= 177, "token {} out of range", t);
        }
    }

    #[test]
    fn test_tokenize_apostrophe() {
        let g2p = MisakiG2P::new().unwrap();
        // Apostrophe is kept as part of the word in tokenize()
        let tokens = g2p.text_to_tokens("don't").unwrap();
        assert!(!tokens.is_empty());
        for &t in &tokens {
            assert!(t >= 1 && t <= 177, "token {} out of range", t);
        }
    }

    #[test]
    fn test_phoneme_string_to_tokens_unknown_char_skipped() {
        let g2p = MisakiG2P::new().unwrap();
        // A word whose IPA representation contains a character not in the vocab
        // should silently skip that character. Exercise via a word not in the dict.
        // 'ß' is not in PHONEME_VOCAB; the fallback letter_fallback() also skips it.
        let tokens = g2p.text_to_tokens("ßtest").unwrap();
        // Just must not panic; tokens for 't','e','s','t' should appear
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_letter_fallback_x_produces_ks() {
        let g2p = MisakiG2P::new().unwrap();
        // 'x' maps to "ks" (two chars) in letter_fallback
        // Use a word not in the dictionary that starts with 'x'
        let tokens = g2p.text_to_tokens("xray").unwrap();
        // k token = 53; s token = 61 — both should appear
        assert!(
            tokens.contains(&53) || tokens.contains(&61),
            "expected k(53) or s(61) from 'x' expansion: {:?}",
            tokens
        );
    }

    #[test]
    fn test_number_words_all_produce_tokens() {
        let g2p = MisakiG2P::new().unwrap();
        for word in [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        ] {
            let tokens = g2p.text_to_tokens(word).unwrap();
            assert!(!tokens.is_empty(), "empty tokens for '{}'", word);
            for &t in &tokens {
                assert!(
                    t >= 1 && t <= 177,
                    "token {} out of range for '{}'",
                    t,
                    word
                );
            }
        }
    }

    #[test]
    fn test_space_not_prepended_to_first_word() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.text_to_tokens("hello").unwrap();
        // Space token is 16; it must NOT be the first token
        assert_ne!(
            tokens[0], 16,
            "space should not be first token: {:?}",
            tokens
        );
    }

    #[test]
    fn test_space_inserted_between_words() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.text_to_tokens("hello world").unwrap();
        assert!(
            tokens.contains(&16),
            "space token (16) missing: {:?}",
            tokens
        );
    }

    // ── Line 468: is_punctuation else branch (empty string → false) ──────────

    #[test]
    fn test_is_punctuation_empty_string_returns_false() {
        let g2p = MisakiG2P::new().unwrap();
        // Empty → `if let Some(ch)` never matches → returns false (line 468)
        assert!(!g2p.is_punctuation(""));
    }

    #[test]
    fn test_is_punctuation_single_non_alpha_is_true() {
        let g2p = MisakiG2P::new().unwrap();
        assert!(g2p.is_punctuation("."));
        assert!(g2p.is_punctuation(","));
        assert!(g2p.is_punctuation("!"));
    }

    #[test]
    fn test_is_punctuation_letter_is_false() {
        let g2p = MisakiG2P::new().unwrap();
        assert!(!g2p.is_punctuation("a"));
        assert!(!g2p.is_punctuation("Z"));
    }

    #[test]
    fn test_is_punctuation_multi_char_is_false() {
        let g2p = MisakiG2P::new().unwrap();
        // Two characters: chars.next().is_none() returns false
        assert!(!g2p.is_punctuation(".."));
        assert!(!g2p.is_punctuation("hi"));
    }

    // ── letter_fallback: cover every uncovered match arm (lines 505-528) ──────
    // Each test passes a word that is NOT in PRONUNCIATION_DICT so that
    // letter_fallback is invoked.  The word is crafted so that it exercises
    // only the specific match arms that were reported uncovered.

    /// Lines 505 ('i') and 507 ('u'): vowel arms not hit by existing tests.
    #[test]
    fn test_letter_fallback_arms_i_u() {
        let g2p = MisakiG2P::new().unwrap();
        // "biufz" is not in the dict; letters i and u exercise lines 505, 507.
        let tokens = g2p.letter_fallback("iu").unwrap();
        // ɪ (U+026A) → token 102; ʌ (U+028C) → token 138
        assert!(
            tokens.contains(&102),
            "expected ɪ (102) from 'i': {:?}",
            tokens
        );
        assert!(
            tokens.contains(&138),
            "expected ʌ (138) from 'u': {:?}",
            tokens
        );
    }

    /// Lines 508 ('b') and 509 ('c'): consonant arms.
    #[test]
    fn test_letter_fallback_arms_b_c() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.letter_fallback("bc").unwrap();
        // b → token 44; c → k → token 53
        assert!(
            tokens.contains(&44),
            "expected b (44) from 'b': {:?}",
            tokens
        );
        assert!(
            tokens.contains(&53),
            "expected k (53) from 'c': {:?}",
            tokens
        );
    }

    /// Lines 511 ('f') and 512 ('g'): consonant arms.
    #[test]
    fn test_letter_fallback_arms_f_g() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.letter_fallback("fg").unwrap();
        // f → token 48; ɡ (U+0261) → token 92
        assert!(
            tokens.contains(&48),
            "expected f (48) from 'f': {:?}",
            tokens
        );
        assert!(
            tokens.contains(&92),
            "expected ɡ (92) from 'g': {:?}",
            tokens
        );
    }

    /// Lines 514 ('j') and 515 ('k'): consonant arms.
    #[test]
    fn test_letter_fallback_arms_j_k() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.letter_fallback("jk").unwrap();
        // ʤ (U+02A4) → token 82; k → token 53
        assert!(
            tokens.contains(&82),
            "expected ʤ (82) from 'j': {:?}",
            tokens
        );
        assert!(
            tokens.contains(&53),
            "expected k (53) from 'k': {:?}",
            tokens
        );
    }

    /// Line 517 ('m'): consonant arm.
    #[test]
    fn test_letter_fallback_arm_m() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.letter_fallback("m").unwrap();
        // m → token 55
        assert!(
            tokens.contains(&55),
            "expected m (55) from 'm': {:?}",
            tokens
        );
    }

    /// Line 520 ('q'): consonant arm.
    #[test]
    fn test_letter_fallback_arm_q() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.letter_fallback("q").unwrap();
        // q → k → token 53
        assert!(
            tokens.contains(&53),
            "expected k (53) from 'q': {:?}",
            tokens
        );
    }

    /// Lines 524 ('v') and 525 ('w'): consonant arms.
    #[test]
    fn test_letter_fallback_arms_v_w() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.letter_fallback("vw").unwrap();
        // v → token 64; w → token 65
        assert!(
            tokens.contains(&64),
            "expected v (64) from 'v': {:?}",
            tokens
        );
        assert!(
            tokens.contains(&65),
            "expected w (65) from 'w': {:?}",
            tokens
        );
    }

    /// Line 528 ('z'): consonant arm.
    #[test]
    fn test_letter_fallback_arm_z() {
        let g2p = MisakiG2P::new().unwrap();
        let tokens = g2p.letter_fallback("z").unwrap();
        // z → token 68
        assert!(
            tokens.contains(&68),
            "expected z (68) from 'z': {:?}",
            tokens
        );
    }

    /// Covers all uncovered letter_fallback arms in a single word not in the dict.
    #[test]
    fn test_letter_fallback_all_uncovered_arms_combined() {
        let g2p = MisakiG2P::new().unwrap();
        // Each letter exercises its specific match arm (lines 505-528).
        // The string contains: i u b c f g j k m q v w z
        let tokens = g2p.letter_fallback("iubcfgjkmqvwz").unwrap();
        assert!(
            !tokens.is_empty(),
            "expected non-empty tokens: {:?}",
            tokens
        );
    }

    // ── Line 490: phoneme_string_to_tokens else branch ────────────────────────
    // Reached when a character in the IPA string is NOT in PHONEME_VOCAB.
    // We call the private method directly (accessible within the same module)
    // with a string that contains the Unicode character U+FFFD (replacement
    // character), which is not a Kokoro phoneme and thus not in the vocab.

    /// Line 490: log::debug! fires when an IPA char is absent from the vocab.
    #[test]
    fn test_phoneme_string_to_tokens_missing_char_logs_and_skips() {
        let g2p = MisakiG2P::new().unwrap();
        // U+FFFD (REPLACEMENT CHARACTER) is guaranteed not to be in PHONEME_VOCAB.
        // The else branch on line 490 should execute and the character is skipped.
        let result = g2p.phoneme_string_to_tokens("\u{FFFD}").unwrap();
        // The unknown character is skipped; result should be empty.
        assert!(
            result.is_empty(),
            "expected no tokens for unknown phoneme char: {:?}",
            result
        );
    }

    /// Line 490: multiple unknown chars mixed with known ones — only known ones produce tokens.
    #[test]
    fn test_phoneme_string_to_tokens_mixed_known_unknown() {
        let g2p = MisakiG2P::new().unwrap();
        // 'h' (U+0068) → token 50; U+E000 (private use area) → not in vocab → skip
        let result = g2p.phoneme_string_to_tokens("h\u{E000}").unwrap();
        assert_eq!(result, vec![50], "expected only h(50) token: {:?}", result);
    }
}
