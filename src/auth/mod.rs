use jsonwebtoken::{encode, decode, Header, Algorithm, EncodingKey, DecodingKey, Validation};
use serde::{Deserialize, Serialize};
use chrono::{Utc, Duration};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: i64,
    pub iat: i64,
    pub username: String,
}

pub struct JwtHandler {
    secret: String,
}

impl JwtHandler {
    pub fn new(secret: &str) -> Self {
        Self {
            secret: secret.to_string(),
        }
    }
    
    pub fn create_token(&self, username: &str) -> Result<String, Box<dyn std::error::Error>> {
        let now = Utc::now();
        let exp = (now + Duration::hours(1)).timestamp();
        
        let claims = Claims {
            sub: username.to_string(),
            exp,
            iat: now.timestamp(),
            username: username.to_string(),
        };
        
        let token = encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(self.secret.as_ref()),
        )?;
        
        Ok(token)
    }
    
    pub fn verify_token(&self, token: &str) -> Result<Claims, Box<dyn std::error::Error>> {
        let data = decode(
            token,
            &DecodingKey::from_secret(self.secret.as_ref()),
            &Validation::new(Algorithm::HS256),
        )?;
        
        Ok(data.claims)
    }
}

pub struct UserStore {
    users: dashmap::DashMap<String, String>,
}

impl UserStore {
    pub fn new() -> Self {
        Self {
            users: dashmap::DashMap::new(),
        }
    }
    
    pub fn add_user(&self, username: &str, password_hash: &str) {
        self.users.insert(username.to_string(), password_hash.to_string());
    }
    
    pub fn verify_user(&self, username: &str, password: &str) -> bool {
        if let Some(entry) = self.users.get(username) {
            bcrypt::verify(password, entry.value()).unwrap_or(false)
        } else {
            false
        }
    }
}

impl Default for UserStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── JwtHandler ──────────────────────────────────────────────────────────

    #[test]
    fn jwt_new_and_create_token_returns_non_empty_string() {
        let handler = JwtHandler::new("test_secret");
        let token = handler.create_token("alice").expect("create_token failed");
        assert!(!token.is_empty(), "token should be a non-empty string");
    }

    #[test]
    fn jwt_verify_valid_token_returns_correct_username() {
        let handler = JwtHandler::new("test_secret");
        let token = handler.create_token("alice").expect("create_token failed");
        let claims = handler.verify_token(&token).expect("verify_token failed");
        assert_eq!(claims.sub, "alice");
        assert_eq!(claims.username, "alice");
    }

    #[test]
    fn jwt_verify_invalid_token_returns_err() {
        let handler = JwtHandler::new("test_secret");
        let result = handler.verify_token("this.is.not.a.valid.token");
        assert!(result.is_err(), "expected Err for invalid token");
    }

    #[test]
    fn jwt_verify_token_with_wrong_secret_returns_err() {
        let handler_a = JwtHandler::new("secret_a");
        let handler_b = JwtHandler::new("secret_b");
        let token = handler_a.create_token("bob").expect("create_token failed");
        let result = handler_b.verify_token(&token);
        assert!(result.is_err(), "expected Err when verifying with wrong secret");
    }

    // ── UserStore ────────────────────────────────────────────────────────────

    #[test]
    fn user_store_new_and_default_are_equivalent() {
        let _s1 = UserStore::new();
        let _s2 = UserStore::default();
        // Both must construct without panic; no further assertion needed.
    }

    #[test]
    fn user_store_add_and_verify_correct_password() {
        let store = UserStore::new();
        let hash = bcrypt::hash("secret123", 4).expect("hash failed");
        store.add_user("carol", &hash);
        assert!(store.verify_user("carol", "secret123"), "correct password should return true");
    }

    #[test]
    fn user_store_verify_wrong_password_returns_false() {
        let store = UserStore::new();
        let hash = bcrypt::hash("correct_password", 4).expect("hash failed");
        store.add_user("dave", &hash);
        assert!(!store.verify_user("dave", "wrong_password"), "wrong password should return false");
    }

    #[test]
    fn user_store_verify_unknown_username_returns_false() {
        let store = UserStore::new();
        assert!(!store.verify_user("nonexistent_user", "any_password"), "unknown user should return false");
    }
}
