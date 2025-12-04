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
