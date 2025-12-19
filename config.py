from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: str
    
    # Collection Names
    employee_collection: str = "employee_requests"
    policies_collection: str = "company_policies"
    cases_collection: str = "logged_cases"
    
    # Gemini API
    gemini_api_key: str
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug_mode: bool = True
    
    # CORS
    allowed_origins: str = "http://localhost:3000"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_allowed_origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()