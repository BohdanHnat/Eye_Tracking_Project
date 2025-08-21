from datetime import timedelta
import secrets
import os

SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(64)

DEBUG = os.getenv("DEBUG", False).lower() in ("true", "1", "yes")

PORT = int(os.getenv("PORT", 5000))

JWT_ACCESS_TOKEN_EXPIRES = timedelta(
    seconds=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 3600))
)

WTF_CSRF_ENABLED = os.getenv("WTF_CSRF_ENABLED", True).lower() in ("true", "1", "yes")