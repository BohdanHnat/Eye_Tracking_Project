from datetime import timedelta
import secrets
import os

SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(64)

DEBUG = os.getenv('DEBUG', False)

PORT = int(os.getenv("PORT", 5000))

JWT_ACCESS_TOKEN_EXPIRES = os.getenv('JWT_ACCESS_TOKEN_EXPIRES', timedelta(hours=1))

WTF_CSRF_ENABLED = os.getenv("WTF_CSRF_ENABLED", True)