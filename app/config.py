import secrets
import os

SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(64)

DEBUG = os.getenv('DEBUG', False)

PORT = os.getenv("PORT", 5000)