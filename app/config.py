import os
from datetime import timedelta

SECRET_KEY = os.getenv('SECRET_KEY', 'secret')
DEBUG = os.getenv('DEBUG', False)
PORT = os.getenv('PORT', 100)
JWT_ACCESS_TOKEN_EXPIRES = os.getenv('JWT_ACCESS_TOKEN_EXPIRES', timedelta(hours=1))
WTF_CSRF_ENABLED = os.getenv('WTF_CSRF_ENABLED', False)