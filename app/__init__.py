from flask import Flask
from dotenv import load_dotenv

app = Flask(__name__, template_folder='templates')

load_dotenv()

app.config.from_pyfile('config.py')

@app.errorhandler(404)
def page_not_found(error):
    return "<h1>Looks like you're lost</h1>", 404

@app.errorhandler(500)
def internal_server_error(error):
    return '<h1>Internal Server Error</h1>', 500

from app import routes