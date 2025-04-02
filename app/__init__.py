from flask import Flask
from .config import initialize_api
from .routes import register_routes

def create_app():
    app = Flask(__name__)
    initialize_api(app)  # Load API keys
    register_routes(app)  # Register all API endpoints
    return app
