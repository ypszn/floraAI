from flask import Flask
from app.routes import api
from app.utils import setup_logging

def create_app():
    # Set up logging
    setup_logging()

    app = Flask(__name__)

    # Register API routes
    app.register_blueprint(api, url_prefix="/api")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
