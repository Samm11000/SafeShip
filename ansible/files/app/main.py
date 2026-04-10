"""
main.py
Path: C:\deploy-gate\app\main.py

PURPOSE:
  Flask application entry point.
  Registers all routes and starts the server.

HOW TO RUN:
  cd C:\deploy-gate
  python app\main.py

  Server starts at: http://localhost:5000
"""

import os
import sys

# Make sure app folder is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from routes.score import score_bp
from routes.dashboard import dashboard_bp
# ─────────────────────────────────────────────────────────────────────────────
# CREATE FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

def create_app():
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(score_bp)
    app.register_blueprint(dashboard_bp)

    return app


app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print("=" * 60)
    print("SMART DEPLOY GATE — Starting API Server")
    print("=" * 60)
    print(f"  Port  : {port}")
    print(f"  Debug : {debug}")
    print(f"\nEndpoints:")
    print(f"  POST http://localhost:{port}/score    <- Jenkins calls this")
    print(f"  POST http://localhost:{port}/log      <- outcome logger")
    print(f"  POST http://localhost:{port}/signup   <- new tenant")
    print(f"  GET  http://localhost:{port}/health   <- health check")
    print("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=debug)