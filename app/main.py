# """
# main.py
# Path: C:\deploy-gate\app\main.py

# PURPOSE:
#   Flask application entry point.
#   Registers all routes and starts the server.

# HOW TO RUN:
#   cd C:\deploy-gate
#   python app\main.py

#   Server starts at: http://localhost:5000
# """

# import os
# import sys

# # Make sure app folder is in path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from flask import Flask
# from routes.score import score_bp
# from routes.dashboard import dashboard_bp
# # ─────────────────────────────────────────────────────────────────────────────
# # CREATE FLASK APP
# # ─────────────────────────────────────────────────────────────────────────────

# def create_app():
#     app = Flask(__name__)

#     # Register blueprints
#     app.register_blueprint(score_bp)
#     app.register_blueprint(dashboard_bp)

#     return app


# app = create_app()


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     port  = int(os.getenv("PORT", 5000))
#     debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

#     print("=" * 60)
#     print("SMART DEPLOY GATE — Starting API Server")
#     print("=" * 60)
#     print(f"  Port  : {port}")
#     print(f"  Debug : {debug}")
#     print(f"\nEndpoints:")
#     print(f"  POST http://localhost:{port}/score    <- Jenkins calls this")
#     print(f"  POST http://localhost:{port}/log      <- outcome logger")
#     print(f"  POST http://localhost:{port}/signup   <- new tenant")
#     print(f"  GET  http://localhost:{port}/health   <- health check")
#     print("=" * 60)

#     app.run(host="0.0.0.0", port=port, debug=debug)

"""
main.py - Flask app entry point with session support
Path: C:\deploy-gate\app\main.py
"""
import os, sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from routes.score     import score_bp
from routes.dashboard import dashboard_bp

def create_app():
    app = Flask(__name__)

    # Secret key for sessions — change this to any random string
    app.secret_key          = os.getenv("SECRET_KEY", "safeship-secret-change-this-2026")
    app.permanent_session_lifetime = timedelta(days=30)

    app.register_blueprint(score_bp)
    app.register_blueprint(dashboard_bp)
    return app

app = create_app()

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG","false").lower() == "true"
    print("="*50)
    print("SAFESHIP — Starting")
    print(f"  http://localhost:{port}")
    print("="*50)
    app.run(host="0.0.0.0", port=port, debug=debug)
    from flask import request, jsonify, render_template, session, redirect
import hashlib

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    data = request.get_json()

    api_key   = data.get('api_key')
    tenant_id = data.get('tenant_id')
    email     = data.get('email')

    if not api_key:
        return jsonify({"success": False, "error": "API key required"}), 400

    # hash the key
    hashed_key = hashlib.sha256(api_key.encode()).hexdigest()

    # TODO: call your dynamo validation
    # tenant = validate_tenant(tenant_id, hashed_key)

    # TEMP DEBUG
    print("LOGIN ATTEMPT:", tenant_id, email)

    # fake success for now
    session['tenant_id'] = tenant_id or "test"

    return jsonify({
        "success": True,
        "tenant_id": tenant_id or "test",
        "redirect": "/dashboard"
    })