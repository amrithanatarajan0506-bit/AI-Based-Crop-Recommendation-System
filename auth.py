# ============================================================
#  auth.py
#  Handles user Registration, Login, and storage
#  Used by app.py — do not run this file directly
# ============================================================

import json
import os
import hashlib
from datetime import datetime

USER_DB_FILE = "users.json"


def _hash_password(password):
    """Convert plain password into a secure SHA-256 hash before storing."""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users():
    """Load all registered users from the JSON file. Creates file if missing."""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)
        return {}

    with open(USER_DB_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_users(users):
    """Write the updated users dictionary back to the JSON file."""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username, password, full_name, location):
    """
    Register a new farmer account.

    Returns:
        (True, "success message")  on success
        (False, "error message")   on failure
    """
    username = username.strip().lower()

    if not username or not password:
        return False, "Username and password cannot be empty."

    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    users = _load_users()

    if username in users:
        return False, "This username is already taken. Please choose another."

    users[username] = {
        "password":   _hash_password(password),
        "full_name":  full_name.strip(),
        "location":   location.strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history":    []   # stores past crop predictions for this user
    }

    _save_users(users)
    return True, "Account created successfully! Please log in."


def login_user(username, password):
    """
    Verify login credentials.

    Returns:
        (True, user_data_dict)  on success
        (False, "error message") on failure
    """
    username = username.strip().lower()
    users = _load_users()

    if username not in users:
        return False, "No account found with this username. Please register first."

    stored_hash = users[username]["password"]
    if stored_hash != _hash_password(password):
        return False, "Incorrect password. Please try again."

    return True, users[username]


def save_prediction_to_history(username, prediction_record):
    """
    Append a new crop prediction result to the user's history.
    prediction_record is a dict like:
        {"timestamp": ..., "crop": ..., "confidence": ..., "inputs": {...}}
    """
    username = username.strip().lower()
    users = _load_users()

    if username in users:
        users[username]["history"].append(prediction_record)
        # Keep only the latest 20 records to keep file small
        users[username]["history"] = users[username]["history"][-20:]
        _save_users(users)


def get_user_history(username):
    """Return the list of past predictions for a user."""
    username = username.strip().lower()
    users = _load_users()
    if username in users:
        return users[username].get("history", [])
    return []
