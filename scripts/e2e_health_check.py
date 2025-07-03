#!/usr/bin/env python3
"""
E2E Infra Health Check Script
Checks if all required services for E2E tests are up and healthy.
"""
import sys
import socket
import requests

SERVICES = [
    {"name": "Streamlit App", "url": "http://localhost:8501", "type": "http"},
    {"name": "Ollama API", "url": "http://localhost:11434/api/tags", "type": "http"},
    {"name": "ChromaDB", "host": "localhost", "port": 8000, "type": "tcp"},
    {"name": "Redis", "host": "localhost", "port": 6379, "type": "tcp"},
]

OLLAMA_MODELS = ["mistral", "nomic-embed-text"]

failures = []

def check_http(url, name):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            failures.append(f"{name} HTTP {url} returned status {resp.status_code}")
            return False
        return True
    except Exception as e:
        failures.append(f"{name} HTTP {url} failed: {e}")
        return False

def check_tcp(host, port, name):
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except Exception as e:
        failures.append(f"{name} TCP {host}:{port} failed: {e}")
        return False

def check_ollama_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        tag_names = [m.get("name", "") for m in tags]
        missing = []
        for m in OLLAMA_MODELS:
            if m not in tag_names and f"{m}:latest" not in tag_names:
                missing.append(m)
        if missing:
            failures.append(f"Ollama missing required models: {', '.join(missing)}")
            return False
        return True
    except Exception as e:
        failures.append(f"Ollama model check failed: {e}")
        return False

def main():
    print("\n\U0001F50D E2E Infra Health Check\n--------------------------")
    all_ok = True
    for svc in SERVICES:
        if svc["type"] == "http":
            ok = check_http(svc["url"], svc["name"])
        elif svc["type"] == "tcp":
            ok = check_tcp(svc["host"], svc["port"], svc["name"])
        else:
            continue
        print(f"{svc['name']}: {'✅' if ok else '❌'}")
        all_ok = all_ok and ok
    # Ollama model check
    ok = check_ollama_models()
    print(f"Ollama Models: {'✅' if ok else '❌'}")
    all_ok = all_ok and ok
    if failures:
        print("\n\U0001F6A8 Failures:")
        for f in failures:
            print(f"  - {f}")
    print("\nSummary: " + ("ALL SERVICES HEALTHY \U0001F7E2" if all_ok else "SOME SERVICES UNHEALTHY \U0001F534"))
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main() 