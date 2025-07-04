import socket
from pathlib import Path

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def update_env(env_path, key, value):
    path = Path(env_path)
    lines = path.read_text().splitlines() if path.exists() else []
    updated = False

    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")

    path.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    ip = get_local_ip()
    update_env(".env", "VIDEO_SERVER_HOST", ip)
    print(f"✅ VIDEO_SERVER_HOST={ip} actualizado en .env")
