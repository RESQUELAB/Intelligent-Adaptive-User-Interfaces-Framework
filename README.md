# RL4UI: Intelligent Adaptive User Interfaces Framework

This repository orchestrates an adaptive user interface (UI) experimentation platform using reinforcement learning (RL) and human feedback. It integrates a Django backend, RL teacher servers, a video server, and an Electron-based desktop client for experiment management and data collection.

## Repository Structure

To address the need for clear separation between the reusable tool and sample implementations, this repository is strictly divided into core framework components and illustrative examples:

- **`/core-environment/`**: Contains the complete RL4UI software environment. This is divided into:
  - The reusable server-side RLHF backend (Adaptation Decision Engine, Django API, Video Server, etc.) and Docker configurations.
- **`/examples/`**: Contains simple illustrative scripts and sample Target Applications demonstrating how a third-party UI connects to the backend to receive adaptations.

---

## Features

- **Modular RL Servers**: Train adaptive agents using Reinforcement Learning from Human Feedback (RLHF).
- **Dynamic Action Spaces**: The framework automatically infers the RL action space directly from the client's UI capabilities.
- **Django Backend**: Manages user data, experiment configuration, and serves as the main API.
- **Video Server**: Handles video storage and streaming for explicit UI adaptation comparisons.
- **Dockerized Deployment**: All core services are containerized for easy setup and reproducibility.

---

## Prerequisites

- **Docker / Docker Desktop** (with Docker Compose; on Windows this also requires WSL2)
- **Python 3.x** for helper scripts

On a clean Windows machine, run `core-environment/setup_docker.bat` to automatically install Docker Desktop, WSL2, and Ubuntu.

---

## Demonstration Video

A video showing the installation steps and the full workflow (session start, comparison videos, "Experiments" list population, and automated agent training progress in the RL Teacher interface) is available at:

[https://drive.google.com/drive/folders/1O6zyvvOrLdvXQu2KHrQNAbxg2SQWHul4?usp=sharing](https://drive.google.com/drive/folders/1O6zyvvOrLdvXQu2KHrQNAbxg2SQWHul4?usp=sharing)

---

## Getting Started

### 1. Launch All Services (Docker + Orchestrator)

```sh
cd core-environment
run_all.bat
```

This script automatically:
- Creates `.env` from `.env.example` and detects your server IP
- Downloads the Orchestrator (Electron app)
- Downloads the pre-trained clips (~468 MB) and pre-generated comparison videos (~19 MB) from the release assets
- Builds and starts all Docker containers (`django_app`, RL servers, video server, DB)
- Applies database migrations on first startup
- Launches the Electron desktop app

### 2. Install the Client App

In a **second terminal**:

```sh
cd examples
get_client.bat
```

This downloads the client from release `adaptive_app_v1.0.2`, extracts it into `client_app`, and writes the detected server IP into `client_app/resources/app/config.json` automatically.

### 3. Run an Experiment

1. Open the `adaptiveapp.exe` inside `examples/client_app`.
2. Register a new user account and log in.
3. On the selection screen, click the first card to start your session. This launches the comparison videos and creates your experiment trees.
4. Open the **RL Teacher** interface at [http://localhost:8000](http://localhost:8000), log in with the same account, and your experiments will now appear under **Experiments**.
5. Answer the comparison questions; once enough feedback is gathered, the agent training starts automatically.

> Note: the "Experiments" list is populated once a session is started (comparison trees are created on first login from the client). A freshly registered account with no session started shows an empty list.

### 4. Access the System

- **Orchestrator App**: The desktop client will launch automatically (or run `core-environment/electron_app/adaptiveuiserver.exe`).
- **Django Backend / RL Teacher**: Accessible at [http://localhost:8000](http://localhost:8000)
- **Video Server**: Accessible at [http://localhost:5000](http://localhost:5000)
- **RL Teacher Servers**: Exposed on ports 9998 and 9997.

---

## Integration and Customization

RL4UI is designed to be highly modular so it can be integrated into existing software systems without needing to rewrite the reinforcement learning backend.

### 1. Connecting Your Client Application

To adapt your own UI, you need to establish a WebSocket connection to the server. The RL4UI backend automatically infers the available UI features and generates the RL Action Space dynamically based on the payload you send during authentication.

Your client must send its current state (`mutations`) and a dictionary of all possible adaptation states (`all_mutations`). Below is an illustrative JavaScript snippet demonstrating this integration:

```javascript
// 1. Define the current state of the UI
const currentMutations = {
    theme: 'light',
    language: 'en',
    display: 'list',
    font_size: 'default'
};

// 2. Define all possible adaptation variants for this specific UI
const allPossibleMutations = {
    theme: ['light', 'dark'],
    display: ['list', 'grid2', 'grid3', 'grid4'],
    font_size: ['small', 'default', 'medium', 'big']
};

// 3. Connect to the RL4UI WebSocket Server
const socket = io(`http://${HOST}:${PORT}`, {
    reconnection: false,
    auth: {
        sessionID: loginInfo.sessionID,
        username: loginInfo.username,
        page: new URL(document.location).pathname,
        mutations: currentMutations,
        all_mutations: allPossibleMutations
    },
    cors: { origin: "*" }
});
```

### 2. Customizing the Observation Space (Context)

If your target application tracks specific contextual variables (e.g., User Age, Device Type, Environmental Location), you must configure the environment's Observation Space to accept these variables.

To do this, edit the `config.json` file located at: `core-environment/rl-teacher-ui-adapt/ui_adapt/ui_adapt/config.json`

Example configuration:
```json
{
    "USER": {
        "AGE": ["young", "adult", "senior"]
    },
    "PLATFORM": {
        "DEVICE": ["mobile", "desktop", "tablet"]
    },
    "ENVIRONMENT": {
        "LOCATION": ["home", "work", "public"]
    }
}
```

## Logs

- All user interactions and agent events are logged in CSV files under `logs/` and `electron_app/resources/app/logs/`.

---

## Client-Side Example (Testing)

A ready-to-use Windows client binary is available for demonstration and quick testing.  
Download the latest release from [Adaptive-app v1.0.2](https://github.com/RESQUELAB/Adaptive-app/releases/tag/adaptive_app_v1.0.2).

A batch file is included at `examples/get_client.bat` to automatically download, extract, and configure the client into a `client_app` folder (the server IP is written into `client_app/resources/app/config.json` automatically during setup).

If you need to change the server address manually, edit `config.json`:

```json
{
  "TARGET_SERVER": "127.0.0.1"
}
```

---

## License

See [LICENSE](LICENSE) for details.

---

