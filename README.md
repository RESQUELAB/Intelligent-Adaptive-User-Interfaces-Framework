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

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- (Windows) [Python 3.x](https://www.python.org/) for helper scripts

---

## Getting Started

### 1. Configure Environment

Copy the example environment file and edit as needed:

```sh
cd core-environment
cp .env.example .env
# Edit .env with your settings (DB credentials, email, etc.)
```

### 2. Launch All Services

On Windows, use the provided batch script:

```sh
cd core-environment
run_all.bat
```

This will:
- Set up environment variables
- Copy `.env` to the Orchestrator (Electron app)
- Build and start all Docker containers (`django_app`, RL servers, video server, DB)
- Launch the Electron desktop app

Alternatively, you can run the services manually:

```sh
cd src
docker-compose build
docker-compose up -d
# Then start the Electron app manually from electron_app/
```

### 3. Access the System

- **Orchestrator App**: The desktop client will launch automatically (or run `core-environment/electron_app/adaptiveuiserver.exe`).
- **Django Backend**: Accessible at [http://localhost:8000](http://localhost:8000)
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
Download the latest release from [Adaptive-app v1.0.1](https://github.com/RESQUELAB/Adaptive-app/releases/tag/adaptive_app_v1.0.1).

A batch file is included at `examples/get_client.bat` to automatically download and extract the client into a `client_app` folder.  
**Before running the app, configure the SERVER address in `client_app/resources/app/config.json`.**

+ `config.json` example:

```json
{
  "TARGET_SERVER": "127.0.0.1"
}
```

---

## License

See [LICENSE](LICENSE) for details.

---

