# RL_UI_Adapt_orchestrator

This repository orchestrates an adaptive user interface (UI) experimentation platform using reinforcement learning (RL) and human feedback. It integrates a Django backend, RL teacher servers, a video server, and an Electron-based desktop client for experiment management and data collection.

## Features

- **Electron App**: Desktop client for experiment setup, monitoring, and log visualization.
- **Django Backend**: Manages user data, experiment configuration, and serves as the main API.
- **RL Teacher Servers**: Modular RL servers for adaptive agent training using human feedback.
- **Video Server**: Handles video storage and streaming for UI adaptation experiments.
- **PostgreSQL Database**: Central data storage for experiments and user data.
- **Dockerized Deployment**: All services are containerized for easy setup and reproducibility.

## Quick Start


### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- (Windows) [Python 3.x](https://www.python.org/) for helper scripts

### 1. Configure Environment

Copy the example environment file and edit as needed:

```sh
cp src/.env.example src/.env
# Edit src/.env with your settings (DB credentials, email, etc.)
```

### 2. Launch All Services

On Windows, use the provided batch script:

```bat
cd src
run_all.bat
```

This will:
- Set up environment variables
- Copy `.env` to the Electron app
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

- **Electron App**: The desktop client will launch automatically (or run `electron_app/adaptiveuiserver.exe`).
- **Django Backend**: Accessible at [http://localhost:8000](http://localhost:8000)
- **Video Server**: Accessible at [http://localhost:5000](http://localhost:5000)
- **RL Teacher Servers**: Exposed on ports 9998 and 9997.

## Logs

- All user interactions and agent events are logged in CSV files under `logs/` and `electron_app/resources/app/logs/`.

## Customization

- **Experiment Configuration**: Modify or add RL environments and UI adaptation logic in `rl-teacher-ui-adapt/`.
- **Email Notifications**: Configure email settings in `.env` for agent status notifications.

## License

See [LICENSE](LICENSE) for details.

---

**Note:** For advanced usage, development, or troubleshooting, refer to the Dockerfiles and scripts in each service directory.