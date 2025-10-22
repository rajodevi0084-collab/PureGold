# Apple Silicon Setup Guide

Follow these steps to bootstrap a fresh Apple Silicon development machine.

## 1. Install Xcode Command Line Tools
Run the following command and follow the prompts:

```bash
xcode-select --install
```

## 2. Install Python 3.11 universal2 from python.org
1. Download the latest 3.11.x universal2 installer from [python.org/downloads](https://www.python.org/downloads/macos/).
2. Run the `.pkg` installer and accept the defaults.
3. After installation, confirm the interpreter:
   ```bash
   /usr/local/bin/python3 --version
   ```

## 3. Install `uv`
Use the official installer script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This places the `uv` binary in `~/.local/bin` by default. Ensure that directory is in your `PATH`.

## 4. Bootstrap the project
1. Clone the repository.
2. From the project root, run:
   ```bash
   make bootstrap
   ```
   This runs `uv sync` with the locked dependencies.

## 5. Run lint checks
After bootstrapping, verify the environment by running:

```bash
make lint
```

You should see `ruff` check the repository using the dependencies recorded in `uv.lock`.
