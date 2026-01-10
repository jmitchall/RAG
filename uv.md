# Install UV
Use irm to download the script and execute it with iex:

    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
Changing the execution policy allows running a script from the internet.

Request a specific version by including it in the URL:

    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.8.17/install.ps1 | iex"


# Enable long paths in Windows 10, version 1607, and later

Starting in Windows 10, version 1607, MAX_PATH limitations have been removed from many common Win32 file and directory functions. However, your app must opt-in to the new behavior.

To enable the new long path behavior per application, two conditions must be met. A registry value must be set, and the application manifest must include the longPathAware element.

## Registry setting to enable long paths

```
Important

Understand that enabling this registry setting will only affect applications that have been modified to take advantage of the new feature. Developers must declare their apps to be long path aware, as outlined in the application manifest settings below. This isn't a change that will affect all applications.
```
The registry value ` HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled (Type: REG_DWORD)` must exist and be set to 1. The registry value will be cached by the system (per process) after the first call to an affected Win32 file or directory function (see below for the list of functions). The registry value will not be reloaded during the lifetime of the process. In order for all apps on the system to recognize the value, a reboot might be required because some processes may have started before the key was set.

You can also copy this code to a `.reg` file which can set this for you, or use the PowerShell command from a terminal window with elevated privileges:

**Registry (.reg) File**
```
 Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem]
"LongPathsEnabled"=dword:00000001
```

**Powershell (Admin -Mode)**
```
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```



# UV Overview

**UV** is a modern, fast Python package manager and project manager built in Rust. Created by Astral (makers of Ruff), UV is designed to be a drop-in replacement for pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.

## Key Features:
- **Blazing Fast**: 10-100x faster than pip for package installation
- **Unified Tool**: Replaces multiple Python tools with a single binary
- **pip Compatible**: Drop-in replacement for pip workflows
- **Python Version Management**: Install and manage Python versions
- **Project Management**: Create, manage, and publish Python projects
- **Lock Files**: Reproducible dependency resolution
- **Virtual Environment Management**: Seamless venv creation and management

# UV CLI Cheat Sheet

## Installation & Updates

### Install UV (Windows PowerShell)
```powershell
# Install latest version
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install specific version
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.8.17/install.ps1 | iex"
```

### Install UV (Linux/macOS)
```bash
# Install latest version
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with pip
pip install uv
```

### Update UV
```bash
uv self update
```

## Python Version Management

### Install Python versions
```bash
# Install latest Python
uv python install

# Install specific Python version
uv python install 3.12
uv python install 3.11.9

# List available Python versions
uv python list --all-versions

# List installed Python versions
uv python list
```

### Use specific Python version
```bash
# Use for current project
uv python pin 3.12

# Check current Python version
uv python find
```

## Virtual Environment Management

### Create virtual environments
```bash
# Create venv with default Python
uv venv

# Create venv with specific Python version
uv venv --python 3.12
uv venv --python python3.11

# Create venv with seed packages (pip, setuptools, wheel)
uv venv --seed

# Create venv in specific directory
uv venv .venv
uv venv my-project-env
```

### Activate virtual environments
```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

## Package Installation & Management

### Install packages (pip replacement)
```bash
# Install single package
uv pip install requests

# Install multiple packages
uv pip install requests numpy pandas

# Install with version constraints
uv pip install "django>=4.0,<5.0"
uv pip install numpy==1.24.3

# Install from requirements.txt
uv pip install -r requirements.txt

# Install editable package
uv pip install -e .
```

### Install packages in virtual environment
```bash
# Auto-create venv and install
uv add requests
uv add "fastapi[all]"

# Add development dependencies
uv add --dev pytest black ruff

# Add optional dependencies
uv add --optional dev pytest coverage
```

### Upgrade and uninstall packages
```bash
# Upgrade package
uv pip install --upgrade requests

# Upgrade all packages
uv pip install --upgrade-package all

# Uninstall package
uv pip uninstall requests

# Remove from project
uv remove requests
```

## Project Management

### Initialize new project
```bash
# Create new project
uv init my-project
cd my-project

# Initialize in existing directory
uv init

# Create with specific Python version
uv init --python 3.12 my-project
```

### Project structure commands
```bash
# Add dependencies to project
uv add requests fastapi

# Add development dependencies
uv add --dev pytest black ruff mypy

# Sync dependencies (install from lock file)
uv sync

# Sync only production dependencies
uv sync --no-dev

# Update dependencies
uv lock --upgrade
```

### Run commands in project environment
```bash
# Run Python script
uv run python script.py

# Run module
uv run -m pytest

# Run arbitrary commands
uv run python -c "import requests; print(requests.__version__)"

# Install and run tool temporarily
uv run --with requests -- python -c "import requests"
```

## Dependency Resolution & Lock Files

### Generate lock files
```bash
# Generate uv.lock file
uv lock

# Update specific package
uv lock --upgrade-package requests

# Upgrade all packages
uv lock --upgrade
```

### Export requirements
```bash
# Export to requirements.txt
uv export --output-file requirements.txt

# Export without hashes
uv export --no-hashes --output-file requirements.txt

# Export only production dependencies
uv export --no-dev --output-file requirements.txt
```

## Tool Management (pipx replacement)

### Install global tools
```bash
# Install global tool
uv tool install black
uv tool install ruff

# Install specific version
uv tool install "black==23.0.0"

# List installed tools
uv tool list

# Upgrade tool
uv tool upgrade black

# Uninstall tool
uv tool uninstall black
```

### Run tools without installing
```bash
# Run tool temporarily
uv tool run black --check .
uv tool run ruff check .

# Run with specific version
uv tool run --from "black==23.0.0" black --check .
```

## Build & Publish

### Build packages
```bash
# Build wheel and source distribution
uv build

# Build only wheel
uv build --wheel

# Build only source distribution
uv build --sdist
```

### Publish packages
```bash
# Publish to PyPI
uv publish

# Publish to test PyPI
uv publish --repository testpypi

# Publish with token
uv publish --token $PYPI_TOKEN
```

## Cache Management

### Cache operations
```bash
# Show cache directory
uv cache dir

# Clean cache
uv cache clean

# Show cache size
uv cache size

# Clean specific package from cache
uv cache clean requests
```

## Configuration & Environment

### Show environment info
```bash
# Show UV version
uv version

# Show Python installations
uv python list

# Show project info
uv tree

# Show installed packages
uv pip list

# Show package info
uv pip show requests
```

### Configuration
```bash
# Set global config
uv config set global.index-url https://pypi.org/simple/

# Show current config
uv config show

# Reset config
uv config unset global.index-url
```

## Performance & Advanced Usage

### Fast package installation
```bash
# Install with multiple workers
uv pip install --resolution highest requests

# Install from local wheel
uv pip install ./dist/mypackage-1.0.0-py3-none-any.whl

# Install from git repository
uv pip install git+https://github.com/user/repo.git

# Install from specific branch/tag
uv pip install git+https://github.com/user/repo.git@main
```

### Reproducible installs
```bash
# Generate exact lockfile
uv pip freeze > requirements-lock.txt

# Install from exact lockfile
uv pip install -r requirements-lock.txt

# Compile requirements with constraints
uv pip compile requirements.in --output-file requirements.txt
```

## Common Workflows

### Start new Python project
```bash
uv init my-project
cd my-project
uv add requests fastapi
uv add --dev pytest black ruff
uv run python main.py
```

### Replace pip in existing project
```bash
# Instead of: pip install -r requirements.txt
uv pip install -r requirements.txt

# Instead of: pip install package
uv pip install package

# Instead of: pip freeze > requirements.txt
uv pip freeze > requirements.txt
```

### Migrate from Poetry/Pipenv
```bash
# Convert pyproject.toml to uv format
uv init

# Install dependencies
uv sync

# Add new dependencies
uv add requests
```

## Quick Reference

| Command | Description | Example |
|---------|-------------|---------|
| `uv init` | Initialize new project | `uv init my-project` |
| `uv add` | Add dependency to project | `uv add requests` |
| `uv remove` | Remove dependency | `uv remove requests` |
| `uv sync` | Install project dependencies | `uv sync` |
| `uv run` | Run command in project env | `uv run python script.py` |
| `uv venv` | Create virtual environment | `uv venv --python 3.12` |
| `uv pip install` | Install packages (pip style) | `uv pip install requests` |
| `uv tool install` | Install global tool | `uv tool install black` |
| `uv python install` | Install Python version | `uv python install 3.12` |
| `uv lock` | Generate lock file | `uv lock --upgrade` |
| `uv build` | Build package | `uv build` |
| `uv publish` | Publish to PyPI | `uv publish` |

## Performance Tips

1. **Memory Optimization**:
   - Use `--dtype float16` to reduce memory usage
   - Set `--gpu-memory-utilization 0.9` for maximum efficiency
   - Enable `--kv-cache-dtype fp8` for additional memory savings

2. **Context Length**:
   - Default: 32768 tokens
   - Reduce with `--max-model-len 4096` if memory constrained
   - Increase for longer conversations if you have sufficient VRAM

3. **Batch Processing**:
   - Mixtral handles multiple requests efficiently
   - Use appropriate `max_tokens` to prevent excessive generation

4. **Model Loading**:
   - First run downloads ~46GB model weights
   - Subsequent runs load from cache (~2-3 minutes)
   - Consider using `--download-dir` to specify cache location

















