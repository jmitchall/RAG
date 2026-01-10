# Install WSL2 in Windows 11 Home and Enable Hyper-V Requirements with Ubuntu

## Prerequisites

Before installing WSL2, ensure your system meets these requirements:

- **Windows 11 Home** (version 21H2 or later)
- **System Architecture**: x64 or ARM64
- **BIOS/UEFI**: Virtualization enabled
- **RAM**: At least 4GB (8GB+ recommended)
- **Storage**: At least 1GB free space for Ubuntu

## Step 1: Enable Required Windows Features

### Method 1: Using PowerShell (Recommended)

Open **PowerShell as Administrator** and run the following commands:

```powershell
# Enable WSL and Virtual Machine Platform features
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Enable Hyper-V (for Windows 11 Home)
dism.exe /online /enable-feature /featurename:HypervisorPlatform /all /norestart
```

### Method 2: Using Windows Features Dialog

1. Press `Win + R`, type `optionalfeatures.exe`, and press Enter
2. Check the following boxes:
   - ✅ **Windows Subsystem for Linux**
   - ✅ **Virtual Machine Platform**
   - ✅ **Hyper-V** (if available)
3. Click **OK** and restart when prompted

## Step 2: Enable Virtualization in BIOS/UEFI

**⚠️ Important**: Restart your computer and enter BIOS/UEFI setup:

1. Restart your computer
2. Press the BIOS key during startup (usually `F2`, `F12`, `Delete`, or `Esc`)
3. Navigate to **Advanced** or **CPU Configuration**
4. Enable these settings:
   - **Intel VT-x** (Intel processors) or **AMD-V** (AMD processors)
   - **Intel VT-d** or **AMD IOMMU** (if available)
   - **Hyper-V** or **Virtualization Technology**
5. Save changes and exit BIOS

## Step 3: Download and Install WSL2 Linux Kernel Update

1. Download the **WSL2 Linux kernel update package** for x64 machines:
   - [WSL2 Linux kernel update package for x64 machines](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)

2. Run the downloaded `.msi` file and follow the installation wizard

3. Restart your computer after installation

## Step 4: Set WSL2 as Default Version

Open **PowerShell as Administrator** and set WSL2 as the default version:

```powershell
wsl --set-default-version 2
```

# WSL CLI Command Reference & Cheatsheet

This comprehensive reference covers all Windows Subsystem for Linux (WSL) commands with descriptions, usage examples, and practical scenarios.

## Installation & Setup Commands

### Install WSL
```powershell
# Install WSL with default Ubuntu distribution
wsl --install

# Install specific Linux distribution
wsl --install -d Ubuntu-22.04
wsl --install -d Debian
wsl --install -d openSUSE-Leap-15.4

# List available distributions for installation
wsl --list --online
wsl -l -o
```

**Description**: Installs WSL and a Linux distribution. The `--install` command enables required Windows features, downloads the Linux kernel, and installs your chosen distribution.

### Update WSL
```powershell
# Update WSL to the latest version
wsl --update

# Update to pre-release version
wsl --update --pre-release

# Check WSL version
wsl --version
```

**Description**: Updates the WSL kernel and components to the latest version from the Microsoft Store.

## Distribution Management

### List Distributions
```powershell
# List installed distributions
wsl --list
wsl -l

# List with detailed information (version, state)
wsl --list --verbose
wsl -l -v

# List only running distributions
wsl --list --running

# List available distributions for download
wsl --list --online
wsl -l -o
```

**Description**: Shows installed, running, or available Linux distributions. Use `-v` for detailed status including WSL version (1 or 2).

### Set Default Distribution
```powershell
# Set default distribution for wsl command
wsl --set-default Ubuntu-22.04
wsl -s Ubuntu-22.04

# Set default WSL version for new distributions
wsl --set-default-version 2
```

**Description**: Changes which distribution launches when you run `wsl` without specifying a distribution name.

### Convert WSL Versions
```powershell
# Convert distribution to WSL 2
wsl --set-version Ubuntu-22.04 2

# Convert distribution to WSL 1
wsl --set-version Ubuntu-22.04 1
```

**Description**: Converts existing distributions between WSL 1 and WSL 2. WSL 2 offers better performance and full system call compatibility.

## Distribution Lifecycle

### Launch Distributions
```powershell
# Launch default distribution
wsl

# Launch specific distribution
wsl -d Ubuntu-22.04
wsl --distribution Ubuntu-22.04

# Launch with specific user
wsl -d Ubuntu-22.04 -u root
wsl --distribution Ubuntu-22.04 --user root

# Launch in specific directory
wsl -d Ubuntu-22.04 --cd /home/username
wsl ~ # Launch in user's home directory
```

**Description**: Starts a Linux shell session. You can specify distribution, user, and starting directory.

### Execute Commands
```powershell
# Run single command in default distribution
wsl ls -la

# Run command in specific distribution
wsl -d Ubuntu-22.04 ls -la

# Run command as specific user
wsl -u root apt update

# Run command and return to Windows
wsl -e bash -c "echo 'Hello from WSL'"
```

**Description**: Executes Linux commands without opening an interactive shell. Useful for scripts and automation.

### Stop & Restart Distributions
```powershell
# Terminate specific distribution
wsl --terminate Ubuntu-22.04
wsl -t Ubuntu-22.04

# Shutdown all running distributions
wsl --shutdown

# Restart WSL service (stops all distributions)
wsl --shutdown
```

**Description**: `--terminate` stops a specific distribution, while `--shutdown` stops all distributions and the WSL service.

## Import & Export

### Export Distributions
```powershell
# Export distribution to tar file
wsl --export Ubuntu-22.04 C:\Backups\ubuntu-backup.tar

# Export with compression
wsl --export Ubuntu-22.04 C:\Backups\ubuntu-backup.tar.gz
```

**Description**: Creates a backup of your Linux distribution as a tar archive. Useful for backups or creating templates.

### Import Distributions
```powershell
# Import distribution from tar file
wsl --import MyUbuntu C:\WSL\MyUbuntu C:\Backups\ubuntu-backup.tar

# Import with specific WSL version
wsl --import MyUbuntu C:\WSL\MyUbuntu C:\Backups\ubuntu-backup.tar --version 2
```

**Description**: Creates a new distribution from a tar archive. First parameter is the new distribution name, second is the installation location, third is the tar file path.

### Unregister Distributions
```powershell
# Remove distribution (deletes all data)
wsl --unregister Ubuntu-22.04

# Warning: This permanently deletes the distribution and all its data
```

**Description**: Completely removes a distribution and all its data. This action cannot be undone.

## Advanced Commands

### File System Access
```powershell
# Mount Windows drive in WSL
wsl --mount \\.\PHYSICALDRIVE1

# Mount with specific file system
wsl --mount \\.\PHYSICALDRIVE1 --type ext4

# Unmount drive
wsl --unmount \\.\PHYSICALDRIVE1
```

**Description**: Mount physical drives or VHD files in WSL. Useful for accessing Linux file systems on external drives.

### Status & Information
```powershell
# Show WSL status and configuration
wsl --status

# Display help information
wsl --help

# Show running distributions and resource usage
wsl --list --running --verbose
```

**Description**: Provides system information, configuration details, and help documentation.

## Configuration & Management

### Using Distribution ID
```powershell
# Launch using distribution ID (useful for scripting)
wsl --distribution-id {439a78b3-9763-443a-9526-5ec80fb9bee7} --cd ~

# Get distribution ID
wsl -l -v  # Shows distribution names and versions
```

**Description**: Each installed distribution has a unique ID that can be used instead of the name. Useful in automated scripts.

## Common Usage Patterns

### Daily Operations
```powershell
# Quick Ubuntu access
wsl

# Run as root for system administration
wsl -u root

# Execute package updates
wsl -u root -- apt update && apt upgrade -y

# Check disk usage
wsl df -h

# Access Windows files from WSL
wsl ls /mnt/c/Users/
```

### Development Workflows
```powershell
# Start development session in project directory
wsl --cd /mnt/c/Users/YourName/Projects/myproject

# Run development server
wsl -d Ubuntu-22.04 python manage.py runserver

# Execute tests
wsl pytest tests/
```

### System Administration
```powershell
# View system information
wsl uname -a
wsl lsb_release -a

# Monitor system resources
wsl htop
wsl free -h

# Network configuration
wsl ip addr show
wsl netstat -tulpn
```

## Troubleshooting Commands

### Diagnostic Commands
```powershell
# Check WSL service status
wsl --status

# Restart WSL completely
wsl --shutdown
# Wait a few seconds, then start again
wsl

# Reset network stack
wsl --shutdown
# Restart Windows network services if needed
```

### Recovery Operations
```powershell
# Force terminate unresponsive distribution
wsl --terminate Ubuntu-22.04

# Reset distribution to clean state (nuclear option)
wsl --unregister Ubuntu-22.04
wsl --install -d Ubuntu-22.04
```

## Quick Reference Table

| Command | Description | Example |
|---------|-------------|---------|
| `wsl --install` | Install WSL and default distribution | `wsl --install -d Ubuntu-22.04` |
| `wsl -l -v` | List distributions with details | Shows name, state, version |
| `wsl -d <name>` | Launch specific distribution | `wsl -d Ubuntu-22.04` |
| `wsl -u <user>` | Launch as specific user | `wsl -u root` |
| `wsl --cd <path>` | Launch in specific directory | `wsl --cd /home/user` |
| `wsl -t <name>` | Terminate distribution | `wsl -t Ubuntu-22.04` |
| `wsl --shutdown` | Stop all distributions | Stops WSL service completely |
| `wsl --export` | Backup distribution | `wsl --export Ubuntu backup.tar` |
| `wsl --import` | Restore distribution | `wsl --import MyUbuntu C:\WSL backup.tar` |
| `wsl --unregister` | Remove distribution | `wsl --unregister Ubuntu-22.04` |
| `wsl --set-version` | Change WSL version | `wsl --set-version Ubuntu 2` |
| `wsl --set-default` | Set default distribution | `wsl --set-default Ubuntu-22.04` |
| `wsl --update` | Update WSL components | Updates kernel and tools |
| `wsl --status` | Show WSL system status | Configuration and version info |
| `wsl --mount` | Mount physical drive | `wsl --mount \\.\PHYSICALDRIVE1` |

## Advanced Tips

### Performance Optimization
```powershell
# Set memory limit in .wslconfig
# File: C:\Users\<username>\.wslconfig
[wsl2]
memory=4GB
processors=2
swap=2GB
```

### Network Configuration
```powershell
# Access WSL from Windows
# WSL services are available at localhost

# Access Windows from WSL
wsl ip route show | grep default  # Shows Windows host IP
```

### Integration with Windows Tools
```powershell
# Open current directory in Windows Explorer
wsl explorer.exe .

# Use Windows tools from WSL
wsl notepad.exe file.txt
wsl code .  # Open VS Code from WSL directory
```

---

## Install Ubuntu from Microsoft Store

### Method 1: Microsoft Store (Recommended)

1. Open **Microsoft Store**
2. Search for **"Ubuntu"**
3. Choose your preferred version:
   - **Ubuntu** (latest LTS version)
   - **Ubuntu 22.04 LTS**
   - **Ubuntu 20.04 LTS**
4. Click **Install**

### Method 2: Command Line Installation

Open **PowerShell as Administrator**:

```powershell
# List available Linux distributions
wsl --list --online

# Install Ubuntu (latest LTS)
wsl --install -d Ubuntu

# Or install specific version
wsl --install -d Ubuntu-22.04
```

## Initialize Ubuntu

1. Launch **Ubuntu** from the Start Menu
2. Wait for the initial setup to complete (this may take several minutes)
3. Create a **UNIX username** (lowercase, no spaces)
4. Create a **password** (you won't see characters as you type)
5. Confirm the password

**Example:**
```bash
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username: ********
New password: [type password]
Retype new password: [type password again]
```

## Update Ubuntu System

Once Ubuntu is initialized, update the system packages:

```bash
# Update package list
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl wget git
```

## Verify WSL2 Installation

### Check WSL Version
```powershell
# List installed distributions and their versions
wsl --list --verbose

# Should show something like:
#   NAME      STATE           VERSION
# * Ubuntu    Running         2
```

### Check Ubuntu System Information
```bash
# In Ubuntu terminal
lsb_release -a
uname -a
```

## Configure WSL2 Settings (Optional)

Create a `.wslconfig` file in your Windows user directory to optimize WSL2:

**File Location**: `C:\Users\[YourUsername]\.wslconfig`

```ini
[wsl2]
# Limits VM memory to use no more than 4 GB
memory=4GB

# Sets the VM to use two virtual processors
processors=2

# Specify a custom Linux kernel to use with your installed distros
# kernel=C:\\temp\\myCustomKernel

# Sets additional kernel parameters
# kernelCommandLine = vsyscall=emulate

# Sets amount of swap storage space to 8GB
swap=8GB

# Sets swapfile path location
# swapfile=C:\\temp\\wsl-swap.vhdx

# Disable page reporting so WSL retains all allocated memory
# pageReporting=false

# Turn off default connection to bind WSL 2 localhost to Windows localhost
# localhostforwarding=true
```

**Apply changes**: Restart WSL by running `wsl --shutdown` in PowerShell, then relaunch Ubuntu.

## Access Ubuntu from Windows

### Launch Ubuntu
- **Start Menu**: Search for "Ubuntu" and click the app
- **Windows Terminal**: Type `wsl` or `ubuntu`
- **PowerShell**: Type `wsl -d Ubuntu`

### Access Windows Files from Ubuntu
```bash
# Windows C: drive is mounted at /mnt/c/
ls /mnt/c/Users/

# Navigate to your Windows user directory
cd /mnt/c/Users/[YourWindowsUsername]/
```

### Access Ubuntu Files from Windows
Open **File Explorer** and navigate to:
```
\\wsl$\Ubuntu\home\[your-ubuntu-username]\
```

## Install Windows Terminal (Recommended)

1. Open **Microsoft Store**
2. Search for **"Windows Terminal"**
3. Click **Install**
4. Set as default terminal in Windows 11 Settings

## Troubleshooting Common Issues

### Error: "WSL 2 requires an update to its kernel component"
- Download and install the WSL2 kernel update from Step 3

### Error: "Please enable the Virtual Machine Platform Windows feature"
- Enable Virtual Machine Platform feature (see Step 1)
- Restart your computer
- Check BIOS virtualization settings (see Step 2)

### Error: "Installation failed with error 0x80073D05"
- Ensure Windows is fully updated
- Run Windows Store Apps troubleshooter
- Reset Microsoft Store cache: `wsreset.exe`

### Ubuntu won't start or shows initialization errors
```powershell
# Reset Ubuntu distribution
wsl --unregister Ubuntu
wsl --install -d Ubuntu
```

### Check WSL Service Status
```powershell
# Restart WSL service
wsl --shutdown
wsl --list --running

# Update WSL
wsl --update
```

## Performance Tips

1. **Store project files in Ubuntu file system** (`/home/username/`) for better performance
2. **Use Windows Terminal** for better experience
3. **Configure Git** in Ubuntu:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

4. **Install VS Code WSL extension** for seamless development

## Next Steps

Now that WSL2 with Ubuntu is installed, you can:

1. **Install Python and UV** (see sections below)
2. **Set up your development environment**
3. **Install vLLM** for AI model serving
4. **Configure Docker** for containerized applications

---

## Docker

    https://docs.astral.sh/uv/guides/integration/docker/