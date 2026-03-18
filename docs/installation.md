# Manual Installation Guide

This guide provides step-by-step instructions for manually installing the LeHome Challenge environment.

## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- GPU driver and CUDA supporting IsaacSim5.1.0.

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

### 2. Install Dependencies with uv

```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

### 3. Clone and Configure IsaacLab

```bash
git clone -b lehome-cloth-mimic-compat https://github.com/alex-luci/IsaacLab.git third_party/IsaacLab
```

For the MimicGen annotation / generation workflow in this repository, use the
`alex-luci/IsaacLab` `lehome-cloth-mimic-compat` branch. It includes the
cloth-settle and recorder compatibility changes expected by the current scripts.

### 4. Install IsaacLab

Activate the virtual environment and install IsaacLab:

```bash
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

### 5. Install LeHome Package

Finally, install the LeHome package in development mode:

```bash
uv pip install -e ./source/lehome
```

If your `.venv` was previously pointing at IsaacLab packages from another checkout, re-running the two commands above after
cloning `third_party/IsaacLab` will repair the editable installs to use this repository's local paths.

---
###
If you are using a server, please download the system dependencies.

```bash
    #step 1
    apt update
    apt install -y \
    libglu1-mesa \
    libgl1 \
    libegl1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxext6 \
    libx11-6
    #step 2
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
```


## Next Steps

Now that you have installed the environment, you can:

- [Prepare Assets and Data](datasets.md)
- [Start Training](training.md)
- [Evaluate Policies](policy_eval.md)
- [Back to README](../README.md)
