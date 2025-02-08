# Use an NVIDIA CUDA base image that includes runtime libraries
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system packages, Python, pip, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Jupyter
RUN pip3 install --upgrade pip && \
    pip3 install notebook jupyterlab

# (Optional) If you want data science libraries, add:
# RUN pip3 install numpy pandas matplotlib scipy scikit-learn

# Expose port 8888 for Jupyter
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
