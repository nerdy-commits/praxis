FROM python:3.10-slim

# Install minimal system dependencies (libgl1 is required by opencv)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000 to comply with Hugging Face policies
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Switch to the non-root user
USER user

# Set the working directory
WORKDIR $HOME/app

# Install numpy<2 FIRST — torch 2.1.0 CPU is compiled against numpy 1.x.
# If numpy 2.x gets installed, torch's C extensions crash at import time.
RUN pip install --no-cache-dir --user "numpy<2"

# Install CPU-only PyTorch and torchvision FIRST (avoids 2GB+ CUDA download)
RUN pip install --no-cache-dir --user \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu


# Copy the HF-specific requirements file (no torch, no strict pinning)
COPY --chown=user requirements_hf.txt $HOME/app/requirements_hf.txt

# Install remaining dependencies
RUN pip install --no-cache-dir --user -r requirements_hf.txt

# Copy the rest of the application files
COPY --chown=user . $HOME/app

# Expose Streamlit port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Run Streamlit — disable CORS/XSRF for Hugging Face's reverse proxy
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]
