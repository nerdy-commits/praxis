FROM python:3.10-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
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

# Copy requirements.txt
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# Install CPU-only PyTorch and torchvision first to prevent timeout and out-of-memory errors
RUN pip install --no-cache-dir --user torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir --user --upgrade -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . $HOME/app

# Expose Streamlit port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
