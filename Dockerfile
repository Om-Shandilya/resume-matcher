# Specifying the base image for this build.
FROM python:3.10-slim

# Setting the default working directory.
WORKDIR /code

# Set the Hugging Face home directory to a local folder inside our project.
ENV HF_HOME /code/cache/
# Pre-creating the cache directory with the correct permissions during the build.
RUN mkdir -p /code/cache/ && \
    chown -R 1000:1000 /code/cache/

# Copying and installing Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the application code into the container.
COPY . .

# Defining the command to run the application server port 7860 as huggingface spaces uses this port.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]