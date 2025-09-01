# Specifying the base image for this build. 'python:3.10-slim' is a lightweight,
# official Python image, which results in a smaller and more secure final container.
FROM python:3.10-slim

# Setting the default working directory for all subsequent commands within the container.
WORKDIR /code

# Copying the Python dependencies file from the local project root into the
# container's working directory.
COPY requirements.txt .

# Executing the pip command to install the packages listed in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copying the Python script responsible for downloading NLTK data into the container.
COPY download_nltk_data.py .

# Executing the script to download and save the required NLTK packages.
RUN python download_nltk_data.py

# Copying the entire content of the local project directory into the container's
# working directory. This includes the 'backend', 'src', and 'pipelines' folders.
COPY . .

# Defining the command that will be executed when the container starts.
# This runs the Uvicorn server, making the FastAPI application accessible.
# host 0.0.0.0: Binds the server to all network interfaces, which is required for it to be accessible from outside the container.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]