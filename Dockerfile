# Specifying the base image for this build.
FROM python:3.10-slim

# Setting the default working directory.
WORKDIR /code

# Setting the Hugging Face home directory to a local folder inside the project.
# This forces all transformers/datasets/hub downloads to be stored in /code/cache,
# Helps avoiding PermissionErrors.
ENV HF_HOME /code/cache/

# Copying and installing Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Downloading necessary NLTK data files.
RUN python -m nltk.downloader wordnet averaged_perceptron_tagger averaged_perceptron_tagger_eng omw-1.4 punkt

# Copying the application code into the container.
COPY . .

# Defining the command to run the application server.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]