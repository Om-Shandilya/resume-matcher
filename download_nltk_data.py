import nltk
import ssl
import os

# Project Root and Data Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(ROOT_DIR, "nltk_data")

# SSL Certificate Fix
def _disable_ssl_verification():
    """
    Disables SSL certificate verification to prevent download errors on
    networks with self-signed certificates.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

# Required NLTK Packages
# A definitive list of all data packages needed for the text cleaning functions.
REQUIRED_PACKAGES = [
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "omw-1.4",
    "punkt",
]

def download_data():
    """
    Unconditionally downloads all required NLTK data packages into the
    project's 'nltk_data' directory. This is the most robust method for a Docker build.
    """
    _disable_ssl_verification()

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    print(f"--- Starting NLTK data download to: {DOWNLOAD_DIR} ---")
    for package_id in REQUIRED_PACKAGES:
        try:
            print(f"Downloading '{package_id}'...")
            # Download to our specific, project-local directory.
            nltk.download(package_id, download_dir=DOWNLOAD_DIR)
            print(f"Successfully downloaded '{package_id}'.")
        except Exception as e:
            print(f"Error downloading '{package_id}': {e}")
    print("--- NLTK data download complete ---")

if __name__ == "__main__":
    download_data()