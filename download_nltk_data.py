import nltk
import ssl
import os

# --- Project Root and Data Path ---
# Define the target directory for NLTK data right inside our project.
# This makes the project self-contained and portable.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(ROOT_DIR, "nltk_data")

# --- SSL Certificate Fix ---
def _disable_ssl_verification():
    """
    Disables SSL certificate verification to prevent download errors on
    networks with self-signed certificates.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # For Python versions that don't have this attribute
        pass
    else:
        # Override the default context
        ssl._create_default_https_context = _create_unverified_https_context

# --- Required NLTK Packages ---
# This dictionary contains the standard NLTK download IDs for the resources
# your application needs. 'punkt' is added for tokenization.
REQUIRED_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    # Added explicitly to prevent LookupError in some environments:
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "omw-1.4": "corpora/omw-1.4",
    "punkt": "tokenizers/punkt",
}

def download_data():
    """
    Downloads all required NLTK data packages into the project's 'nltk_data' directory.
    """
    # Apply the SSL fix before attempting to download
    _disable_ssl_verification()

    # Ensure the target directory exists.
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    print(f"--- Starting NLTK data download to: {DOWNLOAD_DIR} ---")
    for package_id, resource_path in REQUIRED_PACKAGES.items():
        try:
            # Check if the resource is already available before downloading
            if not nltk.data.find(resource_path):
                print(f"Downloading '{package_id}'...")
                nltk.download(package_id, download_dir=DOWNLOAD_DIR)
                print(f"Successfully downloaded '{package_id}'.")
            else:
                print(f"'{package_id}' already downloaded.")
        except Exception as e:
            print(f"Error downloading '{package_id}': {e}")
    print("--- NLTK data download complete ---")

if __name__ == "__main__":
    download_data()

