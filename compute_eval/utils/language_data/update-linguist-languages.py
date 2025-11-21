import gzip
from pathlib import Path

import requests

LINGUIST_URL = "https://raw.githubusercontent.com/github-linguist/linguist/HEAD/lib/linguist/languages.yml"

response = requests.get(LINGUIST_URL)

# If the file exists (status code is 200), write the content to a new gzipped file
if response.status_code == 200 and response.text:
    print("Writing languages.yml.gz file")
    script_dir = Path(__file__).parent
    output_path = script_dir / "languages.yml.gz"
    with gzip.open(output_path, "wt") as f:
        f.write(response.text)
else:
    print("Failed to fetch languages.yml file")
