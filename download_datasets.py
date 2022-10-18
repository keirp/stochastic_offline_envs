# The offline_dataset.zip file is public on Google drive at https://drive.google.com/file/d/1xZxLhgg-cW9VzIxrc4MKlyDl-uhvUagq/view?usp=sharing. This script downloads it, unzips it, and dumps the contents into the offline_data directory. Uses gdown.

import os
import zipfile
import gdown
import shutil

# Path to the zip file
url = 'https://drive.google.com/uc?id=1xZxLhgg-cW9VzIxrc4MKlyDl-uhvUagq'
output = 'offline_dataset.zip'

# Download the zip file
gdown.download(url, output, quiet=False)

# Unzip the file
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('offline_data')

# Delete the zip file
os.remove(output)
