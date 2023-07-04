from pathlib import Path
from time import time, sleep
from urllib import request

from tqdm import tqdm

interval = 1  # Minimum interval requested between subsequent downloads, in seconds
urls_file = "../dataset/files.txt"  # Path to the text file containing URLs for download
local_download_path = '/home/fanta/Downloads/arxiv'  # Destination dir for downloaded files

with open(urls_file, 'r') as f:
    urls = f.read().splitlines()

for url in tqdm(urls):
    file_name = url[url.rfind('/') + 1:]
    local_file = Path(local_download_path) / file_name
    if local_file.exists():
        continue
    last_request = time()
    try:
        request.urlretrieve(url, str(local_file))
    except Exception as ex:
        print(f'Got exception while trying to download {url}: {ex}')
    time_since_last = time() - last_request
    if time_since_last < interval:
        sleep(interval - time_since_last)

print('Job done')
