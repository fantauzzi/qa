import asyncio
from time import sleep, time

import aiohttp

chunk_size = 1024 * 1024
concurrent_reqs = 4


class DownloadFailed(Exception):
    pass


async def download_file(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            filename = url.split('/')[-1]
            with open(f'/home/fanta/Downloads/arxiv/{filename}.pdf', 'wb') as f:
                while True:
                    chunk = await response.content.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"Downloaded {filename} from {url}")
        else:
            print(f"Failed to download {url} with response status {response.status}")
            raise DownloadFailed()
        sleep(1)


async def download_files(urls):
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrent_reqs)  # Limits concurrent requests to concurrent_reqs
        jobs = []
        for url in urls:
            async with semaphore:
                job = asyncio.ensure_future(download_file(session, url))
                jobs.append(job)
        await asyncio.gather(*jobs)


async def main():
    urls_file = "../dataset/files.txt"  # Path to the text file containing URLs
    with open(urls_file, 'r') as f:
        urls = f.read().splitlines()
    idx = 0
    start_time = time()
    while idx < len(urls):
        batch = urls[idx: idx + concurrent_reqs]
        await download_files(batch)
        completed = min(len(urls), idx+concurrent_reqs)
        current_time = time()
        elapsed = current_time - start_time
        rate = completed / elapsed
        estimate_time_left = (len(urls) - completed)/rate
        print(f'Downloaded {completed} files in {int(elapsed)} sec, estimated time left {int(estimate_time_left)} sec, {estimate_time_left/3600} hours')
        sleep(1.1)
        idx+=concurrent_reqs


if __name__ == "__main__":
    asyncio.run(main())
