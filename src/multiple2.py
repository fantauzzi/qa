import asyncio

import aiohttp
import backoff

chunk_size = 1024 * 1024
concurrent_reqs = 4


class DownloadFailed(Exception):
    pass


@backoff.on_exception(backoff.expo, Exception)
async def download_file(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            filename = url.split('/')[-1]
            with open(f'../dataset/download/{filename}.pdf', 'wb') as f:
                while True:
                    chunk = await response.content.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"Downloaded {filename} from {url}")
        else:
            print(f"Failed to download {url} with response status {response.status}")
            raise DownloadFailed()


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
    urls_file = "../dataset/download/files.txt"  # Path to the text file containing URLs
    with open(urls_file, 'r') as f:
        urls = f.read().splitlines()
        await download_files(urls)


if __name__ == "__main__":
    asyncio.run(main())
