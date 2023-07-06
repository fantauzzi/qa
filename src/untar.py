import asyncio
import os
import shutil
from pathlib import Path
from time import sleep, time

chunk_size = 1024 * 1024
concurrent_reqs = 20


class DownloadFailed(Exception):
    pass


def shell_cmd(cmd: str):
    print(f'Running: {cmd}')
    res = os.system(cmd)
    print(f'Got {res}')
    return res


async def copy_over_file(untarred_pdf):
    cmd = f'aws s3 cp {str(untarred_pdf)} s3://fantaarxiv/pdf2/{untarred_pdf.name}'
    print(f'Running: {cmd}')
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    print(f'Got stdout: {stdout}')
    print(f'Got stderr: {stderr}')


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
        sleep(1)


async def copy_over(to_be_copied_over):
    # async with aiohttp.ClientSession() as session:
    # semaphore = asyncio.Semaphore(concurrent_reqs)  # Limits concurrent requests to concurrent_reqs
    jobs = []
    for untarred_pdf in to_be_copied_over:
        job = asyncio.ensure_future(copy_over_file(untarred_pdf))
        jobs.append(job)
    await asyncio.gather(*jobs)


async def main():
    """
    urls_file = "../dataset/files2.txt"  # Path to the text file containing URLs
    with open(urls_file, 'r') as f:
        urls = f.read().splitlines()
        await download_files(urls)
    """

    with open('../dataset/short.txt') as tar_list:
        tar_files = tar_list.readlines()

    tar_files = [item.rstrip('\n') for item in tar_files]

    with open('../dataset/files.txt') as pdf_list:
        pdf_files = pdf_list.readlines()

    pdf_files = [item.rstrip('\n').split('/')[-1] for item in pdf_files]
    pdf_files = [item.split('v')[0] + '.pdf' for item in pdf_files]

    pdf_files = set(pdf_files)

    copied_count = 0
    start_time = time()
    for idx, tar_file in enumerate(tar_files):
        # copy the tar file from the arxiv s3 bucket to here
        cmd = f'aws s3 cp s3://arxiv/pdf/{tar_file} {tar_file} --request-payer requester'
        shell_cmd(cmd)
        # untar the tar file
        if not Path('untarred').exists():
            Path('untarred').mkdir()
        cmd = f'cd untarred; tar xvf ../{tar_file}'
        shell_cmd(cmd)
        # read from the dir the list of untarred files
        just_untarred = list(Path('.').glob('untarred/*/*'))
        # from every file in the list, check if it is in pdf_files; if so, copy it to my s3 bucket
        to_be_copied_over = [untarred_pdf for untarred_pdf in just_untarred if untarred_pdf.name in pdf_files]
        await copy_over(to_be_copied_over)
        copied_count += len(to_be_copied_over)
        """
        for untarred_pdf in just_untarred:
            filename = untarred_pdf.name
            if filename in pdf_files:
                to_be_copied_over
                
                cmd = f'aws s3 cp {str(untarred_pdf)} s3://fantaarxiv/pdf/{filename}'
                shell_cmd(cmd)
                copied_count += 1
        """
        # remove the local copy of the tar file
        Path(tar_file).unlink()
        # remove the untarred files
        shutil.rmtree('untarred')
        elapsed = time() - start_time
        processed_rate = (idx + 1) / elapsed
        estimated_remaining_time = (len(tar_files) - (idx + 1)) / processed_rate
        print(
            f'Processed {idx + 1} tar files out of {len(tar_files)}, copied {copied_count} PDF so far. Elapsed {int(elapsed)} sec, estimated remaining time {estimated_remaining_time / 3600} h')

    print(f'Completed copying {copied_count} PDF files')


if __name__ == "__main__":
    asyncio.run(main())
