import os
from time import time
import shutil
from pathlib import Path

with open('../dataset/short.txt') as tar_list:
    tar_files = tar_list.readlines()

tar_files = [item.rstrip('\n') for item in tar_files]

with open('../dataset/files.txt') as pdf_list:
    pdf_files = pdf_list.readlines()

pdf_files = [item.rstrip('\n').split('/')[-1] for item in pdf_files]
pdf_files = [item.split('v')[0] + '.pdf' for item in pdf_files]

pdf_files = set(pdf_files)


def shell_cmd(cmd: str) -> None:
    print(f'Running: {cmd}')
    res = os.system(cmd)
    print(f'Got {res}')

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
    for untarred_pdf in just_untarred:
        filename = untarred_pdf.name
        if filename in pdf_files:
            cmd = f'aws s3 cp {str(untarred_pdf)} s3://fantaarxiv/pdf/{filename}'
            shell_cmd(cmd)
            copied_count += 1
    # remove the local copy of the tar file
    Path(tar_file).unlink()
    # remove the untarred files
    shutil.rmtree('untarred')
    print(f'Processed {idx + 1} tar files out of {len(tar_files)}, copied {copied_count} PDF so far')

elapsed = time()-start_time
print(f'Completed copying {copied_count} PDF files in {int(elapsed)} sec')