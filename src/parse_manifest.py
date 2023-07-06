import pandas as pd
import xmltodict

from tqdm import tqdm

# aws s3 ls s3://arxiv/pdf --recursive --request-payer requester
# aws s3 cp s3://arxiv/pdf/arXiv_pdf_9911_001.tar ./arXiv_pdf_9911_001.tar --request-payer requester
# aws s3api copy-object --copy-source arxiv/pdf/arXiv_pdf_0001_001.tar --request-payer requester --key arXiv_pdf_0001_001.tar --bucket fantaarxiv

with open('../dataset/arXiv_pdf_manifest.xml', 'tr') as manifest_file:
    manifest = manifest_file.read()
# tree = ET.parse('../dataset/arXiv_pdf_manifest.xml')
# root = tree.getroot()
parsed = xmltodict.parse(manifest)

metadata = pd.DataFrame(parsed['arXivPDF']['file'])
years = metadata['yymm'].map(lambda x: int(x[:2]))
relevant = (22 <= years) & (years <= 23)
total_size_relevant = metadata['size'].astype(int).dot(relevant)
print(
    f'Total size of tar files within dates of interest {total_size_relevant} B, {total_size_relevant / 1024 / 1024 / 1024} GB')
with open('../dataset/files.txt', 'tr') as file_with_list:
    files_list = file_with_list.readlines()

files_list = [line.split('/')[-1].rstrip('\n').split('v')[0] for line in files_list]
metadata = metadata[relevant]
metadata['first_item_date'] = metadata['first_item'].transform(lambda item: item.split('.')[0])
metadata['first_item_progr'] = metadata['first_item'].transform(lambda item: item.split('.')[-1])
metadata['last_item_date'] = metadata['last_item'].transform(lambda item: item.split('.')[0])
metadata['last_item_progr'] = metadata['last_item'].transform(lambda item: item.split('.')[-1])

# metadata = metadata[pd.to_numeric(metadata['first_item_date'], errors='coerce').notnull()]
# metadata = metadata[pd.to_numeric(metadata['first_item_progr'], errors='coerce').notnull()]

metadata['first_item_progr'] = pd.to_numeric(metadata['first_item_progr'])
metadata['last_item_progr'] = pd.to_numeric(metadata['last_item_progr'])
metadata['first_item_date'] = pd.to_numeric(metadata['first_item_date'])
metadata['last_item_date'] = pd.to_numeric(metadata['last_item_date'])

files_needed = []
for file in tqdm(files_list):
    file_date, file_progr = map(lambda item: int(item), file.split('.'))
    if file_date // 100 < 22 or file_date // 100 > 23:
        continue
    selected = metadata[(metadata.first_item_date <= file_date) & (metadata.last_item_date >= file_date)]
    selected2 = selected[(selected.first_item_progr <= file_progr) & (selected.last_item_progr >= file_progr)]
    assert 0 <= len(selected2) <= 1
    if len(selected2) > 0:
        files_needed.append({'tar_file_name': selected2.filename.iloc[0], 'pdf_file_name': file})

files_needed_set = set([file['tar_file_name'] for file in files_needed])
print(f'Number of tar files needed {len(files_needed_set)}')
total_size_relevant2 = 0
for item in files_needed_set:
    total_size_relevant2 += int(metadata[(metadata.filename == item)].iloc[0]['size'])

print(f'Total size of tar files needed {total_size_relevant2} B, {total_size_relevant2/1024/1024/1024} GB')

metadata['filename'].to_csv('../dataset/tarfiles.txt', header=False, index=False)