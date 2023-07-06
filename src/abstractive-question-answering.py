import hashlib
from pathlib import Path
from pprint import pprint

import pinecone
import torch
# from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# loader = PyPDFLoader("../dataset/HAI_AI-Index-Report_2023.pdf")
# documents = loader.load()
filename = 'HAI_AI-Index-Report_2023.pdf'
file_path = '../dataset' / Path(filename)
reader = PdfReader(file_path)

import pandas as pd

from transformers import BartTokenizer, BartForConditionalGeneration

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get Bart tokenizer and generator model from 🤗
tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)


def tokenized_len(text: str) -> int:
    """
    Given a text returns the number of token making up the text.
    :param text: the given text.
    :return: the number of token in the text
    """
    # TODO could add an argument with the tokenizer, and set it with a partial
    tokens = tokenizer(text)
    return len(tokens.data['input_ids'])


chunk_size = 400

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=200,  # number of tokens overlap between chunks
    length_function=tokenized_len,
    separators=['\n\n', '\n', ' ', '']
)
# text_splitter = SpacyTextSplitter()

print('Chunking the text...')
item_hash = hashlib.md5()
concatenated = ''.join([page.extract_text() for page in reader.pages])
chunks = text_splitter.split_text(concatenated)
chunked_docs = []
for i, chunk in enumerate(chunks):
    item_hash.update(filename.encode('utf-8'))
    uid = item_hash.hexdigest()[:12]
    new_item = {'id': f'{uid}-{i}',
                'text': chunk,
                'source': filename}
    chunked_docs.append(new_item)

"""    
for item in tqdm(documents):
    chunks = text_splitter.split_text(item.page_content)
    item_hash.update(item.metadata['source'].encode('utf-8'))
    uid = item_hash.hexdigest()[:12]
    for i, chunk in enumerate(chunks):
        new_item = {'id': f'{uid}-{i}',
                    'text': chunk,
                    'source': item.metadata['source'],
                    'page': item.metadata['page']}
        chunked_docs.append(new_item)
"""
df = pd.DataFrame(chunked_docs)

print('Creating the Pinecone index...')
# Connect to Pinecone
pinecone.init(
    api_key="a6ece8de-801f-450b-ad1e-b071ca4b73ca",
    environment="us-west1-gcp-free"
)

# Create a new index in Pinecone
index_name = "abstractive-question-answering-pdf"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=768,  # TODO double-check why 768
        metric="cosine"
    )

index = pinecone.Index(index_name)

# Fetch the retriever model from 🤗
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)

print(f'Device is {device}')

batch_size = 128

print('Uploading the index to Pinecone...')
for i in tqdm(range(0, len(df), batch_size)):
    i_end = min(i + batch_size, len(df))  # End of batch
    batch = df.iloc[i:i_end]
    emb = retriever.encode(batch["text"].tolist()).tolist()  # Embeddings for the given batch
    metadata = batch.to_dict(orient="records")
    ids = batch.id
    to_upsert = list(zip(ids, emb, metadata))
    _ = index.upsert(vectors=to_upsert)

index.describe_index_stats()


def query_pinecone(query, top_k):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc


def format_query(query, context):
    # extract text from Pinecone search result and add the <P> tag
    context = [f"<P> {cntx['metadata']['text']}" for cntx in context]
    # concatenate all context passages
    context = " ".join(context)
    # concatenate the query and context passages
    query = f"question: {query} context: {context}"
    return query


def generate_answer(query):
    # tokenize the query to get input_ids
    # For max_length see https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
    inputs = tokenizer([query], max_length=1024, truncation=True, return_tensors="pt").to(device)  # TODO why 1024?
    # use generator to predict output ids
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20,
                             max_length=200)  # TODO with 400 it may allucinate
    # use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer



extraction_query = "What is SuperGLUE?"
extraction_result = query_pinecone(extraction_query, top_k=1)
# print(extraction_result)

# format the query in the form generator expects the input
generative_query = format_query(extraction_query, extraction_result["matches"])
pprint(generative_query)

answer = generate_answer(generative_query)
print('==========================')
pprint(answer)


print('##############################')

extraction_query = "Who has produced the most important machine learning models? The academia or the industry?"
extraction_result = query_pinecone(extraction_query, top_k=1)
# print(extraction_result)

# format the query in the form generator expects the input
generative_query = format_query(extraction_query, extraction_result["matches"])
pprint(generative_query)

answer = generate_answer(generative_query)
print('==========================')
pprint(answer)


print('##############################')

extraction_query = "How many organizations have adopted AI according to the McKinsey report?"
extraction_result = query_pinecone(extraction_query, top_k=1)
# print(extraction_result)

# format the query in the form generator expects the input
generative_query = format_query(extraction_query, extraction_result["matches"])
pprint(generative_query)



answer = generate_answer(generative_query)
print('==========================')
pprint(answer)

"""
TODO
Read from one PDF file from arxive instead of Wikipedia and answer questions -> Done
Clean-up, especially comment -> Done
Should all the pages in the PDF file be concatenated before chunking them? Try that out! -> Done
Resolve pending issues on max_length and how many tokens per extracted text, max_length etc. -> Done

Try replacing pynecone with a local vectorial database and see if you can get at least the same performance
Try another solution, like LangChain
Add reading for parameters from YAML file
Decide for a dataset format/package, preferabily that allows streaming from a dataset without having it all in memory (??) 
Read from multiple PDF files from arxive (make a selection of PDFs) and answer questions
Test various possibilities for the input/output of the first stage (chunkcs how large, how many results, how long)
Try to make it scale from a couple PDF to the whole machine learning arxive
"""
