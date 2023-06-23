import logging
import urllib.request
from time import sleep

import feedparser
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning
error = _logger.error

url = 'http://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=lastUpdatedDate&sortOrder=descending'
page_size = 100
current_position = 0
earliest_year = 2022
entries_as_list = []
max_pages = 100
# while earliest_date_in_page >= earliest_year and current_page_n < max_pages:
while True:
    query = f'{url}&start={current_position}&max_results={page_size}'
    info(f'Running query starting from position {current_position}: {query}')
    data = urllib.request.urlopen(query)
    res = data.read().decode('utf-8')
    parsed = feedparser.parse(res)

    total_results = int(parsed.feed.opensearch_totalresults)
    start_index = int(parsed.feed.opensearch_startindex)
    assert start_index == current_position
    items_per_page = int(parsed.feed.opensearch_itemsperpage)

    entries_in_page = [{'updated': entry['updated'],
                        'title': entry['title'],
                        'link': entry['link'],
                        'doc_id': entry['id'],
                        'primary_category': entry['arxiv_primary_category']['term']} for entry in parsed.entries]
    if len(entries_in_page) < page_size:
        info(
            f'Received only {len(entries_in_page)} entries from position {current_position} out of {page_size} maximum')
    if entries_in_page:
        info(
            f"Received entries from {entries_in_page[-1]['updated']} to {entries_in_page[0]['updated']} with total_results={total_results} start_index={start_index} items_per_page={items_per_page}")
        to_be_added = [entry for entry in entries_in_page if int(entry['updated'][:4]) >= earliest_year]
        entries_as_list.extend(to_be_added)
        # Found an entry that pre-dates the earliest year of interest, stop querying
        if int(entries_in_page[-1]['updated'][:4]) < earliest_year:
            info(f'Received query result that pre-dates {earliest_year}, query completed')
            break
        # Read the whole expected output of the query, stop querying
        if start_index + len(entries_in_page) >= total_results:
            info(
                f'Received {start_index + len(entries_in_page)} results overall out of {total_results} expected, query completed')
            if len(entries_as_list) != total_results:
                warning(f'Expected {total_results} results overall, got {len(entries_as_list)} instead')
            break
        # The query returned at least one entry, but the results aren't complete, so keep on querying
        info(f'Query at position {current_position} completed')
        current_position += len(entries_in_page)
    else:  # If the query didn't retrieve anything, pause and then try again the same query
        sleep(3)

info(f'Fetched {len(entries_as_list)} entries overall')
entries = pd.DataFrame(entries_as_list)
entries.rename({'updated': 'updated_orig'}, inplace=True, axis=1)
entries['updated'] = pd.to_datetime(entries['updated_orig'])
entries.drop(['updated_orig'], inplace=True, axis=1)
entries.to_csv('../dataset/metadata.csv')

assert entries.link.equals(entries.doc_id)

duplicates = entries.duplicated(keep=False)
if any(duplicates):
    warning('Found duplicate rows')
