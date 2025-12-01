import copy
import json
import os
import hashlib
import time
import requests
from thefuzz import fuzz

# TODO
# 1. Get list of papers with the same title - with year, journal name, author name fields also selected
#  1.1. (Done) Case agnostic - already by default on openalex
#  1.2. TODO Some strange errors, "Invalid" reference, looks like it's splitting on punctuation?
# 2. TODO Filter by publication year (need better querying of API)
# 3. (Done) Fuzzy search by journal name (for abbreviations)
# 4. TODO Fuzzy search on authors (for initials and name order) - Implement yourself, as Levenshtein alone won't work

# TODO Set a threshold via experimentation
FUZZY_MATCH_THRESHOLD = 75
OPENALEX_CACHE_PATH = os.environ.get("OPENALEX_CACHE_PATH", os.path.join("data", "openalex_cache.json"))
OPENALEX_CACHE_TTL_HOURS = int(os.environ.get("OPENALEX_CACHE_TTL_HOURS", "24"))


class _JsonCache:
    def __init__(self, path: str, ttl_hours: int):
        self.path = path
        self.ttl_seconds = max(1, ttl_hours) * 3600
        self._store = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._store = json.load(f)
        except FileNotFoundError:
            self._store = {}
        except Exception:
            self._store = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._store, f)
        except Exception:
            pass

    def _expired(self, entry):
        ts = entry.get("ts") or 0
        return (time.time() - ts) > self.ttl_seconds

    def get(self, key):
        entry = self._store.get(key)
        if not entry:
            return None
        if self._expired(entry):
            return None
        return entry.get("data")

    def set(self, key, value):
        self._store[key] = {"ts": time.time(), "data": value}


_CACHE = _JsonCache(OPENALEX_CACHE_PATH, OPENALEX_CACHE_TTL_HOURS)

#TODO Implement test mode: Test OpenAlex accuracy by querying on objects that have a DOI in input - after fixing querying problems
test_mode = False

# Populate a dictionary with values from the original e.g., if no match was found on OpenAlex
def default_copy(orig, klist):
    obj_dict = dict()
    for key in klist:
        if key == 'authors': # Need to do a deep copy for lists inside dictionaries
            obj_dict[key] = copy.deepcopy(orig[key])
        else:
            obj_dict[key] = orig.get(key)
    return obj_dict

# Process the OpenAlex output into the format we want
def parse_query_results(query_output):
    #print(query_output)
    try:
        info = query_output['results']
        parsed_list = []
        if len(info) == 0:
            #print('No results found on OpenAlex...')
            return []
        for result in info:
            #print(result)
            obj_dict = dict()
            obj_dict['doi'] = result['doi']
            obj_dict['openalex_id'] = result['id']
            obj_dict['title'] = result['title']
            obj_dict['year'] = result['publication_year']
            if result['primary_location']:
                if result['primary_location']['source']:
                    if result['primary_location']['source']['display_name']:
                        obj_dict['journal'] = result['primary_location']['source']['display_name']
                    else:
                        obj_dict['journal'] = None
                else:
                    obj_dict['journal'] = None
            else:
                obj_dict['journal'] = None
            
            author_list_raw = result['authorships']
            obj_dict['author'] = []
            for author in author_list_raw:
                obj_dict['author'].append(author['author']['display_name'])
            
            parsed_list.append(obj_dict)
        return parsed_list
    except KeyError:
        etyp = query_output['error']
        emsg = query_output['message']
        #print('ERROR: Could not retrieve results from OpenAlex!')
        #print(etyp, emsg)
        return []

def match_title_and_year(title,year):
    url = 'https://api.openalex.org/works?filter=title.search:"{title}"&mailto=cruzersoulthrender@gmail.com'.format(title=title, year=year) #TODO Generic email
    # Cache by URL
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    response = requests.get(url)
    try:
        res = response.json()
        title_matches = parse_query_results(res)
        try:
            _CACHE.set(key, title_matches)
            _CACHE.save()
        except Exception:
            pass
        return title_matches
    except json.decoder.JSONDecodeError:
        print(response)
        return []

def match_year(title_matches, year):
    year_matches = []
    for result in title_matches:
        res_year = result['year']
        if res_year == year:
            year_matches.append(result)
    return year_matches

def fuzzy_match_journal(year_matches, journal):
    max_similarity = 0 # TODO Select the highest or all above threshold? Or both?
    journal_matches = []
    for result in year_matches:
        obj_dict = copy.deepcopy(result) # Deep copy to avoid modifying the input
        res_journal = result['journal']
        similarity = fuzz.ratio(res_journal, journal)
        obj_dict['journal'] = similarity
        journal_matches.append(obj_dict)
        assert 'match' not in result # Sanity check to make sure we created a new object
    return journal_matches

def fuzzy_match_authors(journal_matches, author_list):
    # TODO Multiple types of fuzzy matching (order and abbreviations)
    # TODO Save the match scores for each comparison to find
    # TODO Author list converted to dictionary with different match scores for each author
    max_similarity = 0 # TODO Select the highest or all above threshold? Or both?
    author_matches = []
    for result in journal_matches:
        obj_dict = copy.deepcopy(result)
        obj_dict['author_list'] = dict()
        assert isinstance(result['author_list'], list) # Sanity check to make sure we created a new object
        res_author_list = result['author_list']
        if len(author_list) == len(res_author_list):
            matched_author_list = '' # To make sure that one author name is not matched to multiple
            for a1 in author_list:
                max_basic = 0
                max_order = 0
                matched_author = ''
                for a2 in res_author_list:
                    similarity_basic = fuzz.ratio(a1, a2)
                    similarity_order = fuzz.token_sort_ratio(a1, a2)
                    if similarity_basic > max_basic or similarity_order > max_order: # If it is the same order then either reordering or initialization should give higher scores
                        max_basic = similarity_basic
                        max_order = similarity_order
                        matched_author = 'a2'
                matched_author_list.append(matched_author)
                obj_dict['author_list'][a1] = (matched_author, max_basic, max_order)
            if len(set(matched_author_list)) == len(author_list): # To make sure one author is not falsely matched to multiple
                author_matches.append(obj_dict)
    return author_matches

#TODO Main method

fname = 'data/first_preprint_references_without_doi_crossref.json'
fout = 'data/openalex_doi_matched_preprints_with_references.json'

with open(fname, 'r') as f:
    reflist = json.load(f)

parsed_data = []

ctr = 0


references_key_list = ['ref_id', 'doi', 'title', 'authors', 'journal', 'year']

ref_doi = []
for ref in reflist:
    tm = dict()
    tm['ref_id'] = ref['ref_id']
    if ref['has_doi']:
        if not test_mode:
            tm = default_copy(ref, references_key_list)
    else:
        title_matches = match_title_and_year(ref['title'], ref['year'])
        #print(ref['title'], len(title_matches))
        if len(title_matches) > 0: #TODO Better selection (e.g. with fuzzy matching) than just first result with doi
            tm0 = title_matches[0]
            if tm0['doi'] is None:
                for pm in title_matches[1:]:
                    if pm['doi'] is not None:
                        tm0 = pm
            for key in tm0:
                tm[key] = tm0[key]
        else:
            tm = default_copy(ref, references_key_list)
    ref_doi.append(tm)

with open(fout, 'w') as f:
    json.dump(parsed_data, f)
