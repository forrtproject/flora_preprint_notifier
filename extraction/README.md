# Extracting author information from OSF preprints

The script `extractAuthorList.R` extracts and combines information on authors of OSF preprints. 

It takes author information from XML files, names and orcids scraped from preprint PDFs and combines all this information. It extracts additional information from OSF and ORCIDs to make authors as identifiable as possible and get at least one email per preprint. 
Input: 

   * XML files from GROBID (path defined below) OR RDS named `paper_list.rds` of read in xmls with papercheck (same folder) 
  * JSON file from `extract_orcids_from_pdfs.py` (same folder)
  * JSON file from `extract_emails_from_pdfs.py` (same folder)

Output:
 
   * CSV file containing author information: `authorList_ext.csv`

If `save.int` is TRUE, it also saves intermediate CSV files. 

## Process

1. Extract information from XML files produced by GROBID > converted into table using `papercheck::author_table`
2. Use the OSF id of the preprint to extract all coauthors and their information, e.g., ORCID, names, affiliations... OSF assigns ids to all coauthors, regardless of whether they have an OSF account
3. Fuzzy join the XML information with the information based on the author OSF ids
	1. Based on full names
	2. For those were full names did not produce a match, use the first name and the surname
	3. Then, use the given name initials and the surname
	4. Then, use only the given name
	5. Last, use only the surname
4. Disregard all coauthors extracted with GROBID that could not be matched with an OSF id as these are likely to be acknowledged and not coauthors; then, get rid of duplicate coauthors
6. Add ORCIDs scraped from the PDFs by finding the closest name match
7. Add ORCIDs based on the author names from OSF > only consider perfect matches
8. Extract email addresses and personal information from ORCID
9. Add email addresses scraped from the PDFs > not assigned to a specific person, only to a specific preprint
10. Add affiliations based on ORCID > this can possibly be used later to find out institutional email addresses

## Output columns


* `id` : preprint OSF id
* `name.surname` : author surname from the XML
* `name.given` : author given names from the XML
* `email` : author email address
* `affiliation` : author affiliation from the XML
* `orcid.xml` : ORCID based on XML, if present, otherwise NA
* `source` : source of the preprint, e.g. OSF or psyarxiv
* `osf.id` : OSF id of the author
* `osf.name` : full name based on the OSF id
* `osf.name.given` : given name(s) based on the OSF id
* `osf.name.surname` : surname based on the OSF id
* `osf.affiliation` : current affiliation(s) based on the OSF id
* `github` : github based on the OSF id
* `orcid.osf` : ORCID based on OSF id* `orcid` : ORCID, based on all the sources for ORCIDs
* `orcid` : relevant ORCID
* `orcid.source` :  source of the ORCID in the `orcid` column
* `email.source` : source of the email address in the `email` column
* `orcid.name.given` : given name based on the ORCID
* `orcid.name.surname` : surname based on the ORCID 
* `pdf.all.emails` : email addresses scraped from the PDF, not assigned to a specific author
* `affiliation.orcid` : affiliation based on the ORCID profile
