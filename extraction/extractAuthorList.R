# This script takes author information from XML files, names and orcids scraped 
# from preprint PDFs and combines all this information. It extracts additional 
# information from OSF and ORCIDs to make authors as identifiable as possible
# and get at least one email per preprint. 
# Input: 
#      * XML files from GROBID (path defined below)
#   OR * RDS named "paper_list.rds" of read in xmls with papercheck (same folder) 
#      * JSON file from "extract_orcids_from_pdfs.py" (same folder)
#      * JSON file from "extract_emails_from_pdfs.py" (same folder)
#
# Output: 
#      * CSV file containing author information: "authorList_ext.csv"
# 
# If save.int is TRUE, it also saves intermediate CSV files 

# load libraries > need to be installed with install.packages("name") first
library(papercheck)
library(tidyverse)
library(osfr)
library(RecordLinkage)
library(httr)
library(jsonlite)
library(stringdist)
library(fuzzyjoin)

# save intermediary CSV files?
save.int = F

# function to capture percentages of ORCIDs and emails
checkNumbers = function(df) {
  
  library(tidyverse)
  
  # Suppress summarise info
  options(dplyr.summarise.inform = FALSE)
  
  df.sum = df %>% ungroup() %>%
    mutate(percOrcid = mean(!is.na(orcid) )*100, 
           nOrcid = sum(!is.na(orcid)),
           percEmail = mean(!is.na(email) )*100, 
           nEmail = sum(!is.na(email)) ) %>% 
    group_by(id, nOrcid, percOrcid, nEmail, percEmail) %>%
    summarise(checkEmail = sum(!is.na(email)) > 0 ) %>%
    ungroup() %>%
    mutate(nEmailpreprint = sum(checkEmail),
           percEmailpreprint = mean(checkEmail)*100) %>%
    select(-checkEmail, -id) %>%
    distinct()
  
  df.sum$n.authors  = nrow(df)
  df.sum$n.preprint = length(unique(df$id))
  
  print(df.sum)
  
}


# Extract information from XML files --------------------------------------

# read in all the xml files
if (file.exists("paper_list.rds")) {
  paper.lst = readRDS("paper_list.rds")
} else {
  # path to GROBID xmls [!possibly needs to be adjusted]
  path = "../data/preprints"
  paper.lst = read(path) # takes ~2-3min
  saveRDS(paper.lst, "paper_list.rds")
}

# convert to table
df = author_table(paper.lst) %>%
  mutate(
    # separate the source and the OSF id
    source = dirname(id),
    id = basename(id),
    # add an NA if there is no email
    email = if_else(email == "", NA, email)
  )

df %>% group_by(id) %>% 
  summarise(checkEmail = sum(!is.na(email)) > 0) %>%
  ungroup() %>%
  summarise(percEmail = mean(checkEmail))

df %>% summarise(percOrcid = mean(!is.na(orcid)), nOrcid = sum(!is.na(orcid)))

if (save.int) {
  write_csv(df, "authorInfo_xml.csv")
}


# Use OSF preprint id to find coauthor IDs --------------------------------

# split per preprint
ls.ids = split(df, df$id)
df.authors = data.frame()

osf_auth(token = "vxNKbbZar9alGO83uILU5euAtzuZkhbtB4MULCmB75wMlxQlKXujI5AnK4unJIZr1neS1C")

# get all the information from OSF
tictoc::tic()
for (i in 1:length(ls.ids)) {
  if (i%%100 == 0) {
    tictoc::toc()
    print(sprintf("Processed %d of %d preprints", i, length(ls.ids)))
    tictoc::tic()
  }
  # retrieve the contributor information of the preprint
  res = GET(sprintf("https://api.osf.io/v2/preprints/%s/contributors/", 
                    names(ls.ids)[i]))
  data = fromJSON(rawToChar(res$content))
  if (is.null(data$data)) {
    next
  }
  # get the authors' OSF ids and put them into a temporary dataframe
  tmp = data.frame(osf.id = data$data$relationships$users$data$id,
                   osf.name = NA, 
                   osf.name.given = NA,
                   osf.name.surname = NA,
                   osf.affiliation = NA,
                   github = NA, orcid = NA, 
                   id = names(ls.ids)[i])
  # loop through this dataframe and retrieve names and orcids based on OSF id
  for (j in 1:nrow(tmp)) {
    osf.info = osf_retrieve_user(tmp$osf.id[j])
    # socials: github and ORCID
    if (length(osf.info$meta[[1]]$attributes$social) != 0) {
      tmp$orcid[j] = check_orcid(osf.info$meta[[1]]$attributes$social$orcid)
      if (!is.null(osf.info$meta[[1]]$attributes$social$github)) {
        tmp$github[j] = osf.info$meta[[1]]$attributes$social$github
      }
    }
    # personal info: name and affiliation
    tmp$osf.name.given[j]  = sprintf("%s %s", 
                                     osf.info$meta[[1]][["attributes"]][["given_name"]], 
                                     osf.info$meta[[1]][["attributes"]][["middle_names"]])
    tmp$osf.name.surname[j]  = osf.info$meta[[1]][["attributes"]][["family_name"]]
    affiliations = c()
    for (k in seq_len(length(osf.info$meta[[1]][["attributes"]][["employment"]]))) {
      # only keep current affiliations
      if (is.null(osf.info$meta[[1]][["attributes"]][["employment"]][[k]][["endYear"]])) {
        affiliations = c(affiliations, 
                         osf.info$meta[[1]][["attributes"]][["employment"]][[k]][["institution"]])
      } else if (osf.info$meta[[1]][["attributes"]][["employment"]][[k]][["endYear"]] == "") {
        affiliations = c(affiliations, 
                         osf.info$meta[[1]][["attributes"]][["employment"]][[k]][["institution"]])
      }
    }
    tmp$osf.affiliation[j] = paste(affiliations, collapse = ";")
  }
  # even unregistered people get an OSF id!!!
  # add dataframe to the author dataframe
  df.authors = rbind(df.authors, tmp)
}

# save this author list
if (save.int) {
  write_csv(df.authors, "authorList_OSFid-unmerged.csv")
}


# Fuzzy join XML and OSF info ---------------------------------------------

# fuzzy join based on full names
df.authors2 = stringdist_join(
     df %>% mutate(name = tolower(sprintf("%s %s", name.given, name.surname)), # combine name
                   # remove all punctuation and numbers
                   name = gsub("[[:punct:][:digit:][:blank:]]", "", name),
                   orcid = if_else(orcid == "FALSE", NA, orcid)), 
     df.authors %>% mutate(name = tolower(sprintf("%s %s", osf.name.given, osf.name.surname)),
                           name = gsub("[[:punct:][:digit:][:blank:]]", "", name),
                           orcid = if_else(orcid == "FALSE", NA, orcid)),
     by = c("id", "name"),  # Columns to match on
     mode = "full",  
     method = "hamming", distance_col = "distance",
     max_dist = 2  # Maximum allowable distance for a match
)

# separate into those that worked and those that did not
df.done = df.authors2 %>% filter(!is.na(name.x) & !is.na(name.y))
df.miss = df.authors2 %>% filter(is.na(name.x)  |  is.na(name.y))

# fuzzy join based on first name and surname
df.authors2 = stringdist_join(
  df %>% merge(., df.miss %>% select(id.x, n) %>% rename(id = id.x)) %>%
    mutate(name = tolower(sprintf("%s %s", gsub(" .*", "", name.given), name.surname)), # combine name
                # remove all punctuation and numbers
                name = gsub("[[:punct:][:digit:][:blank:]]", "", name),
                orcid = if_else(orcid == "FALSE", NA, orcid)), 
  df.authors %>% merge(., df.miss %>% select(id.y, osf.id) %>% rename(id = id.y)) %>%
    mutate(name = tolower(sprintf("%s %s", gsub(" .*", "", osf.name.given), osf.name.surname)),
                        name = gsub("[[:punct:][:digit:][:blank:]]", "", name),
                        orcid = if_else(orcid == "FALSE", NA, orcid)),
  by = c("id", "name"),  # Columns to match on
  mode = "full",  
  method = "hamming", distance_col = "distance",
  max_dist = 2  # Maximum allowable distance for a match
)

# separate again
df.done = rbind(df.done, df.authors2 %>% filter(!is.na(name.x) & !is.na(name.y)))
df.miss = df.authors2 %>% filter(is.na(name.x)  |  is.na(name.y))

# fuzzy join based on given name initials
df.authors2 = stringdist_join(
  df %>% merge(., df.miss %>% select(id.x, n) %>% rename(id = id.x)) %>%
    mutate(name = tolower(sprintf("%s %s", gsub("[^A-Z]", "", name.given), name.surname)), # combine name
           orcid = if_else(orcid == "FALSE", NA, orcid)), 
  df.authors %>% merge(., df.miss %>% select(id.y, osf.id) %>% rename(id = id.y)) %>%
    mutate(name = tolower(sprintf("%s %s", gsub("[^A-Z]", "", osf.name.given), osf.name.surname)),
           orcid = if_else(orcid == "FALSE", NA, orcid)),
  by = c("id", "name"),  # Columns to match on
  mode = "full",  
  method = "hamming", distance_col = "distance",
  max_dist = 2  # Maximum allowable distance for a match
)

# separate again
df.done = rbind(df.done, df.authors2 %>% filter(!is.na(name.x) & !is.na(name.y)))
df.miss = df.authors2 %>% filter(is.na(name.x)  |  is.na(name.y))

# fuzzy join based on given name
df.authors2 = stringdist_join(
  df %>% merge(., df.miss %>% select(id.x, n) %>% rename(id = id.x)) %>%
    mutate(name = tolower(name.given), orcid = if_else(orcid == "FALSE", NA, orcid),
           # remove all punctuation and numbers
           name = gsub("[[:punct:][:digit:][:blank:]]", "", name)), 
  df.authors %>% merge(., df.miss %>% select(id.y, osf.id) %>% rename(id = id.y)) %>%
    mutate(name = tolower(osf.name.given), orcid = if_else(orcid == "FALSE", NA, orcid),
           # remove all punctuation and numbers
           name = gsub("[[:punct:][:digit:][:blank:]]", "", name)),
  by = c("id", "name"),  # Columns to match on
  mode = "full",  
  method = "hamming", distance_col = "distance",
  max_dist = 2  # Maximum allowable distance for a match
)

# separate again
df.done = rbind(df.done, df.authors2 %>% filter(!is.na(name.x) & !is.na(name.y)))
df.miss = df.authors2 %>% filter(is.na(name.x)  |  is.na(name.y))

# fuzzy join based on surname
df.authors2 = stringdist_join(
  df %>% merge(., df.miss %>% select(id.x, n) %>% rename(id = id.x)) %>%
    mutate(name = tolower(name.surname), orcid = if_else(orcid == "FALSE", NA, orcid),
           # remove all punctuation and numbers
           name = gsub("[[:punct:][:digit:][:blank:]]", "", name)), 
  df.authors %>% merge(., df.miss %>% select(id.y, osf.id) %>% rename(id = id.y)) %>%
    mutate(name = tolower(osf.name.surname), orcid = if_else(orcid == "FALSE", NA, orcid),
           # remove all punctuation and numbers
           name = gsub("[[:punct:][:digit:][:blank:]]", "", name)),
  by = c("id", "name"),  # Columns to match on
  mode = "full",  
  method = "hamming", distance_col = "distance",
  max_dist = 2  # Maximum allowable distance for a match
)

# we assume that OSF ids are correct and throw everything else out at this point
df.done = rbind(df.done, 
                df.authors2 %>% filter(!is.na(osf.id)))

# check for preprint ID conflicts
df.done %>% 
  mutate(match = id.x == id.y) %>%
  filter(!match)

# check for ORCID conflicts
df.done %>% 
  mutate(match = orcid.x == orcid.y) %>%
  filter(!match)

# remove columns
df.authors = df.done %>%
  select(-c(id.x, name.x, name.y, distance, id.distance, name.distance)) %>%
  rename(id = id.y, orcid.xml = orcid.x, orcid.osf = orcid.y)

# merge the orcid and note the source(s)
df.authors = df.authors %>%
  mutate(
    orcid = case_when(
      !is.na(orcid.xml) & !is.na(orcid.osf) ~ orcid.osf,
      is.na(orcid.xml) & !is.na(orcid.osf) ~ orcid.osf,
      !is.na(orcid.xml) &  is.na(orcid.osf) ~ orcid.xml
    ),
    orcid.source = case_when(
      !is.na(orcid.xml) & !is.na(orcid.osf) & orcid.osf == orcid.xml ~ "osf;xml",
      !is.na(orcid.xml) & !is.na(orcid.osf) & orcid.osf != orcid.xml ~ "osf",
      is.na(orcid.xml) & !is.na(orcid.osf)  ~ "osf",
      !is.na(orcid.xml) &  is.na(orcid.osf) ~ "xml"
    )
  )


# Disregard coauthors without OSF id --------------------------------------

# get rid of doubles (authors picked up multiple times by XML)
df.authors = df.authors %>%
  # get rid of special characters except hyphen from the names
  mutate(name.given = gsub("ZZZ", "-", gsub("[^[:alnum:]]", "", gsub("-", "ZZZ", name.given))), 
         name.surname = gsub("ZZZ", "-", gsub("[^[:alnum:]]", "", gsub("-", "ZZZ", name.surname)))) %>%
  # replace empty strings with NAs
  mutate(across(where(is.character), ~na_if(., ""))) %>%
  # group by person's OSF id and the preprint's OSF id
  group_by(osf.id, id) %>% 
  # count how many instances of same person per preprint and how many missing values
  mutate(count = n(), na.count = rowSums(is.na(across(everything())))) %>%
  # only keep those with minimum number of missing values
  filter(count == 1 | (count > 1 & na.count == min(na.count))) %>%
  group_by(osf.id, id) %>%
  # if there are still multiple, get rid of all but one
  mutate(idx = row_number(), count = n()) %>% 
  filter(idx == 1) %>%
  # replace XML values because not reliable
  mutate(
    name.surname = if_else(count > 1, NA, name.surname),
    name.given   = if_else(count > 1, NA, name.given),
    affiliation  = if_else(count > 1, NA, affiliation),
    orcid.xml    = if_else(count > 1, NA, orcid.xml)
  ) %>% ungroup() %>% 
  # get rid of n as it is not reliable and other filtering columns
  select(-c(n, count, na.count, idx)) 

checkNumbers(df.authors)

# save this author list
if (save.int) {
  write_csv(df.authors, "authorList_OSFid.csv")
}


# Add ORCIDs from PDF scraping --------------------------------------------

orcid.json = read_json("orcids_from_pdf.json")

df.orcid = data.frame()
for (i in 1:length(orcid.json)) {
  for (j in 1:length(orcid.json[[i]])) {
    if (orcid.json[[i]][j] != "false") {
      # extract information based on the orcid > only works on real ORCID
      orcid.info = orcid_person(orcid.json[[i]][[j]])
      # CHECK if there was an error > invalid orcid
      if (!is.null(orcid.info$error)) {
        next
      }
      # add this information to the dataframe
      df.orcid = rbind(df.orcid, 
                       data.frame(
                         orcid.pdf = orcid.info$orcid,
                         orcid.name.given  = orcid.info$given,
                         orcid.name.surname = orcid.info$family,
                         id = names(orcid.json)[i]
                       ))
    } else {
      next
    }
  }
}

# disregard all ORCID numbers that we already have
df.orcid = df.orcid %>% 
  filter(!(orcid.pdf %in% df.authors$orcid))

# find the closest match for these additional ORCIDs
for (preprint in unique(df.orcid$id)) {
  df.orcid.sel = df.orcid %>% filter(id == preprint)
  for (i in 1:nrow(df.orcid.sel)) {
    similar = c()
    df.sel = df.authors %>% filter(id == preprint & is.na(orcid))
    for (j in 1:nrow(df.sel)) {
      # add the highest similarity value for this orcid / person combination
      similar = c(similar, max(
        # similarity with surname
        levenshteinSim(tolower(df.sel$osf.name.surname[j]),
                       tolower(df.orcid.sel$orcid.name.surname[i])),
        # similarity with full name
        levenshteinSim(tolower(sprintf('%s %s', 
                                       df.sel$osf.name.surname[j],
                                       df.sel$osf.name.given[j])),
                       tolower(sprintf('%s %s', 
                                       df.orcid.sel$orcid.name.surname[i],
                                       df.orcid.sel$orcid.name.given[i]))),
        na.rm = T))
    }
    idx = which(similar == max(similar))
    df.authors[df.authors$osf.name.surname == df.sel$osf.name.surname[idx] & df.authors$id == df.sel$id[idx],]$orcid = df.orcid.sel$orcid.pdf[i]
    df.authors[df.authors$osf.name.surname == df.sel$osf.name.surname[idx] & df.authors$id == df.sel$id[idx],]$orcid.source = "pdf"
  }
}

checkNumbers(df.authors)

if (save.int) {
  write_csv(df.authors, "authorList_ORCIDpdf.csv")
}


# Add ORCIDs based on name ------------------------------------------------

# find those coauthors for which we need this
idx = which(is.na(df.authors$orcid) & 
              !is.na(df.authors$osf.name.surname) & 
              !is.na(df.authors$osf.name.given))
count = 1
tictoc::tic() # ~5min per 250
for (i in idx) {
  check = get_orcid(gsub("[[:punct:]]", "", df.authors$osf.name.surname[i]), 
                    gsub("[[:punct:]]", "", df.authors$osf.name.given[i]))
  if (length(check) == 1) {
    df.authors$orcid[i] = check
    df.authors$orcid.source[i] = "name"
  }
  count = count + 1
  if (count%%250 == 0) {
    tictoc::toc()
    print(sprintf("Checked %d of %d authors", count, length(idx)))
    tictoc::tic()
  }
}

df.authors = df.authors %>%
  mutate(
    orcid = if_else(orcid == "", NA, orcid)
  )

checkNumbers(df.authors)

if (save.int) { 
  write_csv(df.authors, "authorList_ORCIDname.csv")
}


# Extract info from ORCIDs ------------------------------------------------

df.authors = df.authors %>% 
  mutate(email.source = if_else(!is.na(email), "xml", NA),
         orcid.name.given = NA,
         orcid.name.surname = NA)

# check how many emails we have at this point
checkNumbers(df.authors)

# only look at those who have an ORCID
idx = which(!is.na(df.authors$orcid))
count = 1
tictoc::tic() # 2min per 250
for (i in idx) {
  # extract information from ORCID
  check = orcid_person(df.authors$orcid[i])
  if (is.null(check$error)) {
    if (is.na(df.authors$email[i]) & check$email[[1]][1] != "") {
      df.authors$email[i] = check$email[[1]][1]
      df.authors$email.source[i] = "orcid"
    }
    df.authors$orcid.name.given[i]   = check$given
    df.authors$orcid.name.surname[i] = check$family
  }
  count = count + 1
  if (count%%250 == 0) {
    tictoc::toc()
    print(sprintf("%d of %d", count, length(idx)))
    tictoc::tic()
  }
}

# check again how many we have
checkNumbers(df.authors)

if (save.int) {
  write_csv(df.authors, "authorList_ORCIDemail.csv")
}


# Add email addresses scraped from PDFs -----------------------------------

email.json = read_json("emails_from_pdf.json")
df.email = enframe(email.json, name = "id", value = "pdf.all.emails") %>%
  mutate(
    pdf.all.emails = gsub("list\\(|\\)", "", as.character(pdf.all.emails))
  )

# merge with the data frame
df.authors = merge(df.authors, 
                df.email, 
                all.x = T)

# check how many more emails we can find from the PDF of the preprint
df.authors %>% 
  mutate(
    checkBoth = (!is.na(email)) & (!is.na(orcid))
  ) %>%
  group_by(id) %>%
  summarise(checkEmail = sum(!is.na(email) | pdf.all.emails != "false") > 0,
            checkBoth  = sum(checkBoth) > 0) %>%
  ungroup() %>%
  summarise(percEmail = mean(checkEmail), 
            nEmail    = sum(checkEmail),
            percBoth  = mean(checkBoth))

# check whether the email is in one of the pdf emails
df.authors %>% 
  mutate(
    checkEmail = if_else(is.na(email), NA, str_detect(pdf.all.emails, email))
  ) %>% summarise(sameEmail = mean(checkEmail, na.rm = T))


# Add affiliations based on ORCID -----------------------------------------

df.authors = df.authors %>% mutate(affiliation.orcid = NA)

tictoc::tic() # takes about 6 min
for (i in 1:nrow(df.authors)) {
  
  # ORCID public API endpoint
  url = paste0("https://pub.orcid.org/v3.0/", df.authors$orcid[i], "/employments")
  response = GET(url, add_headers("Accept" = "application/json"))
  
  # Check if the request was successful (status 200)
  if (status_code(response) == 200) {
    
    # parse the JSON response
    data = fromJSON(content(response, "text"))
    
    # extract affiliations
    affiliations = data[["affiliation-group"]][["summaries"]]
    
    # check if any affiliations there
    if (is.null(affiliations)) {
      next
    }
    
    # extract institution name
    institution_names = sapply(affiliations, function(x) x$`employment-summary`$organization$name)
    
    df.authors$affiliation.orcid[i] = list(institution_names)
    
  }
  
}
tictoc::toc()

checkNumbers(df.authors)

# save this extended author list
write_csv(df.authors %>% mutate(affiliation.orcid = as.character(affiliation.orcid)), "authorList_ext.csv")
