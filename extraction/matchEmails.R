# load libraries > need to be installed with install.packages("name") first
library(tidyverse)
library(RecordLinkage)

# read in the author dataframe
df.authors = read_csv("authorList_ext.csv")

# select the relevant part of the author dataframe
df.sel = df.authors %>% 
  group_by(id) %>% mutate(n = sum(!is.na(email))) %>%
  ungroup() %>%
  filter(pdf.all.emails != "false" & n > 0 & !is.na(id)) %>%
  select(id, email, pdf.all.emails, name.surname, name.given) %>% 
  mutate(email.possible = NA, email.similarity = NA)

ls.ids = split(df.sel, df.sel$id)

for (i in 1:length(ls.ids)) { #
  preprint = names(ls.ids)[i]
  emails = str_split(ls.ids[[i]]$pdf.all.emails[1], "\", \"")
  for (k in 1:length(ls.ids[[i]]$name.surname)) {
    similar = c()
    for (j in 1:length(emails[[1]])) {
      # getting rid of the @... for each of the email
      email.sel = gsub("@.*", "", tolower(emails[[1]][j]))
      # add the highest similarity value for this email / person combination
      similar = c(similar, max(
        # similarity with surname
        levenshteinSim(tolower(ls.ids[[i]]$name.surname[k]),
                       email.sel),
        # similarity with given name
        levenshteinSim(tolower(ls.ids[[i]]$name.given[k]),
                       email.sel),
        # similarity with full name
        levenshteinSim(tolower(sprintf('%s %s', 
                                       ls.ids[[i]]$name.given[k],
                                       ls.ids[[i]]$name.surname[k])),
                       email.sel),
        levenshteinSim(tolower(sprintf('%s %s', 
                                       ls.ids[[i]]$name.surname[k],
                                       ls.ids[[i]]$name.given[k])),
                       email.sel), na.rm = T))
      
    }
    idx = which(similar == max(similar, na.rm = T))
    if (length(idx) > 1) idx = idx[1]
    r = which(df.sel$id == preprint & df.sel$name.surname == ls.ids[[i]]$name.surname[k] & df.sel$name.given == ls.ids[[i]]$name.given[k])
    df.sel[r,]$email.possible   = emails[[1]][idx]
    df.sel[r,]$email.similarity = similar[idx]
  }
}


df.email = df.sel %>%
  mutate(
    # get rid of quotation marks
    email.possible = gsub("\"", "", email.possible)
  ) %>%
  # figure out best match per preprint and possible email
  group_by(id, email.possible) %>%
  mutate(email.rank = max(rank(email.similarity)) - rank(email.similarity)) %>%
  # only keep the best fit and where we don't have an email yet
  filter(email.rank == 0 & is.na(email))

# figure out a threshold that makes sense, possibly adjusted for length ratio

# there are some that don't fit despite really good similariy, e.g.:
# Melacarne Claudio is fit to elcra@unisi.it > probably not a fit? maybe? how do we tell?

# at the end, this can be merged back to df.authors
