# install.packages(c(
#  "tidyverse","tidymodels","janitor","skimr","here","vip",
#  "ranger","xgboost","themis","yardstick","DALEX","glmnet",
 # "shiny","plumber","readr","rsample"
))
raw_dir <- "/Users/mayurkalpe/Downloads/Projects/Project-Churn/data/raw"
list.files(raw_dir, recursive = FALSE)

# Step 1: load + sneak - peek + date ranges
library(tidyverse)
library(janitor)
library(lubridate)
library(skimr)
root <- "/Users/mayurkalpe/Downloads/Projects/Project-Churn/"
raw <- file.path(root, "data/raw")
read_clean <- function(path) readr::read_csv(path, show_col_types = FALSE) %>% clean_names()


accounts <- read_clean(file.path(raw,"ravenstack_accounts.csv"))
churn_ev <- read_clean(file.path(raw,"ravenstack_churn_events.csv"))
usage <- read_clean(file.path(raw,"ravenstack_feature_usage.csv"))
subs <- read_clean(file.path(raw,"ravenstack_subscriptions.csv"))
tickets <- read_clean(file.path(raw,"ravenstack_support_tickets.csv"))

# Quick shapes and columns

cat("\nRows & columns:\n")
tibble(
  dataset = c("accounts","churn_ev","usage", "subs", "tickets"),
  nrows = c(nrow(accounts), nrow(churn_ev), nrow(usage), nrow(subs), nrow(tickets)),
  ncol  = c(ncol(accounts), ncol(churn_ev), ncol(usage), ncol(subs), ncol(tickets))
  
) %>% print(n=Inf)

cat("\nColumns (first few):\n")
lapply(
  list(accounts=accounts, churn_ev=churn_ev, usage=usage, subs=subs, tickets=tickets),
  function(df) head(df,3)
)

# Find likely date columns and print ranges (robust parsing)

try_parse_date <- function(x) {
  #returns Date vector from char/num/POSIX/Date
  if(inherits(x, "Date")) return (x)
  if(inherits(x, "POSIXct")) return (as.Date(x))
  x <- as.character(x)
  cands <- list(
    suppressWarnings(ymd_hms(x,quiet=TRUE)),
    suppressWarnings(ymd(x,quiet=TRUE)),
    suppressWarnings(mdy(x,quiet=TRUE)),
    suppressWarnings(dmy(x,quiet=TRUE))
  )
  best <- cands[[which.max(sapply(cands,function(v) sum(!is.na(v))))]]
  as.Date(best)
}

date_report <- function(df,name) {
  # Pick columns whose names suggests date/ timestamps
  cn <- names(df)
  idx <- str_detect(cn, "(date|_at$|created|updated|start|end)")
  if (!any(idx)) return(invisible(NULL))
  cols <- cn[idx]
  for(c in cols) {
    d <- try_parse_date(df[[c]])
    if (sum(!is.na(d)) > 0) {
      rng <- range(d,na.rm=TRUE)
      cat(sprintf("%-10s | %-28s | %s to %s | n=%d\n",
                  name, c, as.character(rng[1]), as.character(rng[2]), sum(!is.na(d))))
    }
  }
}

cat("\nDate ranges (by column) : \n")
date_report(accounts, "accounts")
date_report(churn_ev, "churn_ev")
date_report(usage, "usage")
date_report(subs, "subs")
date_report(tickets, "tickets")


# Configuring time based split 80-20

as_of_train <- seq(ymd("2024-02-01"), ymd("2024-09-01"), by = "1 month")
as_of_val <- ymd("2024-10-01")
as_of_test <- seq(ymd("2024-11-01"), ymd("2024-12-01"), by = "1 month")
lookback_days <- 90
label_days <- 30

cat("Snapshots set.\n",
    "Train:", paste(as_of_train, collapse= ", "), "\n",
    "Val:", paste(as_of_val), "\n",
    "Test:", paste(as_of_test, collapse = ", "), "\n\n")

#find the common account identifier (Join key)
common_cols <- Reduce(intersect, lapply(list(accounts, subs, usage, tickets, churn_ev), names))
#Heuristic: prefer something like account_id/customer_id
pref <- c("account_id", "customer_id", "acct_id", "id", "account")
key_candidates <- intersect(pref, common_cols)
if (length(key_candidates) == 0) 
  key_candidates <- common_cols[grepl("account|customer|acct|id", common_cols)]
cat("key candidates:", paste(key_candidates, collapse = ", "), "\n")



































