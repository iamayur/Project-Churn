############################################################
# RavenStack SaaS Churn Prediction in R
############################################################

# -------------------------------
# 0. Install & load packages
# -------------------------------

# install.packages(c("tidyverse", "lubridate", "tidymodels", "janitor")) "Already ran these hence commmeneted out"

library(tidyverse) #For data wrangling / visualisation- dplyr, ggplot2, readr,etc, pipe (%>%), mutate, group_by, summarise
library(lubridate) #To use date/time format- ymd_hms, as_date, etc
library(tidymodels) #for modelling, preprocessing, recipes
library(janitor) # Cleaning functions and basic tabulations like clean_names
library(ranger) #Fast implementation of random forests
library(xgboost)

tidymodels_prefer()  # helps avoid conflicts

# -------------------------------
# 1. Load data
# -------------------------------
accounts       <- read_csv("data/raw/ravenstack_accounts.csv")
subscriptions  <- read_csv("data/raw/ravenstack_subscriptions.csv")
feature_usage  <- read_csv("data/raw/ravenstack_feature_usage.csv")
support_tickets<- read_csv("data/raw/ravenstack_support_tickets.csv")
churn_events   <- read_csv("data/raw/ravenstack_churn_events.csv")  # not strictly needed, but loaded

# Quick sanity check
glimpse(accounts)
glimpse(subscriptions)
glimpse(feature_usage)
glimpse(support_tickets)
glimpse(churn_events)

# -------------------------------
# 2. Feature engineering
#    (Aggregate to account level)
# -------------------------------

# 2.1 Subscriptions-level aggregates per account
# Compute a global max date for tenure calculations
subs_dates <- c(as.Date(subscriptions$start_date),
                as.Date(subscriptions$end_date))
max_date <- max(subs_dates, na.rm = TRUE)

subscriptions_clean <- subscriptions %>%
  mutate(
    start_date = as.Date(start_date),
    end_date   = as.Date(end_date)
  )

subs_agg <- subscriptions_clean %>%
  group_by(account_id) %>%
  summarise(
    n_subscriptions        = n_distinct(subscription_id),
    n_active_subscriptions = sum(is.na(end_date)),
    avg_mrr                = mean(mrr_amount, na.rm = TRUE),
    max_mrr                = max(mrr_amount, na.rm = TRUE),
    total_mrr              = sum(mrr_amount, na.rm = TRUE),
    avg_arr                = mean(arr_amount, na.rm = TRUE),
    mean_seats             = mean(seats, na.rm = TRUE),
    max_seats              = max(seats, na.rm = TRUE),
    any_upgrade            = any(upgrade_flag),
    any_downgrade          = any(downgrade_flag),
    frac_annual_billing    = mean(billing_frequency == "annual"),
    any_trial_sub          = any(is_trial),
    subscription_tenure_days = as.numeric(
      difftime(
        max(replace_na(end_date, max_date)),
        min(start_date, na.rm = TRUE),
        units = "days"
      )
    ),
    .groups = "drop"
  )

# 2.2 Feature usage aggregates per account
usage_with_account <- feature_usage %>%
  left_join(
    subscriptions_clean %>%
      select(subscription_id, account_id),
    by = "subscription_id"
  )

usage_agg <- usage_with_account %>%
  group_by(account_id) %>%
  summarise(
    total_feature_events = sum(usage_count, na.rm = TRUE),
    total_usage_seconds  = sum(usage_duration_secs, na.rm = TRUE),
    total_errors         = sum(error_count, na.rm = TRUE),
    beta_events          = sum(if_else(is_beta_feature, usage_count, 0L), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    avg_seconds_per_event = if_else(
      total_feature_events > 0,
      total_usage_seconds / total_feature_events,
      0
    ),
    beta_feature_share = if_else(
      total_feature_events > 0,
      beta_events / total_feature_events,
      0
    )
  )

# 2.3 Support tickets aggregates per account
tickets_agg <- support_tickets %>%
  mutate(
    submitted_at = ymd_hms(submitted_at),
    closed_at    = ymd_hms(closed_at)
  ) %>%
  group_by(account_id) %>%
  summarise(
    n_tickets              = n_distinct(ticket_id),
    avg_resolution_hours   = mean(resolution_time_hours, na.rm = TRUE),
    avg_first_response_min = mean(first_response_time_minutes, na.rm = TRUE),
    avg_satisfaction       = mean(satisfaction_score, na.rm = TRUE),
    escalation_rate        = mean(escalation_flag, na.rm = TRUE),
    .groups = "drop"
  )

# -------------------------------
# 3. Build modeling dataset
# -------------------------------
model_df <- accounts %>%
  mutate(
    signup_date  = as.Date(signup_date),
    # Turn bool churn into factor outcome
    churn_flag   = factor(if_else(churn_flag, "yes", "no")),
    industry        = as.factor(industry),
    country         = as.factor(country),
    referral_source = as.factor(referral_source),
    plan_tier       = as.factor(plan_tier)
  ) %>%
  # Join aggregations
  left_join(subs_agg,  by = "account_id") %>%
  left_join(usage_agg, by = "account_id") %>%
  left_join(tickets_agg, by = "account_id") %>%
  clean_names()

#To convert all logical columns into numeric before running xgboost
# Convert all logical columns (TRUE/FALSE) to numeric 0/1
logical_cols <- sapply(model_df, is.logical)
model_df[logical_cols] <- lapply(model_df[logical_cols], as.integer)

# How many accounts & churn rate?
model_df %>%
  count(churn_flag) %>%
  mutate(prop = n / sum(n))

# -------------------------------
# 4. Train/test split
# -------------------------------
set.seed(42)
churn_split <- initial_split(model_df, prop = 0.75, strata = churn_flag)
churn_train <- training(churn_split)
churn_test  <- testing(churn_split)

# -------------------------------
# 5. Modeling recipe (feature prep)
# -------------------------------

# We'll create a numeric "days since min signup" feature
min_signup_date <- min(churn_train$signup_date, na.rm = TRUE)

churn_recipe <- recipe(churn_flag ~ ., data = churn_train) %>%
  # Treat IDs as identifiers, not predictors
  update_role(account_id, account_name, new_role = "id") %>%
  # Create a numeric signup age feature
  step_mutate(
    signup_days_since_min = as.numeric(signup_date - min_signup_date)
  ) %>%
  step_rm(signup_date) %>%
  # Impute missing numeric & categorical predictors
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # One-hot encode categoricals
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  # Remove zero-variance predictors
  step_zv(all_predictors())

# Check the recipe
churn_recipe

# -------------------------------
############################################################
# 6. Define three models: Logistic, Random Forest, XGBoost
############################################################

# 6a. Logistic regression (baseline)
log_spec <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

log_wf <- workflow() %>%
  add_model(log_spec) %>%
  add_recipe(churn_recipe)

# 6b. Random Forest
library(ranger)

rf_spec <- rand_forest(
  trees = 500            # more trees usually = more stable
  # mtry and min_n left as defaults for simplicity
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(churn_recipe)

# 6c. XGBoost (gradient boosting)
library(xgboost)

xgb_spec <- boost_tree(
  trees        = 500,
  learn_rate   = 0.05,   # smaller = slower but more stable learning
  tree_depth   = 4,
  min_n        = 5,
  loss_reduction = 0.0,
  sample_size  = 0.8,
  
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(churn_recipe)

############################################################
# 7. Fit all three models on the training data
############################################################

set.seed(42)
log_fit <- fit(log_wf, data = churn_train)

set.seed(42)
rf_fit  <- fit(rf_wf,  data = churn_train)

set.seed(42)
xgb_fit <- fit(xgb_wf, data = churn_train)

############################################################
# 8. Evaluate all three models on the test set
############################################################

# Helper function to get metrics, AUC, confusion matrix, top 10 churn risk
evaluate_model <- function(fitted_wf, test_data, model_name = "model") {
  preds <- fitted_wf %>%
    predict(test_data, type = "prob") %>%
    bind_cols(predict(fitted_wf, test_data, type = "class")) %>%
    bind_cols(test_data %>% select(account_id, churn_flag))
  
  # preds has columns: .pred_no, .pred_yes, .pred_class, account_id, churn_flag
  
  cat("\n==============================\n")
  cat("Results for:", model_name, "\n")
  cat("==============================\n\n")
  
  # Overall metrics
  model_metrics <- metrics(preds, truth = churn_flag, estimate = .pred_class)
  print(model_metrics)
  
  # ROC AUC
  model_auc <- roc_auc(preds, truth = churn_flag, .pred_yes)
  cat("\nROC AUC:\n")
  print(model_auc)
  
  # Confusion matrix
  cat("\nConfusion matrix:\n")
  print(conf_mat(preds, truth = churn_flag, estimate = .pred_class))
  
  # Top 10 churn risk accounts
  top_churn <- preds %>%
    arrange(desc(.pred_yes)) %>%
    slice_head(n = 10)
  
  cat("\nTop 10 accounts by predicted churn probability:\n")
  print(top_churn)
  
  invisible(list(
    preds   = preds,
    metrics = model_metrics,
    auc     = model_auc,
    top10   = top_churn
  ))
}

# Run evaluation for each model
log_results <- evaluate_model(log_fit,  churn_test, model_name = "Logistic regression")
rf_results  <- evaluate_model(rf_fit,   churn_test, model_name = "Random Forest")
xgb_results <- evaluate_model(xgb_fit,  churn_test, model_name = "XGBoost")






