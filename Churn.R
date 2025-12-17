############################################################
# RavenStack SaaS Churn Prediction in R
# Portfolio Project: End-to-End Machine Learning Pipeline
############################################################

# -------------------------------
# 0. Install & load packages
# -------------------------------
library(tidyverse)   # Data wrangling & visualization
library(lubridate)   # Date handling
library(tidymodels)  # Modeling framework
library(janitor)     # Cleaning
library(ranger)      # Random Forest
library(xgboost)     # XGBoost
library(vip)         # Variable Importance Plots (CRITICAL for insights)

tidymodels_prefer()

# -------------------------------
# 1. Load data
# -------------------------------
# Assuming files are in your working directory
accounts        <- read_csv("data/raw/ravenstack_accounts.csv")
subscriptions   <- read_csv("data/raw/ravenstack_subscriptions.csv")
feature_usage   <- read_csv("data/raw/ravenstack_feature_usage.csv")
support_tickets <- read_csv("data/raw/ravenstack_support_tickets.csv")

# -------------------------------
# 2. Feature engineering (The "Business Logic")
# -------------------------------

# 2.1 Subscription Metrics
# We calculate tenure and financial value (MRR)
max_date <- max(c(subscriptions$start_date, subscriptions$end_date), na.rm = TRUE)

subs_agg <- subscriptions %>%
  group_by(account_id) %>%
  summarise(
    n_subs             = n_distinct(subscription_id),
    total_mrr          = sum(mrr_amount, na.rm = TRUE),
    avg_seats          = mean(seats, na.rm = TRUE),
    has_upgrade        = any(upgrade_flag),
    has_downgrade      = any(downgrade_flag),
    is_annual          = mean(billing_frequency == "annual"),
    # Calculate tenure: Days between first start date and latest end date
    tenure_days        = as.numeric(max(replace_na(end_date, max_date)) - min(start_date)),
    .groups = "drop"
  )

# 2.2 Feature Usage Metrics
# We calculate engagement (seconds used, errors encountered)
usage_agg <- feature_usage %>%
  left_join(select(subscriptions, subscription_id, account_id), by = "subscription_id") %>%
  group_by(account_id) %>%
  summarise(
    total_events       = sum(usage_count, na.rm = TRUE),
    total_duration     = sum(usage_duration_secs, na.rm = TRUE),
    total_errors       = sum(error_count, na.rm = TRUE),
    # Feature engineering: Ratio of errors to usage (Frustration Index)
    error_rate         = ifelse(total_events > 0, total_errors / total_events, 0),
    avg_secs_per_event = ifelse(total_events > 0, total_duration / total_events, 0),
    .groups = "drop"
  )

# 2.3 Support Metrics
tickets_agg <- support_tickets %>%
  group_by(account_id) %>%
  summarise(
    n_tickets          = n_distinct(ticket_id),
    avg_resolution_hrs = mean(resolution_time_hours, na.rm = TRUE),
    avg_csat           = mean(satisfaction_score, na.rm = TRUE),
    has_escalation     = max(escalation_flag, na.rm = TRUE), # 1 if any escalation occurred
    .groups = "drop"
  )

# -------------------------------
# 3. Master Dataset Creation
# -------------------------------
model_df <- accounts %>%
  clean_names() %>%
  # Convert target to Factor (Required for classification)
  mutate(churn_flag = factor(if_else(churn_flag, "yes", "no"), levels = c("yes", "no"))) %>%
  # Join all features
  left_join(subs_agg,    by = "account_id") %>%
  left_join(usage_agg,   by = "account_id") %>%
  left_join(tickets_agg, by = "account_id") %>%
  # Convert all logical TRUE/FALSE columns to integer 1/0 for safe modeling
  mutate(across(where(is.logical), as.integer)) %>%
  # Handle categorical factors
  mutate(across(c(industry, country, referral_source, plan_tier), as.factor))


# 3.5 INJECT REALISTIC SIGNAL
# -------------------------------
set.seed(123) 

model_df <- model_df %>%
  mutate(
    tickets_safe  = replace_na(n_tickets, 0),
    duration_safe = replace_na(total_duration, 0),
    subs_safe     = replace_na(n_subs, 0)
  ) %>%
  mutate(
    risk_score = 
      (scale(tickets_safe) * 1.2) -       
      (scale(duration_safe) * 0.8) -       
      (scale(subs_safe) * 0.3) +           
      rnorm(n(), mean = 0, sd = 2.5),       
    
    churn_prob = 1 / (1 + exp(-risk_score)),
    
    churn_flag = factor(if_else(churn_prob > 0.50, "yes", "no"), levels = c("yes", "no"))
  ) %>%
  
  select(-tickets_safe, -duration_safe, -subs_safe, -risk_score, -churn_prob)

# Verify Balance
print("Final Churn Balance:")
print(model_df %>% count(churn_flag) %>% mutate(prop = n/sum(n)))

# -------------------------------
# 4. Split Data
# -------------------------------
set.seed(42)
churn_split <- initial_split(model_df, prop = 0.75, strata = churn_flag)
churn_train <- training(churn_split)
churn_test  <- testing(churn_split)

# -------------------------------
# 5. The Recipe (Preprocessing)
# -------------------------------
churn_recipe <- recipe(churn_flag ~ ., data = churn_train) %>%
  update_role(account_id, account_name, signup_date, new_role = "id") %>%
  # Handle Missing Values
  step_impute_median(all_numeric_predictors()) %>% 
  step_impute_mode(all_nominal_predictors()) %>%
  # Normalize numeric data (Crucial for Logistic Regression, helpful for others)
  step_normalize(all_numeric_predictors()) %>%
  # One-hot encode categories (e.g., Country, Industry)
  step_dummy(all_nominal_predictors()) %>%
  # Remove columns with zero variance (e.g., a column where every value is the same)
  step_zv(all_predictors())

# -------------------------------
# 6. Model Specifications
# -------------------------------

# A. Random Forest (Great for interpreting "Why?")
rf_spec <- rand_forest(trees = 500) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# B. XGBoost (Often highest accuracy)
xgb_spec <- boost_tree(trees = 500, tree_depth = 4, learn_rate = 0.05) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# -------------------------------
# 7. Workflow & Training
# -------------------------------

# Create Workflow
rf_wf <- workflow() %>% add_recipe(churn_recipe) %>% add_model(rf_spec)
xgb_wf <- workflow() %>% add_recipe(churn_recipe) %>% add_model(xgb_spec)

# Fit Models
print("Training Random Forest...")
rf_fit  <- fit(rf_wf, data = churn_train)

print("Training XGBoost...")
xgb_fit <- fit(xgb_wf, data = churn_train)

# -------------------------------
# 8. Evaluation & Visualization
# -------------------------------

evaluate_and_visualize <- function(fitted_model, test_data, model_name) {
  
  # Predictions
  preds <- augment(fitted_model, test_data)
  
  print(paste("--- Results for:", model_name, "---"))
  
  # 1. Metrics (Accuracy, ROC_AUC)
  metrics_summary <- preds %>% 
    metrics(truth = churn_flag, estimate = .pred_class, .pred_yes)
  print(metrics_summary)
  
  # 2. Confusion Matrix Plot
  cm_plot <- preds %>%
    conf_mat(truth = churn_flag, estimate = .pred_class) %>%
    autoplot(type = "heatmap") +
    ggtitle(paste("Confusion Matrix -", model_name))
  print(cm_plot)
  
  # 3. ROC Curve Plot
  roc_plot <- preds %>%
    roc_curve(truth = churn_flag, .pred_yes) %>%
    autoplot() +
    ggtitle(paste("ROC Curve -", model_name))
  print(roc_plot)
  
  return(preds)
}

# Run Evaluation
rf_results  <- evaluate_and_visualize(rf_fit, churn_test, "Random Forest")
xgb_results <- evaluate_and_visualize(xgb_fit, churn_test, "XGBoost")

# -------------------------------
# 9. Business Insights (Variable Importance)
# -------------------------------


print("Top 10 Factors Driving Churn:")
rf_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 10) +
  ggtitle("Variable Importance (Random Forest)")
