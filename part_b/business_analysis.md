# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable**

The target variable is **items sold (sales volume)** — specifically, the number of items sold in a given store during a given month under a given promotion type.

**Candidate Input Features**

| Category | Features |
|---|---|
| Store attributes | Store ID, location type (urban/semi-urban/rural), store size (sq ft), monthly footfall, local competition density, customer demographic profile (age band, income index) |
| Promotion | Promotion type (one-hot encoded: Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points) |
| Temporal | Month of year, year, is_weekend proportion in month, festival flags, season (Spring/Summer/Autumn/Winter) |
| Historical | Rolling average items sold (prior 3 months), prior-month items sold, same-month-prior-year items sold |
| Interaction | Store × Promotion historical performance (mean items sold per promotion type per store) |

**Type of ML Problem and Justification**

This is a **supervised regression** problem framed as a **recommendation** task. Concretely:

- For each (store, month) pair, all five promotion options are scored using the regression model — i.e., the model predicts *items sold* for each promotion option — and the promotion yielding the highest predicted volume is recommended.
- It is **regression** (not classification) because the outcome is a continuous numeric quantity (items sold), and the ranking of promotions depends on the magnitude of predicted volume, not just a category label.
- It is **supervised** because historical records contain the actual items sold under each promotion, providing labelled training examples.

An alternative framing as a **multi-class classification** (predicting which promotion is "best") is weaker because it discards the magnitude of the difference between promotions, making model evaluation and business trade-offs harder to communicate.

---

### B1(b) — Why Items Sold is a Better Target than Revenue

**The problem with revenue as a target variable**

Revenue is the product of items sold and price, and price varies across promotions. For example:

- A **Flat Discount** reduces the unit price, so revenue can appear lower even if more items were sold.
- A **BOGO** promotion effectively halves the average selling price per item, but may double unit volume.
- A **Free Gift** adds cost without changing the ticket price, inflating revenue while reducing actual margin.

Using revenue as the target causes the model to conflate two distinct signals — *how many units moved* and *at what price* — making it difficult to identify which promotions genuinely drive customer purchase behaviour. A model trained on revenue may systematically recommend high-price promotions (e.g., Category-Specific Offers on premium items) simply because they have large ticket sizes, not because they are effective at driving volume.

**Why items sold is more reliable**

Items sold is a **clean, direct signal** of customer response to a promotion. It is not distorted by pricing strategy, discounting depth, or product mix. The model learns what promotions cause customers to buy more — which is the stated business objective.

**Broader principle: target variable should directly encode the decision objective**

This illustrates the principle that **the target variable must be causally aligned with what the business wants to change, not a downstream financial metric that is influenced by confounding variables**. Revenue, profit, and ROI are often tempting targets because they are what stakeholders care about, but they compound multiple factors. Choosing a target that is closer to the raw behavioural signal (volume, clicks, conversions) produces more actionable and interpretable models. Secondary financials can then be computed from the model's volume predictions using separate pricing and cost data.

---

### B1(c) — Modelling Strategy: Store Segmentation over a Single Global Model

**Problem with a single global model**

A single model trained across all 50 stores assumes that the relationship between features and items sold is homogeneous. In practice, an urban flagship store with high footfall, younger demographics, and dense competition will respond to promotions very differently from a small rural outlet. A global model will average these patterns, producing poor predictions for stores at either extreme and systematically misallocating promotions.

**Proposed alternative: Hierarchical or Clustered Modelling**

The recommended strategy is a **two-level approach**:

1. **Cluster stores into segments** (e.g., 3–5 clusters) using store-level features: location type, size, footfall, competition density, and demographic profile. Apply k-means or hierarchical clustering on these static attributes.

2. **Train one model per cluster**. Each cluster model learns the promotion-response patterns that are characteristic of stores in that segment. This gives each model enough training data (multiple stores × 36 months) to generalise, while remaining specific to stores that actually behave similarly.

**Why this is better than fully separate per-store models**

Fully per-store models (50 individual models) are appealing but suffer from **data sparsity** — each store has only 36 monthly observations, which is insufficient for a model with many features. Cluster models pool data across similar stores to produce stable estimates, while still capturing the structural differences between urban, semi-urban, and rural contexts.

**Additional enhancement: store-level random effects**

Within each cluster model, store ID can be included as a categorical feature (or as a random effect in a mixed-effects model) to capture residual store-specific variation not explained by the cluster-level features. This provides a clean middle ground between a single global model and fully separate per-store models.

---

## B2. Data and EDA Strategy

### B2(a) — Joining Tables and Dataset Grain

**Table Descriptions and Join Logic**

| Table | Key Columns | Join Type |
|---|---|---|
| `transactions` | store_id, date, item_id, quantity, revenue | Base table |
| `store_attributes` | store_id, location_type, size, footfall, competition_density, demographics | LEFT JOIN on `store_id` |
| `promotion_details` | promotion_id, promotion_type, start_date, end_date | JOIN on `store_id + date` (map dates to active promotion) |
| `calendar` | date, is_weekend, festival_flag, month, year, season | LEFT JOIN on `date` |

**Join sequence:**

1. Aggregate `transactions` to `(store_id, year, month)` grain — summing items sold and revenue.
2. Map each `(store_id, year, month)` to the single active promotion for that period using `promotion_details` (assuming one promotion per store per month).
3. LEFT JOIN `store_attributes` on `store_id`.
4. LEFT JOIN aggregated `calendar` features (e.g., number of weekend days in month, number of festival days in month) on `(year, month)`.

**Grain of the final modelling dataset**

> **One row = one store × one month × one promotion**

Each row answers: *"In store X, during month Y, promotion Z was run — how many items were sold?"*

**Aggregations performed before modelling**

- `items_sold`: SUM of quantity from transactions
- `total_revenue`: SUM of revenue
- `avg_transaction_size`: total_revenue / COUNT(transactions)
- `weekend_days_in_month`: COUNT of weekend days from calendar
- `festival_days_in_month`: COUNT of festival-flagged days
- `rolling_3m_avg_items_sold`: window function over prior 3 months per store
- `same_month_prior_year_items_sold`: lag feature for seasonality

---

### B2(b) — EDA Strategy

**Analysis 1: Promotion performance distribution by store type**

Plot boxplots of items sold for each promotion type, faceted by location type (urban, semi-urban, rural). Look for whether the ranking of promotions is consistent across location types or if, for example, BOGO performs best in urban stores but poorly in rural ones. **Influence:** Confirms whether location-segmented modelling is necessary; identifies which interaction features (promotion × location type) to engineer.

**Analysis 2: Time-series decomposition per store**

Plot monthly items sold over 36 months for a sample of stores (one per location type). Look for trend, seasonal peaks (e.g., festival months, end-of-year), and structural breaks. **Influence:** Determines which lag and rolling-average features are most informative; identifies whether year-over-year growth trends need to be modelled explicitly or detrended.

**Analysis 3: Promotion frequency heatmap**

Create a heatmap of (store × promotion type) showing how often each promotion was deployed at each store. Look for stores where only one or two promotions were ever used — these stores will have sparse counterfactual data, making causal inference harder. **Influence:** Flags stores where the model will have high uncertainty for untested promotions; may justify using regularisation or hierarchical priors for such stores.

**Analysis 4: Correlation matrix and multicollinearity check**

Compute a correlation matrix across numeric features: footfall, store size, competition density, festival days, rolling averages. Look for features with correlation > 0.85 (potential multicollinearity). **Influence:** Highly correlated features can destabilise linear models and inflate variance in tree-based models' feature importances; findings will inform feature selection or dimensionality reduction (e.g., PCA on demographic variables).

**Analysis 5: Promotion lift analysis**

For each promotion type, compute a simple lift score: mean items sold during promotion versus mean items sold in months with no promotion (baseline), segmented by store cluster. Look for whether the lift is statistically significant (t-test) and consistent. **Influence:** Provides a sanity check that the data supports the premise that promotions drive volume; also reveals which promotions have the most uncertain lift, guiding model confidence calibration.

---

### B2(c) — Addressing Promotion Imbalance (80% Non-Promotion Transactions)

**How imbalance affects the model**

If 80% of rows in the training dataset correspond to no-promotion months, the model will be dominated by the no-promotion signal during training. It will learn the baseline demand pattern very well but will have limited data to distinguish between the five promotion types. Practically, this means:

- Predictions for promotion months may regress toward the no-promotion mean (attenuation bias).
- Feature importances related to promotion type will be underestimated relative to store-level demand drivers.
- The model may perform well on aggregate metrics (because 80% of data is no-promotion) while being poorly calibrated precisely where the decision is being made (promotion months).

**Steps to address imbalance**

1. **Stratified sampling:** When constructing train/validation splits, stratify by `promotion_type` to ensure all five promotion types are proportionally represented in both splits.

2. **Separate models for promotion vs. no-promotion:** Train one model to predict baseline (no-promotion) demand and a second model to predict items sold conditional on a given promotion type. The lift (difference between the two predictions) becomes the basis for promotion recommendation. This cleanly separates the two learning problems.

3. **Sample weighting:** In a single unified model, apply higher sample weights to promotion-month rows. Most tree-based frameworks (XGBoost, LightGBM) and scikit-learn estimators support `sample_weight` directly.

4. **Re-frame as lift prediction:** Rather than predicting absolute items sold, engineer a target variable that is the *lift ratio* (items sold in promotion month / rolling 3-month baseline). This transforms the target to be promotion-centric and removes most of the imbalance effect, since all promotion-month rows now have a meaningful lift signal.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Evaluation Metrics

**Why random splits are inappropriate**

This dataset has a strong **temporal dependency**: items sold in month *t* are correlated with items sold in months *t−1*, *t−2*, etc. (seasonal patterns, trends, rolling averages). A random split will leak future information into the training set — a row from December 2023 could appear in training while a row from January 2023 appears in test. This means:

- Rolling average features computed on "future" data would contaminate training.
- The model will appear to perform well by memorising temporal patterns rather than generalising to genuinely unseen future periods.
- Model performance on the test set will be artificially optimistic and will not reflect real-world deployment performance.

**Recommended split: time-based (walk-forward)**

Given 36 months of data across 50 stores:

- **Training set:** Months 1–24 (first two years, all stores)
- **Validation set:** Months 25–30 (next six months — used for hyperparameter tuning)
- **Test set:** Months 31–36 (final six months — held out until final evaluation)

This mirrors the real deployment scenario: the model is always trained on past data and evaluated on genuinely unseen future months.

For ongoing monitoring, use a **rolling walk-forward validation**: retrain on all data up to month *t*, evaluate on month *t+1*, and roll forward. Average the evaluation metrics across all such folds.

**Evaluation metrics**

| Metric | Formula | Business Interpretation |
|---|---|---|
| **RMSE** (Root Mean Squared Error) | √(mean((ŷ − y)²)) | Average magnitude of prediction error in units of items sold. Penalises large errors heavily — relevant because a large mis-prediction can lead to significant inventory over/under-stocking. |
| **MAE** (Mean Absolute Error) | mean(\|ŷ − y\|) | Average absolute error in items sold. More robust to outliers than RMSE; good for communicating to business stakeholders ("on average, predictions are off by X units"). |
| **MAPE** (Mean Absolute Percentage Error) | mean(\|ŷ − y\| / y) × 100 | Percentage error; useful for comparing performance across stores of different sizes (a 50-unit error means very different things for a 100-unit store vs. a 1,000-unit store). Caution: undefined when y = 0. |
| **Promotion ranking accuracy** | % of store-months where model correctly identifies the highest-volume promotion | Most directly aligned with the business objective: even if absolute volume predictions are imperfect, the model is useful if it correctly ranks promotions. |

---

### B3(b) — Feature Importance to Explain Differing Recommendations

**Scenario:** Model recommends Loyalty Points Bonus for Store 12 in December but Flat Discount in March.

**Step 1: Compute SHAP values for each prediction**

Use SHAP (SHapley Additive exPlanations) to decompose the predicted items sold for each promotion option into the contribution of each feature. For Store 12 in December, run SHAP for all five promotion options and compare their predicted values. Identify which features drive the higher predicted score for Loyalty Points Bonus over Flat Discount in December.

**Step 2: Identify the differentiating features**

Expected findings based on domain logic:

- **December:** `festival_days_in_month` will be high (Christmas, New Year), and historical data likely shows that customers in December are already buying gifts and are receptive to accumulating Loyalty Points for use in January. The SHAP value for `festival_days_in_month × Loyalty Points` interaction will be large and positive.
- **March:** Festival days drop to near zero, footfall returns to baseline, and historical data may show that customers respond better to immediate price incentives (Flat Discount) in off-season months.

**Step 3: Construct a visual explanation for the marketing team**

Produce a **waterfall chart** (standard SHAP output) for each recommendation, showing the base prediction and the additive contribution of each feature. Overlay the two charts side by side (December vs. March) for Store 12. Annotate in plain language:

> *"In December, the model predicts Loyalty Points Bonus will sell 15% more units than a Flat Discount. The main drivers are: (1) December has 4 festival days, during which our data shows customers respond strongly to loyalty incentives; (2) Store 12's demographic profile (high repeat-purchase rate) amplifies this effect. In March, with no festivals and lower footfall, customers respond better to immediate savings."*

This translates the model's mathematics into a causal narrative that is actionable for the marketing team without requiring them to understand SHAP values directly.

---

### B3(c) — End-to-End Deployment Process

**1. Saving the trained model**

After final training on all available data (months 1–36), serialise the model artefact using `joblib` (for scikit-learn / XGBoost) or the native save format (e.g., `model.save_model()` for XGBoost/LightGBM). Save the following artefacts together in a versioned model registry (e.g., MLflow, AWS SageMaker Model Registry):

- Trained model binary
- Feature preprocessing pipeline (scaler, one-hot encoders, imputers)
- Feature names and expected schema
- Training metadata (data range used, validation metrics, SHAP baseline values)
- Model version tag and timestamp

**2. Preparing new monthly data**

At the start of each month *t+1*, an automated data pipeline (e.g., Apache Airflow DAG) executes the following:

1. Pull transactions, store attributes, and calendar data for month *t* from the data warehouse.
2. Apply the same aggregation logic used during training: compute monthly items sold, festival days, weekend count.
3. Update rolling lag features: `rolling_3m_avg` now uses months *t−2*, *t−1*, *t*; `same_month_prior_year` pulls from month *t−12*.
4. Apply the saved preprocessing pipeline (same scaler/encoder fitted during training) to the new feature matrix.
5. For each of the 50 stores, construct 5 feature vectors — one per promotion type — with all other features held constant.

**3. Generating recommendations**

Pass the 250 feature vectors (50 stores × 5 promotions) through the saved model to obtain predicted items sold for each option. For each store, select the promotion with the highest predicted value as the recommendation. Output a ranked recommendation table (all 5 promotions with predicted volumes) for the marketing team's review.

**4. Model monitoring and drift detection**

| Monitoring Type | Method | Alert Threshold |
|---|---|---|
| **Prediction drift** | Track the distribution of predicted items sold each month (mean, std, percentiles). Compare to the historical distribution using a KS test or PSI (Population Stability Index). | PSI > 0.2 triggers review |
| **Feature drift** | Monitor input feature distributions (footfall, festival days, competition density) each month using PSI or Jensen-Shannon divergence. | PSI > 0.25 on key features |
| **Actuals vs. predictions** | Once month-end actuals are available (~30 days after recommendations are made), compute MAE and MAPE for that month. Track on a rolling 3-month basis. | MAPE > 20% for 2 consecutive months |
| **Promotion rank accuracy** | Track whether the recommended promotion was actually the best-performing one in hindsight (requires waiting for actuals). | Rank accuracy < 60% over rolling quarter |

**5. Retraining trigger**

Retraining is triggered if any of the following occur:

- Monitoring alerts fire on two consecutive months (gradual drift).
- A known structural change occurs: new store opens, a promotion type is retired, a major competitive event changes market dynamics.
- Scheduled full retraining every 6 months regardless of drift signals, to incorporate the most recent data and prevent silent degradation.

When retraining, expand the training window to include all available data up to the present. Use the same walk-forward validation to confirm the retrained model outperforms the current production model before deployment. Store the new version in the model registry with clear version history.

---


