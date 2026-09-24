import pandas as pd
from sklearn.preprocessing import StandardScaler #introduced at step 29
from sklearn.compose import ColumnTransformer #introduced at step 30

#introduced at step 31
import numpy as np
from scipy.sparse import issparse

#introduced at step 33
from sklearn.linear_model import LogisticRegression

#introduced at step 35
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


#step 1: load oirginal incident dataset 
data = pd.read_csv("data/incident_event_log.csv")

#initial verification
print(f"Inital Total rows: {len(data)}")
print(f"Initial Unique incidents: {data["number"].nunique()}")

#inspect few orginal time stamp values before converting them
print("timestamp values: \n")
print(data[["number", "opened_at", "sys_updated_at", "incident_state"]].head(10))


# Step 2: Convert timestamp strings into datetime values
data["opened_at"] = pd.to_datetime(data["opened_at"], dayfirst=True)
data["sys_updated_at"] = pd.to_datetime(data["sys_updated_at"], dayfirst=True)

# Sort records chronologically by their update timestamp
data = data.sort_values(by="sys_updated_at", ascending=True, kind="stable")

# Keep only the earliest available record for each incident
data = data.drop_duplicates(subset="number", keep="first")

# Reset row indexes after removing duplicates
data = data.reset_index(drop=True)

# Verify the resulting dataset
print("\nTotal rows:", len(data))
print("Unique incidents:", data["number"].nunique())

# Examine the earliest recorded state of each incident
print("\nIncident state distribution:")
print(data["incident_state"].value_counts(dropna=False))

print("\nEarliest recorded events:")
print(data[["number", "opened_at", "sys_updated_at", "incident_state"]].head(10))

# Step 3: Investigate incident timestamps (since some data are updated before ticket created/opened)

# Calculate elapsed time between opening and the first recorded update.
data["minutes_since_open"] = (data["sys_updated_at"] - data["opened_at"]).dt.total_seconds()/60

#Identify incidents updated before their opening timestamp.
negative_time =  data[data["minutes_since_open"] < 0]
print("\nIncidents with negative elapsed time:", len(negative_time))
print(negative_time[["number", "opened_at", "sys_updated_at", "minutes_since_open"]].head(5))

#Find incidents whose first update was within 60 minutes of opening.
within_one_hour = data[(data["minutes_since_open"] >= 0) & (data["minutes_since_open"] <= 60)]
print("\nIncidents updated within one hour:", len(within_one_hour))

#Investigate incidents whose earliest recorded state is Resolved.
resolved_incidents = data[data["incident_state"] == "Resolved"]
print("\nInitially resolved incidents:", len(resolved_incidents))

#Check how many resolved incidents have an early update.
resolved_within_hour = resolved_incidents[(resolved_incidents["minutes_since_open"] >= 0) &
                                          (resolved_incidents["minutes_since_open"] <= 60)]
print("\nResolved incidents updated within one hour:", len(resolved_within_hour))

#Inspect examples of incidents already marked Resolved.
print("\nSample resolved incidents:")
print(resolved_incidents[["number", "opened_at", "sys_updated_at", "minutes_since_open"]].head(5))

#step 4: Investigate the resolution timestamps

#Convert the resolution timestamp to datetime.
data["resolved_at"] = pd.to_datetime(data["resolved_at"], dayfirst=True, errors="coerce")

# Select incidents whose earliest recorded state is Resolved.
resolved_incidents = data[data["incident_state"] == "Resolved"].copy()

# Calculate the actual resolution duration in minutes.
resolved_incidents["resolution_minutes"] = (resolved_incidents["resolved_at"] - 
                                            resolved_incidents["opened_at"]).dt.total_seconds() / 60

# Examine the resolution durations.
print("\nInitially resolved incidents: ", len(resolved_incidents))
print("\nResolved in zero minutes:: ", (resolved_incidents["resolution_minutes"] == 0).sum())
print("Resolved in more than zero minutes: ", (resolved_incidents["resolution_minutes"] > 0).sum())
print("Negative resolution duration: ", (resolved_incidents["resolution_minutes"] < 0).sum())
print("Missing resolution duration: ", resolved_incidents["resolution_minutes"].isna().sum())

# Inspect actual timestamps for the first 10 incidents.
print("\nSample resolved incidents:")
print(
    resolved_incidents[
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "resolved_at",
            "resolution_minutes"
        ]
    ].head(10).to_string(index=False)
)

# Investigate incidents whose earliest recorded state is New.
new_incidents = data[data["incident_state"] == "New"]

print("\nInitially New incidents:", len(new_incidents))
print("New incidents with a resolution timestamp:",new_incidents["resolved_at"].notna().sum())
print("New incidents with a future resolution timestamp:",(new_incidents["resolved_at"] > new_incidents["opened_at"]).sum())

#from the observation, found many missing resolution duration
# Step 5: Recover missing resolution timestamps from incident history

# Load the original dataset containing all event records.
history = pd.read_csv("data/incident_event_log.csv")

# Convert resolution timestamps into datetime values.
history["resolved_at"] = pd.to_datetime(history["resolved_at"], dayfirst=True, errors="coerce")

# Group all historical records by incident ID.
# max() returns the latest available resolution timestamp for each incident.
resolution_history = history.groupby("number")["resolved_at"].max()

# Match each ticket in our prepared dataset to its historical resolution timestamp.
data["history_resolved_at"] = data["number"].map(resolution_history)

# Select incidents initially marked Resolved with a missing resolution timestamp.
missing_resolution = data[(data["incident_state"] == "Resolved") & (data["resolved_at"].isna())].copy()

print("\nInitially resolved incidents with missing timestamps: ", len(missing_resolution))
print(
    "\nRecovered resolution timestamps:",
    missing_resolution["history_resolved_at"].notna().sum()
)
print(
    "\nStill missing resolution timestamps:",
    missing_resolution["history_resolved_at"].isna().sum()
)
print("\nSample incidents:")

print(
    missing_resolution[
        ["number", "resolved_at", "history_resolved_at"]
    ].head(5).to_string(index=False)
)

# Step 6: Validate resolution durations across all incidents

# Calculate resolution duration in hours.
# The duration is our future regression target (y).
data["resolution_hours"] = (data["history_resolved_at"] - data["opened_at"]).dt.total_seconds() /3600

# Identify valid, negative, and missing durations.
valid_resolution = data["resolution_hours"] > 0
negative_resolution = data["resolution_hours"] < 0
missing_resolution = data["resolution_hours"].isna()

print("\nValid resolution durations:", valid_resolution.sum())
print("Negative resolution durations:", negative_resolution.sum())
print("Missing resolution durations:", missing_resolution.sum())

# Check target availability for each initial incident state.
# This is a data-quality report, not a list of model input features.
data["valid_resolution"] = valid_resolution
data["missing_resolution"] = missing_resolution

print("\nResolution availability by initial state:")
print(
    data.groupby("incident_state")[
        ["valid_resolution", "missing_resolution"]
    ].sum()
)

# Examine five incidents with valid resolution durations.
print("\nSample incidents with valid resolution durations:")
print(
    data.loc[
        valid_resolution,
        [
            "number",
            "opened_at",
            "history_resolved_at",
            "resolution_hours"
        ]
    ].head(5).to_string(index=False)
)

# Step 7: Validate every resolution duration
# Count positive, zero, negative, and missing durations.
positive_count = (data["resolution_hours"] > 0).sum()
zero_count = (data["resolution_hours"] == 0).sum()
negative_count = (data["resolution_hours"] < 0).sum()
missing_count = (data["resolution_hours"].isna().sum())

print("\nPositive durations:", positive_count)
print("Zero durations:", zero_count)
print("Negative durations:", negative_count)
print("Missing durations:", missing_count)

# Verify that every incident belongs to one of these categories.
total = (positive_count + negative_count + zero_count + missing_count)
print("\nTotal accounted incidents:", total)
print("Actual dataset rows:", len(data))

# Inspect incidents with a resolution duration of zero hours.
zero_incidents = data[data["resolution_hours"] == 0]
print("\nSample zero-hour incidents:")
print(
    zero_incidents[
        [
            "number",
            "opened_at",
            "history_resolved_at",
            "incident_state",
            "resolution_hours"
        ]
    ].head(5).to_string(index=False)
)

#step 8: Verify resolution timing relative to input snapshot

# Keep incidents that are not initially Resolved and whose earliest update is not before opening.
candidates = data[(data["incident_state"]  != "Resolved") & (data["minutes_since_open"] >= 0)].copy()

# Resolution occurred after our input snapshot.
future_resolution = (candidates["history_resolved_at"] > candidates["sys_updated_at"])

# Historical resolution timestamp is unavailable.
missing_resolution = (candidates["history_resolved_at"].isna())

# Resolution occurred at or before our snapshot.
already_resolved = (candidates["history_resolved_at"].notna() & (candidates["history_resolved_at"] <= candidates["sys_updated_at"]))

print("\nTotal candidate incidents:", len(candidates))
print(
    "Resolution after snapshot:",
    future_resolution.sum()
)
print(
    "Missing resolution timestamps:",
    missing_resolution.sum()
)
print(
    "Resolution at or before snapshot:",
    already_resolved.sum()
)

print(
    "\nTotal accounted candidates:", future_resolution.sum() + missing_resolution.sum() + already_resolved.sum()
)

# Examine potentially invalid prediction snapshots.
print("\nIncidents resolved at or before snapshot:")

print(
    candidates.loc[
        already_resolved,
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "history_resolved_at",
            "incident_state"
        ]
    ].head(5).to_string(index=False)
)

# Check how many candidate snapshots were recorded
# more than one hour after the ticket was opened.
print(
    "\nCandidate snapshots after 60 minutes:",
    (candidates["minutes_since_open"] > 60).sum()
)

#step 9: Investigate late incident snapshots

# Select candidate incidents whose earliest available update
late_snapshots = candidates[candidates["minutes_since_open"] > 60].copy()

print("\nTotal late snapshots:", len(late_snapshots))

# Convert elapsed minutes into hours.
late_snapshots["hours_since_open"] = (late_snapshots["minutes_since_open"] / 60)

print("\nMinimum hours since opening: ", late_snapshots["hours_since_open"].min())
print("Median hours since opening: ", late_snapshots["hours_since_open"].median())
print("Maximum hours since opening: ", late_snapshots["hours_since_open"].max())

# Examine incident states among late snapshots.
print("\nLate snapshot state distribution:")
print(late_snapshots["incident_state"].value_counts(dropna=False))

# Sort late snapshots from largest to smallest elapsed time.
late_snapshots = late_snapshots.sort_values(by="hours_since_open", ascending=False)

# Display the five incidents with the largest elapsed times.
print("\nFive latest incident snapshots:")

print(
    late_snapshots[
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "incident_state",
            "hours_since_open"
        ]
    ].head(5).to_string(index=False)
)

#step 10: Establish early-snapshot eligibility

# Keep incidents whose earliest available snapshot occurred
early_candidates = candidates[candidates["minutes_since_open"] <= 60].copy()

print("\nTotal early candidates:", len(early_candidates))

# Count incidents resolved strictly after the input snapshot.
early_future_resolution = (early_candidates["history_resolved_at"] > early_candidates["sys_updated_at"])

# Count incidents without a known resolution timestamp.
early_missing_resolution = (early_candidates["history_resolved_at"].isna())

# Count incidents resolved at or before the input snapshot.
early_already_resolved = (early_candidates["history_resolved_at"].notna() & 
                          ( early_candidates["history_resolved_at"] <= early_candidates["sys_updated_at"]))

print("\nResolution after snapshot: ", early_future_resolution.sum())
print("Missing resolution timestamps: ", early_missing_resolution.sum())
print("Resolution at or before snapshot: ", early_already_resolved.sum())

# Verify that every early candidate is accounted for.
total_accounted = (
    early_future_resolution.sum() +
    early_missing_resolution.sum() +
    early_already_resolved.sum()
)

print("\nTotal accounted early candidates:", total_accounted)
print("Actual early candidate rows:", len(early_candidates))

#step 11: Investigate the original dataset columns

# history contains the original event-log columns.
original_columns = history.columns

print("\nOriginal dataset columns:")
print(original_columns.tolist())
print("\nNumber of original columns:", len(original_columns))

# Display only original columns, excluding the fields
# we created during our data-quality investigation.
print("\nSample early candidate records:")
print(
    early_candidates[original_columns]
    .head(3)
    .to_string(index=False)
)

# Inspect data types and missing values.
print("\nColumn data types and missing values:")

column_summary = pd.DataFrame({
    "dtype": early_candidates[original_columns].dtypes,
    "missing_count": early_candidates[original_columns].isna().sum()
})

print(column_summary.to_string())

#step 12:  Investigate missing and unknown feature values

feature_candidates = [
    "contact_type",
    "location",
    "category",
    "subcategory",
    "u_symptom",
    "caller_id",
    "opened_by",
    "cmdb_ci",
    "assignment_group",
    "assigned_to"
]

# Create an empty list to store our findings.
feature_summary = []

for column in feature_candidates:
    # Count actual missing values (NaN).
    missing_count = early_candidates[column].isna().sum()
    
    # Count values represented by the string "?".
    unknown_count = (early_candidates[column] == "?").sum()
    
    # Count distinct values, including NaN if present.
    unique_count = early_candidates[column].nunique(dropna=False)
    
    # Store the results for this column.
    feature_summary.append({
        "column": column,
        "missing_count": missing_count,
        "unknown_count": unknown_count,
        "unique_count": unique_count
    })

# Convert our results into a summary DataFrame.
feature_summary = pd.DataFrame(feature_summary)

print("\nFeature missing-value summary:")
print(feature_summary.to_string(index=False))
    
# Step 13: Investigate feature stability across incident history

# Count distinct recorded values for each feature
# within the history of each incident.
feature_variations = (history.groupby("number")[feature_candidates].nunique(dropna=False))

# Restrict the investigation to our early candidate incidents.
feature_variations = feature_variations.reindex(early_candidates["number"])

# Create an empty list to store our findings.
stability_summary = []

for col in feature_candidates:

    # An incident has a recorded feature change if
    # more than one distinct value appears in its history.
    changed_count = (feature_variations[col] > 1).sum()

    stability_summary.append({
        "column": col,
        "incidents_with_changes": changed_count
    })

# Display the results as a summary table.
stability_summary = pd.DataFrame(stability_summary)

print("\nFeature stability across incident history:")
print(stability_summary.to_string(index=False))


# Step 14: Investigate priority labels and target leakage

# Examine the distribution of priority labels
# in our early candidate incidents.
print("\nPriority distribution:")

print(early_candidates["priority"].value_counts(dropna=False))

# Investigate changes in priority, impact, and urgency.
priority_columns = ["priority", "impact", "urgency"]

# Count distinct recorded values per incident.
priority_variations = (history.groupby("number")[priority_columns].nunique(dropna=False))

# Restrict the analysis to our early candidate incidents.
priority_variations = priority_variations.reindex(early_candidates["number"])

print("\nIncidents with recorded changes:")

for col in priority_columns:
    changed_count = (
        priority_variations[col] > 1
    ).sum()

    print(col, ":", changed_count)

# Investigate the relationship between impact, urgency, and priority in our early snapshots.
priority_mapping = (early_candidates.groupby(["impact", "urgency"])["priority"].nunique(dropna=False))

print("\nDistinct priority labels per impact-urgency combination:")
print(priority_mapping.to_string())

# Step 15: Compare early-snapshot priority with last recorded priority

# Convert historical update timestamps into datetime values.
history["sys_updated_at"] = pd.to_datetime(history["sys_updated_at"], dayfirst=True)

# Sort historical events and select the last recorded event for each incident.
last_events = (history.sort_values("sys_updated_at", kind="stable")
               .drop_duplicates(subset="number", keep="last")
               .set_index("number"))

# Create a separate DataFrame for target-label investigation.
priority_audit = early_candidates[["number", "sys_updated_at", "priority"]].copy()

# Match each incident with its last recorded priority.
priority_audit["last_priority"] = (priority_audit["number"].map(last_events["priority"]))

# Match each incident with its last recorded update timestamp.
priority_audit["last_updated_at"] = (priority_audit["number"].map(last_events["sys_updated_at"]))

# Identify incidents whose priority differs between the early snapshot and the last recorded event.
priority_changed = (priority_audit["priority"] != priority_audit["last_priority"])

print("\nIncidents with different early and last priority:",priority_changed.sum())
print("Incidents with unchanged early and last priority:",(~priority_changed).sum())
print("\nSample incidents with different priorities:")
print(
    priority_audit.loc[
        priority_changed,
        [
            "number",
            "sys_updated_at",
            "priority",
            "last_updated_at",
            "last_priority"
        ]
    ].head(5).to_string(index=False)
)


# Step 16: Investigate earlier historical resolutions

# Find the earliest recorded resolution timestamp for every incident in the complete event history.
first_resolution_history = (history.groupby("number")["resolved_at"].min())

# Create a separate DataFrame for our investigation.
resolution_audit = early_candidates[["number", "opened_at", "sys_updated_at", "history_resolved_at"]].copy()

# Map the earliest recorded resolution timestamp to the corresponding incident.
resolution_audit["first_resolved_at"] = (resolution_audit["number"].map(first_resolution_history))

# Select incidents whose latest resolution occurred after the input snapshot.
future_resolution_audit = (resolution_audit["history_resolved_at"] >resolution_audit["sys_updated_at"])

# Check whether an earlier resolution was already recorded at or before that same input snapshot.
earlier_resolution = (resolution_audit["first_resolved_at"].notna() &
    (
        resolution_audit["first_resolved_at"] <=
        resolution_audit["sys_updated_at"]
    )
)

# Incidents satisfying both conditions need investigation.
suspicious_incidents = resolution_audit[future_resolution_audit & earlier_resolution]

print("\nIncidents with an earlier resolution before the snapshot:",len(suspicious_incidents))
print("\nSample incidents:")
print(
    suspicious_incidents[
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "first_resolved_at",
            "history_resolved_at"
        ]
    ].head(5).to_string(index=False)
)

#Step 17: Create separate eligible datasets
resolved_after_snapshot = (early_candidates["history_resolved_at"] > early_candidates["sys_updated_at"])

# Classification can also retain incidents without a resolution timestamp.
classification_data = early_candidates[resolved_after_snapshot | early_candidates["history_resolved_at"].isna()].copy()

# Regression requires a recorded resolution after the snapshot.
regression_data = early_candidates[resolved_after_snapshot].copy()

# Print dataset sizes.
print("\nClassification incidents:", len(classification_data))
print("Regression incidents:", len(regression_data))

# Check the priority labels available for classification.
print("\nClassification priority distribution:")
print(classification_data["priority"].value_counts())

# Verify every regression incident is included in classification.
regression_in_classification = (
    regression_data["number"]
    .isin(classification_data["number"])
    .all()
)

print("\nAll regression incidents included in classification:",regression_in_classification)

# STEP 18: Define initial model input features

classification_features = [
    "contact_type",
    "location",
    "category",
    "subcategory",
    "u_symptom"
]

# Reuse classification features and add regression-specific inputs.
regression_features = classification_features + ["impact", "urgency"]

print("\nClassification features:")
print(classification_features)
print("Total:", len(classification_features))

print("\nRegression features:")
print(regression_features)
print("Total:", len(regression_features))

# STEP 19: Define the regression target

# Calculate remaining resolution time in hours.
regression_data["remaining_resolution_hours"] = (
    (
        regression_data["history_resolved_at"]
        - regression_data["sys_updated_at"]
    ).dt.total_seconds() / 3600
)

# Review the target distribution.
print( "\nMinimum remaining hours:", regression_data["remaining_resolution_hours"].min() )
print( "Maximum remaining hours:", regression_data["remaining_resolution_hours"].max() )
print( "Median remaining hours:", regression_data["remaining_resolution_hours"].median() )

# Check for invalid target values.
print( "Missing remaining hours:", regression_data["remaining_resolution_hours"].isna().sum() )
print( "Nonpositive remaining hours:", ( regression_data["remaining_resolution_hours"] <= 0 ).sum() )


# STEP 20: Investigate remaining resolution time distribution

# Calculate key percentiles.
print("\nRemaining resolution time percentiles (hours):")
print( regression_data["remaining_resolution_hours"].quantile( [0.50, 0.90, 0.95, 0.99] ) )

# Count incidents taking longer than 30 days.
over_30_days = (regression_data["remaining_resolution_hours"] > 720 ).sum()

# Count incidents taking longer than 90 days.
over_90_days = ( regression_data["remaining_resolution_hours"] > 2160 ).sum()

print("\nIncidents exceeding 30 days:", over_30_days)
print("Incidents exceeding 90 days:", over_90_days)

# Display the five longest-running incidents.
print("\nFive incidents with the longest remaining resolution times:")
print(
    regression_data.nlargest(
        5, "remaining_resolution_hours"
    )[
        [
            "number",
            "opened_at",
            "sys_updated_at",
            "history_resolved_at",
            "remaining_resolution_hours"
        ]
    ].to_string(index=False)
)
    
# STEP 21: Investigate feature quality

# Classification feature quality
print("\nClassification feature quality:")

for feature in classification_features:
    print(
        feature,
        "| Unknown '?':", classification_data[feature].eq("?").sum(),
        "| Missing:", classification_data[feature].isna().sum(),
        "| Unique:", classification_data[feature].nunique()
    )

# Regression feature quality
print("\nRegression feature quality:")

for feature in regression_features:
    print(
        feature,
        "| Unknown '?':", regression_data[feature].eq("?").sum(),
        "| Missing:", regression_data[feature].isna().sum(),
        "| Unique:", regression_data[feature].nunique()
    )
    
# STEP 22: Standardize missing values

# Create independent working copies.
classification_clean = classification_data.copy()
regression_clean = regression_data.copy()

# Replace '?' placeholders in classification features.
for feature in classification_features:
    classification_clean[feature] = classification_clean[feature].replace(to_replace="?", value=pd.NA)

# Replace '?' placeholders in regression features.
for feature in regression_features:
    regression_clean[feature] = regression_clean[feature].replace(to_replace="?", value=pd.NA)
    
# Verify missing values in classification features.
print("\nClassification missing values:")
print( classification_clean[classification_features] .isna() .sum() )

# Verify missing values in regression features.
print("\nRegression missing values:")
print( regression_clean[regression_features] .isna() .sum() )

# STEP 23: Create time-based features

time_features = [ "opened_hour", "opened_day_of_week" ]

# Classification: extract features from the opening timestamp.
classification_clean["opened_hour"] = classification_clean["opened_at"].dt.hour
classification_clean["opened_day_of_week"] = classification_clean["opened_at"].dt.day_of_week

# Regression: extract the same time-based features.
regression_clean["opened_hour"] = regression_clean["opened_at"].dt.hour
regression_clean["opened_day_of_week"] = regression_clean["opened_at"].dt.day_of_week

# Update the input feature lists.
classification_features = classification_features + time_features
regression_features = regression_features + time_features

# Verify the updated feature lists.
print("\nClassification features:", classification_features)
print("Total:", len(classification_features))

print("\nRegression features:", regression_features)
print("Total:", len(regression_features))

# Check whether the new features contain missing values.
print("\nClassification time feature missing values:")
print(classification_clean[time_features].isna().sum())

print("\nRegression time feature missing values:")
print(regression_clean[time_features].isna().sum())

# STEP 24: Create input features (X) and targets (y)

# Classification
X_classification = classification_clean[classification_features].copy()
y_classification = classification_clean["priority"].copy()

# Regression
X_regression = regression_clean[regression_features].copy()
y_regression = regression_clean["remaining_resolution_hours"].copy()

# Verify the dataset dimensions.
print("\nClassification:")
print("X shape:", X_classification.shape)
print("y shape:", y_classification.shape)

print("\nRegression:")
print("X shape:", X_regression.shape)
print("y shape:", y_regression.shape)

# Verify that neither target contains missing values.
print("\nMissing classification targets:", y_classification.isna().sum())
print("Missing regression targets:", y_regression.isna().sum())


# STEP 25: Validate model inputs and check for leakage

# Columns that must not be used to predict priority.
classification_forbidden = {
    "priority",
    "impact",
    "urgency",
    "resolution_hours",
    "remaining_resolution_hours",
    "history_resolved_at",
    "resolved_at",
    "closed_at",
    "resolved_by",
    "closed_code"
}

# Columns that must not be used to predict resolution time.
regression_forbidden = {
    "remaining_resolution_hours",
    "resolution_hours",
    "history_resolved_at",
    "resolved_at",
    "closed_at",
    "resolved_by",
    "closed_code"
}

# Verify that X contains exactly our selected features.
print( "\nClassification features match:", list(X_classification.columns) == classification_features )
print( "Regression features match:", list(X_regression.columns) == regression_features )

# Identify prohibited columns accidentally included in X.
print( "\nClassification prohibited columns:", sorted(set(X_classification.columns) & classification_forbidden) )
print( "Regression prohibited columns:", sorted(set(X_regression.columns) & regression_forbidden) )

# Verify that each input row matches its target row.
print( "\nClassification X/y indexes aligned:", X_classification.index.equals(y_classification.index) )
print( "Regression X/y indexes aligned:", X_regression.index.equals(y_regression.index) )

# STEP 26: Split classification and regression datasets
from sklearn.model_selection import train_test_split

X_class_train, X_class_test, y_class_train, y_class_test  = (
    train_test_split(
        X_classification,
        y_classification,
        test_size=0.20,
        random_state=42,
        stratify=y_classification
    )
)

# Regression: standard random split.
X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_regression,
        y_regression,
        test_size=0.20,
        random_state=42
    )
)

# Verify the resulting dimensions.
print("\nClassification split:")
print("X train:", X_class_train.shape)
print("X test:", X_class_test.shape)
print("y train:", y_class_train.shape)
print("y test:", y_class_test.shape)

print("\nRegression split:")
print("X train:", X_reg_train.shape)
print("X test:", X_reg_test.shape)
print("y train:", y_reg_train.shape)
print("y test:", y_reg_test.shape)

# Verify priority distribution after stratification.
print("\nClassification training priority distribution:")
print(y_class_train.value_counts())

print("\nClassification testing priority distribution:")
print(y_class_test.value_counts())

# STEP 27: Identify categorical and numerical features

# Classification
classification_categorical = [
    "contact_type",
    "location",
    "category",
    "subcategory",
    "u_symptom"
]
classification_numerical = [ "opened_hour", "opened_day_of_week" ]

# Regression
regression_categorical = classification_categorical + [ "impact", "urgency" ]
regression_numerical = classification_numerical.copy()

# Print feature groups.
print("\nClassification categorical: ", classification_categorical)
print("Classification numerical: ", classification_numerical)
print("Regression categorical: ", regression_categorical)
print("Regression numerical: ", regression_numerical)

# Inspect actual training data types.
print("\nClassification training data types: \n", X_class_train.dtypes)
print("\nRegression training data types: \n", X_reg_train.dtypes)

# Inspect impact and urgency values.
print("\nImpact values: \n", X_reg_train["impact"].unique())
print("\nUrgency values: \n", X_reg_train["urgency"].unique())

# STEP 28: Define categorical preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer #Replaces missing categorical values with "Unknown".
from sklearn.preprocessing import OneHotEncoder #Converts categorical values into numerical columns

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                missing_values=pd.NA,
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

print("\nCategorical preprocessing pipeline:")
print(categorical_pipeline)


# STEP 29: Define numerical preprocessing

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

print("\nNumerical preprocessing pipeline:")
print(numerical_pipeline)
    

# STEP 30: Combine categorical and numerical preprocessing
classification_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            classification_categorical
        ),
        (
            "numerical",
            numerical_pipeline,
            classification_numerical
        )
    ]
)

regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            regression_categorical
        ),
        (
            "numerical",
            numerical_pipeline,
            regression_numerical
        )
    ]
)

print("\nClassification preprocessor:")
print(classification_preprocessor)

print("\nRegression preprocessor:")
print(regression_preprocessor)

# STEP 31: Fit preprocessing on training data only
X_class_train_processed = classification_preprocessor.fit_transform(X_class_train)
X_class_test_processed = classification_preprocessor.transform(X_class_test)

# Regression preprocessing
X_reg_train_processed = ( regression_preprocessor.fit_transform(X_reg_train) )
X_reg_test_processed = ( regression_preprocessor.transform(X_reg_test) )
    
# Verify processed dataset dimensions.
print("\nProcessed classification datasets:")
print("Training:", X_class_train_processed.shape)
print("Testing:", X_class_test_processed.shape)

print("\nProcessed regression datasets:")
print("Training:", X_reg_train_processed.shape)
print("Testing:", X_reg_test_processed.shape)

# STEP 32: Validate processed datasets

processed_datasets = {
    "Classification train": X_class_train_processed,
    "Classification test": X_class_test_processed,
    "Regression train": X_reg_train_processed,
    "Regression test": X_reg_test_processed
}

for name, dataset in processed_datasets.items():

    # Sparse matrices store their nonzero entries in .data.
    # Dense matrices store all values directly.
    values = dataset.data if issparse(dataset) else dataset

    print(
        name,
        "| All values finite:", np.isfinite(values).all(),
        "| Sparse:", issparse(dataset)
    )


# Verify that feature names match the processed column counts.
classification_feature_names = ( classification_preprocessor.get_feature_names_out() )
regression_feature_names = ( regression_preprocessor.get_feature_names_out() )

print( "\nClassification feature count matches: ", 
      len(classification_feature_names) == X_class_train_processed.shape[1] )
print( "Regression feature count matches:", len(regression_feature_names) == X_reg_train_processed.shape[1] )

# STEP 33: Train the Logistic Regression classifier

# Create the model.
logistic_model = LogisticRegression(max_iter=1000)

# Train the model using classification training data only.
logistic_model = logistic_model.fit(
    X_class_train_processed,
    y_class_train
)

# Inspect the trained model.
print("\nLogistic Regression training completed.")

print("\nLearned priority classes:")
print(logistic_model.classes_)

print("\nTraining iterations:")
print(logistic_model.n_iter_)

# STEP 34: Generate Logistic Regression predictions

# Predict priorities for the held-out test incidents.
y_class_pred = logistic_model.predict(X_class_test_processed)

# Verify the prediction count.
print("\nNumber of predictions:", len(y_class_pred))

# Inspect the first 10 predicted priorities.
print("\nFirst 10 predicted priorities:")
print(y_class_pred[:10])

# Inspect the distribution of predicted priority classes.
print("\nPredicted priority distribution:")
print( pd.Series(y_class_pred).value_counts() )

# STEP 35: Evaluate Logistic Regression

# Define the priority class order.
priority_labels = logistic_model.classes_

# 1. Overall accuracy
accuracy = accuracy_score(y_class_test, y_class_pred)
print("\nLogistic Regression accuracy:", accuracy)

# 2. Confusion matrix
cm = confusion_matrix(y_class_test, y_class_pred, labels=priority_labels)

# Convert the matrix into a labeled DataFrame.
cm_df = pd.DataFrame(
    cm,
    index=priority_labels,
    columns=priority_labels
)

print("\nConfusion matrix:")
print(cm_df.to_string())

# 3. Classification report
print("\nClassification report:")
print(
    classification_report(
        y_class_test,
        y_class_pred,
        labels=priority_labels,
        zero_division=0
    )
)

# 4. Majority-class baseline accuracy
baseline_accuracy = (
    y_class_test.value_counts().max()
    / len(y_class_test)
)

print("\nMajority-class baseline accuracy:", baseline_accuracy)