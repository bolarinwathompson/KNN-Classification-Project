# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 14:06:34 2025

@author: SEGUN BOLARINWA
"""

#################################
# KNN for Advanced Classification
#################################

# Import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Import data for the model 
data_for_model = pd.read_pickle("abc_classification_modelling.p")

# Drop unnecessary columns
data_for_model.drop("customer_id", axis=1, inplace=True)

# Shuffle data
data_for_model = shuffle(data_for_model, random_state=42)

# Class balance
print("Class Balance (normalized):")
print(data_for_model["signup_flag"].value_counts(normalize=True))

# Deal with missing data
print("\nMissing data per column:")
print(data_for_model.isna().sum())
data_for_model.dropna(how="any", inplace=True)

# Deal with outliers using boxplot approach
outlier_columns = ["distance_from_store", "total_sales", "total_items"]
for column in outlier_columns:
    lower_quantile = data_for_model[column].quantile(0.25)
    upper_quantile = data_for_model[column].quantile(0.75)
    iqr = upper_quantile - lower_quantile
    iqr_extended = iqr * 2
    min_border = lower_quantile - iqr_extended 
    max_border = upper_quantile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | 
                              (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")

# Split input and output variables
X = data_for_model.drop(["signup_flag"], axis=1)
y = data_for_model["signup_flag"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Deal with categorical variables
categorical_vars = ["gender"]
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

# Generate one-hot encoded data for the categorical features
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded  = one_hot_encoder.transform(X_test[categorical_vars])

# Retrieve feature names for the encoded variables
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert the encoded arrays into DataFrames with proper column names
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_feature_names)
X_test_encoded  = pd.DataFrame(X_test_encoded, columns=encoder_feature_names)

# Concatenate the original data (with reset indices) with the encoded DataFrames
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_test  = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)

# Drop the original categorical column from both training and testing sets
X_train.drop(columns=categorical_vars, inplace=True)
X_test.drop(columns=categorical_vars, inplace=True)

# Feature Scaling using MinMaxScaler
scale_norm = MinMaxScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns=X_test.columns)

# Feature Selection using RFECV with a RandomForestClassifier as the estimator
clf_rf = RandomForestClassifier(random_state=42)
feature_selector = RFECV(estimator=clf_rf, cv=5)
feature_selector.fit(X_train, y_train)

optimal_feature_count = feature_selector.n_features_
print(f"\nOptimal number of features: {optimal_feature_count}")

# Reduce the training and testing sets to the selected features
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

# Plot the cross-validation scores for each number of features selected
scores = feature_selector.cv_results_["mean_test_score"]
plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFECV\nOptimal number of features is {optimal_feature_count}")
plt.tight_layout()
plt.show()

# -------------------------
# KNN Model Training
# -------------------------

# Initialize and train the KNN classifier using the training set
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

# Model Assessment on the test set
y_pred_class = knn_clf.predict(X_test)
y_pred_prob = knn_clf.predict_proba(X_test)[:, 1]

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
# Use a known valid style; here we use "classic"
plt.style.use("classic")
plt.matshow(conf_matrix, cmap="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha="center", va="center", fontsize=20)
plt.show()

# Performance Metrics
acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class)
rec = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# -------------------------
# Finding the Optimal k value for KNN using F1 score
# -------------------------
k_list = list(range(2, 25))
f1_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = f1_score(y_test, y_pred)
    f1_scores.append(score)
    
max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)
optimal_k_value = k_list[max_f1_idx]

print(f"\nOptimal k: {optimal_k_value} with F1 Score: {max_f1:.2f}")

# Plot F1 score vs. k
plt.plot(k_list, f1_scores, marker="o")  # Plot F1 scores vs. k
plt.scatter(optimal_k_value, max_f1, marker="x", color="red")  # Mark the optimal k value
plt.title(f"F1 Score by k\nOptimal k = {optimal_k_value} (F1 Score = {round(max_f1, 2)})")
plt.xlabel("k")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.show()
