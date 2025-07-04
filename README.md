# K-Nearest Neighbors (KNN) Classification for ABC Grocery

## Project Overview:
The **ABC Grocery KNN Classification** project uses the **K-Nearest Neighbors (KNN)** algorithm to predict whether a customer will sign up for the store's promotional offers based on historical transaction data. The model helps ABC Grocery optimize their marketing campaigns by predicting the likelihood of customer sign-ups based on attributes like **distance from the store**, **total sales**, **total items purchased**, and customer **demographics**.

## Objective:
The primary goal of this project is to build a **KNN classification model** that predicts the likelihood of a customer signing up for a promotional offer. This project uses customer transaction data to train the model, allowing ABC Grocery to better understand which customers are more likely to engage with marketing campaigns.

## Key Features:
- **Data Preprocessing**: The raw transaction data is cleaned, missing values are handled, and outliers are removed to ensure accurate model training.
- **Feature Selection**: **Recursive Feature Elimination with Cross-Validation (RFECV)** is used to select the most important features that influence customer sign-up likelihood.
- **Model Training**: The **KNN algorithm** is applied, with **cross-validation** used to ensure the model’s robustness.
- **Model Evaluation**: The model is evaluated based on metrics like **accuracy**, **precision**, **recall**, **F1 score**, and **confusion matrix**.

## Methods & Techniques:

### **1. Data Preprocessing**:
The dataset is loaded and cleaned:
- **Missing Data**: Any missing values in the dataset are handled.
- **Outlier Detection**: Outliers in features like **total sales** and **distance from store** are detected and removed using the **Interquartile Range (IQR)** method.
- **Feature Scaling**: **MinMaxScaler** is applied to normalize the features, ensuring that no single feature dominates the KNN algorithm due to its distance-based nature.

### **2. Feature Selection with RFECV**:
To improve model performance and interpretability, **Recursive Feature Elimination with Cross-Validation (RFECV)** is used to select the optimal set of features for classification. This method recursively removes less important features, ensuring the model is trained only on the most relevant attributes.

### **3. K-Nearest Neighbors (KNN) Classification**:
The **KNN classifier** is used to predict whether a customer will sign up for the promotional offer:
- The model uses **Euclidean distance** to find the closest neighbors and classify the customer.
- **K-Fold Cross Validation** is used to train and validate the model across different subsets of the data, ensuring it generalizes well.

### **4. Model Evaluation**:
Once trained, the model is evaluated using several metrics:
- **Accuracy**: Measures the overall performance of the model.
- **Precision**: The percentage of correctly predicted positive instances among all positive predictions.
- **Recall**: The percentage of correctly predicted positive instances among all actual positives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between them.
- **Confusion Matrix**: A matrix is generated to visualize the model’s predictions compared to the actual labels.

### **5. Hyperparameter Tuning**:
The **optimal number of neighbors (k)** is determined using **F1 score**:
- The **F1 score** is calculated for different values of **k** to identify the value that maximizes the performance.

### **6. Visualization**:
The model’s performance is visualized through a **confusion matrix** and **F1 score vs. k plot**, helping stakeholders understand the classification performance and select the best value for **k**.

## Technologies Used:
- **Python**: Programming language for data manipulation and modeling.
- **scikit-learn**: For implementing **KNN classification**, **cross-validation**, **feature selection**, and **evaluation metrics**.
- **pandas**: For handling and preprocessing the transaction data.
- **matplotlib**: For visualizing the **confusion matrix** and **F1 score vs. k plot**.
- **seaborn**: For enhancing the visual representation of the results.

## Key Results & Outcomes:
- The **KNN classification model** accurately predicts customer sign-up likelihood, helping ABC Grocery target customers more effectively.
- **Feature selection** using **RFECV** improved the model's accuracy by focusing on the most relevant features.
- The **optimal value of k** was determined using **F1 score**, ensuring that the model provides balanced performance in both precision and recall.

## Lessons Learned:
- **Feature Scaling** is crucial for distance-based algorithms like KNN, ensuring that all features contribute equally to the prediction.
- **Cross-validation** and **hyperparameter tuning** are essential to find the optimal settings and improve the generalization of the model.
- **Feature selection** helps simplify the model, improving interpretability and efficiency without sacrificing accuracy.

## Future Enhancements:
- **Model Optimization**: Further fine-tuning of hyperparameters (e.g., distance metrics, weights) could improve model performance.
- **Real-time Predictions**: Deploying the model for real-time customer sign-up predictions would enable ABC Grocery to dynamically adjust marketing strategies.
- **Advanced Algorithms**: Exploring more advanced models like **Gradient Boosting** or **XGBoost** could yield better results for customer prediction tasks.
