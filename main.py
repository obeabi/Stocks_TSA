# Install libraries
import pandas as pd
from ChurnClassifier import plot_class_distribution
from ChurnClassifier import MLClassifier
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
import warnings
warnings.filterwarnings("ignore")
random_state = 42

# Start code here
df = pd.read_csv("C:\\Users\\CA-DObembe\\McDonalds Corp\\Home - Digital\\Adhoc Requests\\2024-03-27 - Churn Monopoly and Points Expiry\\ML Model\\data.csv")
#df = df.sample(400000)
print("\nCount of guests in dataframe is :", df.user_id.nunique())
features = ['Rnew', 'tl_gcs', 'tl_sales', 'RFM_Factor']
target_column_name = 'churned'
# Plot the distribution of the target variable
plot_class_distribution(df, target_column_name)
classifier = MLClassifier(learning_rate=0.01, max_iter=800, n_estimators=100, random_state=random_state)
print("Classifier Model initialized successfully!")
optimal_clusters = classifier.find_optimal_clusters(df[features], max_k=5)
classifier.plot_elbow_curve(max_k=5)
print("\nThe count of optimal clusters is :", optimal_clusters)
# Apply K-means clustering to add a new feature
df['cluster'] = classifier.find_clusters(df[features], n_clusters=optimal_clusters)
# df = pd.get_dummies(df, columns=['cluster'], prefix='cluster')
encoder = OrdinalEncoder()
# encoder = LabelEncoder()
df['cluster_encoded'] = encoder.fit_transform(df[['cluster']])
# Prepare data for training
X = df.drop(columns=[target_column_name])
y = df[target_column_name]

# Split data into training and test sets and train the model
X_train, X_test, y_train, y_test = classifier.split_train_test(X, y, test_size=0.2)
# Train the classifier
classifier.train(X_train, y_train)
evaluation_results = classifier.evaluate(X_test, y_test)
print(f"Best Model: {evaluation_results['best_model']}")
print(f"Accuracy: {evaluation_results['accuracy'] * 100:.2f}%")
print("Confusion Matrix:")
print(evaluation_results["confusion_matrix"])
print(f"AUC: {evaluation_results['auc']:.2f}")
print("Classification Report:")
print(evaluation_results["Classification_report"])
# Plot confusion matrix
conf_matrix = evaluation_results["confusion_matrix"]
classifier.plot_confusion_matrix(conf_matrix)
# Plot ROC curve
fpr, tpr, thresholds = evaluation_results['fpr'], evaluation_results['tpr'], evaluation_results['thresholds']
classifier.plot_roc_curve(fpr, tpr)

# Predict the probability of churn of all guests
churn_probabilities = classifier.predict_churn_probability(X)

# Convert churn_probabilities to a DataFrame
churn_proba_df = pd.DataFrame(churn_probabilities, columns=['Churn_Probability'])
# Concatenate X with churn_proba_df along the columns axis
df_with_churn_proba = pd.concat([df[['user_id', 'churned', 'RFM_Factor']], churn_proba_df], axis=1)

# Define custom bin ranges
custom_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Create bins from Churn_Probability column using custom bin ranges
df_with_churn_proba['Churn_Probability_Bin'] = pd.cut(df_with_churn_proba['Churn_Probability'], bins=custom_bins, right=False)
# Group by the Churn_Probability_Bin and calculate the count of user_id and sum of churned per bin
bin_counts = df_with_churn_proba.groupby('Churn_Probability_Bin').agg(
                        user_count=('user_id', 'count'),
                        churned_sum=('churned', 'sum')
                    ).reset_index()

bin_counts['perc_churned'] = bin_counts['churned_sum'] / bin_counts['user_count']
# Print or display the resulting DataFrame
print(bin_counts)

# Save dataframe
bin_counts.to_csv('ChurnModelSummary.csv', index=False)







