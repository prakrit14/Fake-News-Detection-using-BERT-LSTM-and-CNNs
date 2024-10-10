import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the dataset
data = pd.read_csv('AML_Pro_data_train.csv')

# # Display the first few rows of the dataframe and basic info about the dataset
# print(data.head())
# print(data.info())
# print(data.describe(include='all'))

# Define categorical and numeric columns
categorical_cols = ['subject', 'speaker', 'position', 'state', 'Party']
numeric_cols = ['info_1', 'info_2', 'info_3', 'info_4', 'info_5']

# Create a column transformer to apply the necessary transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Split the data into features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numeric columns
categorical_cols = ['subject', 'speaker', 'position', 'state', 'Party']
numeric_cols = ['info_1', 'info_2', 'info_3', 'info_4', 'info_5']

# Preprocessors
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("Begin training model")
# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
print("training finished")

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
