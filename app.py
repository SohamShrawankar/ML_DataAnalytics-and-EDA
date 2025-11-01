import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import warnings

warnings.filterwarnings('ignore')


# Load the dataset
df = pd.read_csv('/content/cancer-risk-factors.csv')

# Streamlit app title and description
st.title('Cancer Risk Factors Analysis Dashboard')
st.write('This dashboard visualizes the distribution of risk factors, their relationships, and compares the performance of different classification models.')

# Display the raw data
st.header('Raw Data')
st.dataframe(df.head())

# Data Distribution
st.header('Data Distribution')

numerical_features = ['Age', 'BMI', 'Overall_Risk_Score']
categorical_features = ['Cancer_Type', 'Risk_Level']

st.subheader('Distribution of Numerical Features')
fig1, axes1 = plt.subplots(1, len(numerical_features), figsize=(15, 5))
for i, feature in enumerate(numerical_features):
    sns.histplot(df[feature], kde=True, ax=axes1[i])
    axes1[i].set_title(f'Distribution of {feature}')
    axes1[i].set_xlabel(feature)
    axes1[i].set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig1)

st.subheader('Distribution of Categorical Features')
fig2, axes2 = plt.subplots(1, len(categorical_features), figsize=(15, 5))
for i, feature in enumerate(categorical_features):
    sns.countplot(data=df, y=feature, order=df[feature].value_counts().index, ax=axes2[i])
    axes2[i].set_title(f'Distribution of {feature}')
    axes2[i].set_xlabel('Count')
    axes2[i].set_ylabel(feature)
plt.tight_layout()
st.pyplot(fig2)


# Correlation Analysis
st.header('Correlation Analysis')
st.subheader('Correlation Matrix of Numerical Features')

numerical_df = df.select_dtypes(include=['number'])
correlation_matrix = numerical_df.corr()

fig3, ax3 = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
ax3.set_title('Correlation Matrix of Numerical Features')
st.pyplot(fig3)

# Relationship Analysis
st.header('Relationship Analysis')

st.subheader('Numerical Features vs. Risk Level')
numerical_features_rel = ['Age', 'BMI', 'Overall_Risk_Score']
fig4, axes4 = plt.subplots(1, len(numerical_features_rel), figsize=(15, 5))
for i, feature in enumerate(numerical_features_rel):
    sns.boxplot(x='Risk_Level', y=feature, data=df, order=['Low', 'Medium', 'High'], ax=axes4[i])
    axes4[i].set_title(f'{feature} vs. Risk Level')
    axes4[i].set_xlabel('Risk Level')
    axes4[i].set_ylabel(feature)
plt.tight_layout()
st.pyplot(fig4)

st.subheader('Distribution of Categorical Features by Risk Level')
categorical_features_rel = ['Cancer_Type', 'Gender', 'Smoking', 'Alcohol_Use', 'Obesity',
                        'Family_History', 'Diet_Red_Meat', 'Diet_Salted_Processed',
                        'Fruit_Veg_Intake', 'Physical_Activity', 'Air_Pollution',
                        'Occupational_Hazards', 'BRCA_Mutation', 'H_Pylori_Infection',
                        'Calcium_Intake', 'Physical_Activity_Level']

fig5, axes5 = plt.subplots(4, 4, figsize=(18, 15))
axes5 = axes5.flatten() # Flatten the 2D array of axes for easy iteration
for i, feature in enumerate(categorical_features_rel):
    sns.countplot(data=df, x='Risk_Level', hue=feature, order=['Low', 'Medium', 'High'], palette='viridis', ax=axes5[i])
    axes5[i].set_title(f'Distribution of {feature} by Risk Level')
    axes5[i].set_xlabel('Risk Level')
    axes5[i].set_ylabel('Count')
    axes5[i].tick_params(axis='x', rotation=45)
    axes5[i].legend(title=feature, loc='upper right', bbox_to_anchor=(1.3, 1))

# Hide any unused subplots
for j in range(i + 1, len(axes5)):
    fig5.delaxes(axes5[j])

plt.tight_layout()
st.pyplot(fig5)

# Data Preprocessing (needed for model training within the app)
X = df.drop(['Risk_Level', 'Overall_Risk_Score', 'Patient_ID'], axis=1)
y = df['Risk_Level']

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

onehot_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_features)
all_feature_names = list(onehot_feature_names) + numerical_features

X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
X_processed_df = X_processed_df.apply(pd.to_numeric, errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df, y, test_size=0.2, random_state=42, stratify=y
)

# Model Training and Evaluation (re-run within the app for display)
st.header('Model Performance Comparison')

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
precision_knn = precision_score(y_test, y_pred_knn, average='weighted', zero_division=1)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
precision_dt = precision_score(y_test, y_pred_dt, average='weighted', zero_division=1)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=1)

# SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
precision_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=1)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
precision_nb = precision_score(y_test, y_pred_nb, average='weighted', zero_division=1)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
precision_lr = precision_score(y_test, y_pred_lr, average='weighted', zero_division=1)


metrics_data = {
    'Model': ['KNN', 'Decision Tree', 'Random Forest', 'SVM', 'Naive Bayes', 'Logistic Regression'],
    'Accuracy': [accuracy_knn, accuracy_dt, accuracy_rf, accuracy_svm, accuracy_nb, accuracy_lr],
    'F1 Score': [f1_knn, f1_dt, f1_rf, f1_svm, f1_nb, f1_lr],
    'Recall': [recall_knn, recall_dt, recall_rf, recall_svm, recall_nb, recall_lr],
    'Precision': [precision_knn, precision_dt, precision_rf, precision_svm, precision_nb, precision_lr]
}
metrics_df = pd.DataFrame(metrics_data)

st.subheader('Model Evaluation Metrics')
st.dataframe(metrics_df)

st.subheader('Comparison of Model Performance Metrics')
metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

fig6, ax6 = plt.subplots(figsize=(15, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted, palette='tab10', ax=ax6)
ax6.set_title('Comparison of Model Performance Metrics')
ax6.set_xlabel('Model')
ax6.set_ylabel('Score')
ax6.tick_params(axis='x', rotation=45)
ax6.legend(title='Metric')

for container in ax6.containers:
    ax6.bar_label(container, fmt='%.2f', label_type='edge')

plt.tight_layout()
st.pyplot(fig6)

# Function for prediction
def predict_cancer_risk(data, model, preprocessor, feature_names):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Apply the same preprocessing as used for training data
    input_processed = preprocessor.transform(input_df)

    # Convert processed input to DataFrame with correct column names
    input_processed_df = pd.DataFrame(input_processed, columns=feature_names)

    # Ensure all columns are numeric
    input_processed_df = input_processed_df.apply(pd.to_numeric, errors='coerce')

    # Make prediction
    prediction = model.predict(input_processed_df)
    return prediction[0]

# Prediction section
st.header('Predict Cancer Risk for a New Patient')

st.write("Enter the risk factors for a new patient to predict their cancer risk level.")

# Input fields for each risk factor
with st.form("prediction_form"):
    cancer_type = st.selectbox("Cancer Type", df['Cancer_Type'].unique())
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    smoking = st.slider("Smoking Score (0-10)", 0, 10, 5)
    alcohol_use = st.slider("Alcohol Use Score (0-10)", 0, 10, 5)
    obesity = st.slider("Obesity Score (0-10)", 0, 10, 5)
    family_history = st.selectbox("Family History", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    diet_red_meat = st.slider("Diet Red Meat Score (0-10)", 0, 10, 5)
    diet_salted_processed = st.slider("Diet Salted Processed Score (0-10)", 0, 10, 5)
    fruit_veg_intake = st.slider("Fruit and Vegetable Intake Score (0-10)", 0, 10, 5)
    physical_activity = st.slider("Physical Activity Score (0-10)", 0, 10, 5)
    air_pollution = st.slider("Air Pollution Score (0-10)", 0, 10, 5)
    occupational_hazards = st.slider("Occupational Hazards Score (0-10)", 0, 10, 5)
    brca_mutation = st.selectbox("BRCA Mutation", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    h_pylori_infection = st.selectbox("H. Pylori Infection", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    calcium_intake = st.slider("Calcium Intake Score (0-10)", 0, 10, 5)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    physical_activity_level = st.slider("Physical Activity Level Score (0-10)", 0, 10, 5)


    submitted = st.form_submit_button("Predict Risk Level")

    if submitted:
        new_patient_data = {
            'Cancer_Type': cancer_type,
            'Age': age,
            'Gender': gender,
            'Smoking': smoking,
            'Alcohol_Use': alcohol_use,
            'Obesity': obesity,
            'Family_History': family_history,
            'Diet_Red_Meat': diet_red_meat,
            'Diet_Salted_Processed': diet_salted_processed,
            'Fruit_Veg_Intake': fruit_veg_intake,
            'Physical_Activity': physical_activity,
            'Air_Pollution': air_pollution,
            'Occupational_Hazards': occupational_hazards,
            'BRCA_Mutation': brca_mutation,
            'H_Pylori_Infection': h_pylori_infection,
            'Calcium_Intake': calcium_intake,
            'BMI': bmi,
            'Physical_Activity_Level': physical_activity_level
        }

        # Make the prediction using the best-performing model (Logistic Regression)
        predicted_risk = predict_cancer_risk(
            new_patient_data,
            lr_model,  # Using the trained Logistic Regression model
            preprocessor,
            all_feature_names
        )

        st.subheader("Prediction Result")
        st.write(f"The predicted cancer risk level for the new patient is: **{predicted_risk}**")

