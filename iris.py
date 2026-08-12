import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Classification App",
    page_icon="🌸",
    layout="wide"
)

st.title("Iris Species Classification Dashboard")
st.markdown("Explore EDA, train & compare ML algorithms, and make predictions on custom flower measurements.")

# --- Load & Cache Data ---
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    # Rename columns for cleaner UI presentation
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    target_names = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    df['species'] = df['species'].map(target_names)
    return df

df = load_data()

# --- Sidebar: User Controls & Inputs ---
st.sidebar.header("⚙️ Configuration & Inference")

# Interactive Model Hyperparameters
st.sidebar.subheader("Model Parameters")
k_val = st.sidebar.slider("k-NN: Number of Neighbors (k)", min_value=1, max_value=15, value=5)
max_iter = st.sidebar.slider("Logistic Regression: Max Iterations", min_value=50, max_value=500, value=200)
tree_depth = st.sidebar.slider("Decision Tree: Max Depth", min_value=1, max_value=10, value=4)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Input Custom Sample for Prediction")
input_sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
input_sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5, step=0.1)
input_petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
input_petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

# --- Navigation Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Data & EDA", "⚡ Model Comparison", "🎯 Live Inference"])

# ==========================================
# TAB 1: EDA & VISUALIZATIONS
# ==========================================
with tab1:
    st.header("Dataset Overview")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Total Records:** {df.shape[0]} rows | **Features:** {df.shape[1] - 1}")
    
    with col2:
        st.subheader("Class Distribution")
        fig_count, ax_count = plt.subplots(figsize=(6, 3))
        sns.countplot(data=df, x='species', palette="viridis", ax=ax_count)
        plt.title("Species Count Distribution")
        st.pyplot(fig_count)

    st.markdown("---")
    st.header("Feature Analysis")
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.subheader("Feature Correlations")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

    with col_eda2:
        st.subheader("Class Separability (Pairplot)")
        pairplot_fig = sns.pairplot(df, hue='species', corner=True, palette="viridis")
        st.pyplot(pairplot_fig.fig)

# ==========================================
# TAB 2: MODEL TRAINING & EVALUATION
# ==========================================
# Preprocess Data
X = df.drop(columns=['species'])
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
models = {
    "k-NN": KNeighborsClassifier(n_neighbors=k_val),
    "Logistic Regression": LogisticRegression(max_iter=max_iter, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
}

results = {}
for name, model in models.items():
    X_tr = X_train if name == "Decision Tree" else X_train_scaled
    X_te = X_test if name == "Decision Tree" else X_test_scaled
    
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "model": model,
        "accuracy": acc,
        "predictions": y_pred,
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

# Identify Best Model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']

# Save best model and scaler to disk
joblib.dump(best_model, 'iris_best_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')

with tab2:
    st.header("Algorithm Performance Comparison")
    
    # Model Accuracy Cards
    acc_cols = st.columns(3)
    for idx, (name, res) in enumerate(results.items()):
        with acc_cols[idx]:
            st.metric(label=f"🤖 {name}", value=f"{res['accuracy'] * 100:.1f}%")

    st.success(f"🏆 **Best Performing Model:** {best_model_name} (Saved to `iris_best_model.pkl`)")
    st.markdown("---")
    
    # Confusion Matrices and Classification Reports
    st.subheader("Detailed Evaluation Metrics")
    m_tabs = st.tabs(list(models.keys()))
    
    for idx, (name, res) in enumerate(results.items()):
        with m_tabs[idx]:
            col_cm, col_rep = st.columns(2)
            
            with col_cm:
                st.markdown(f"#### Confusion Matrix - {name}")
                fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
                sns.heatmap(
                    res['cm'], 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=np.unique(y), 
                    yticklabels=np.unique(y),
                    ax=ax_cm
                )
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                st.pyplot(fig_cm)
            
            with col_rep:
                st.markdown(f"#### Classification Report - {name}")
                report_df = pd.DataFrame(res['report']).transpose()
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

# ==========================================
# TAB 3: LIVE INFERENCE
# ==========================================
with tab3:
    st.header("Single Sample Prediction")
    st.write("Current sample measurement configured via sidebar:")
    
    sample_df = pd.DataFrame([[
        input_sepal_length, input_sepal_width, input_petal_length, input_petal_width
    ]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    st.dataframe(sample_df, use_container_width=True)
    
    if st.button("🚀 Predict Species", type="primary"):
        # Load saved artifacts to demonstrate persistence flow
        loaded_model = joblib.load('iris_best_model.pkl')
        loaded_scaler = joblib.load('iris_scaler.pkl')
        
        # Scale if model is scaled-dependent
        if not isinstance(loaded_model, DecisionTreeClassifier):
            prepared_sample = loaded_scaler.transform(sample_df)
        else:
            prepared_sample = sample_df
            
        pred_class = loaded_model.predict(prepared_sample)[0]
        
        # Display Prediction Result
        st.balloons()
        st.markdown(f"### Predicted Species: **:green[{pred_class}]**")
        
        if hasattr(loaded_model, "predict_proba"):
            probs = loaded_model.predict_proba(prepared_sample)[0]
            prob_df = pd.DataFrame([probs], columns=loaded_model.classes_)
            st.write("Prediction Probabilities:")
            st.bar_chart(prob_df.T)