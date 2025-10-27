
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

# -----------------------
# Helper Functions
# -----------------------

def perform_one_way_anova(data, cat_col, num_col):
    groups = [group[num_col].dropna().values for name, group in data.groupby(cat_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    return f_stat, p_val

def perform_two_way_anova(data, cat1, cat2, num_col):
    formula = f'{num_col} ~ C({cat1}) + C({cat2}) + C({cat1}):C({cat2})'
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def save_output(df, filename="anova_output.csv", folder="ANOVA_Results"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    st.success(f"✅ Output saved as {filepath}")

def save_figure(fig, filename="anova_plot.png", folder="ANOVA_Results"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, bbox_inches='tight')
    st.success(f"✅ Figure saved as {filepath}")

def decision_explanation(p_val):
    """Return final decision text based on p-value"""
    if p_val < 0.05:
        return "Reject Null Hypothesis (H0) → Significant differences exist"
    else:
        return "Fail to reject Null Hypothesis (H0) → No significant differences"

# -----------------------
# Streamlit Layout
# -----------------------
st.set_page_config(page_title="ANOVA Statistical App", layout="wide")
st.title("📊 One-Way & Two-Way ANOVA ")

# -----------------------
# 1️⃣ Data Input
# -----------------------
st.header("1️⃣ Data Input")
input_method = st.radio("Choose Input Method", ["Upload CSV", "Manual Input"])

data = None

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(data.head())

elif input_method == "Manual Input":
    st.info("Manual input: Type your data below")
    num_rows = st.number_input("Number of rows (max 50)", min_value=2, max_value=50, value=7)
    num_cols = st.number_input("Number of columns (max 10)", min_value=2, max_value=10, value=3)
    
    if st.button("Create Manual Input Table"):
        data = pd.DataFrame(columns=[f"Col{i}" for i in range(1, num_cols+1)])
        st.session_state.manual_data = data.copy()
    
    if 'manual_data' in st.session_state:
        manual_data = st.session_state.manual_data
        for i in range(num_rows):
            row_inputs = []
            for j in range(num_cols):
                val = st.text_input(f"Row {i+1}, Col {j+1}", key=f"cell_{i}_{j}")
                row_inputs.append(val)
            if all(row_inputs):
                manual_data.loc[i] = row_inputs
        data = manual_data
        st.subheader("Manual Input Data Preview")
        st.dataframe(data)

# -----------------------
# 2️⃣ ANOVA Selection & Column Selection
# -----------------------
if data is not None:
    st.header("2️⃣ Select Columns for ANOVA")
    
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except:
            pass
    
    cat_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    st.write(f"Categorical columns: {cat_columns}")
    st.write(f"Numerical columns: {num_columns}")
    
    selected_anova = st.selectbox("Choose ANOVA Type", ["One-Way ANOVA", "Two-Way ANOVA"])
    
    # -----------------------
    # One-Way ANOVA
    # -----------------------
    if selected_anova == "One-Way ANOVA":
        cat_col = st.selectbox("Select Categorical Column", cat_columns)
        num_col = st.selectbox("Select Numerical Column", num_columns)
        
        if st.button("Run One-Way ANOVA"):
            f_stat, p_val = perform_one_way_anova(data, cat_col, num_col)
            
            # -----------------------
            # Hypothesis Detailed Explanation
            # -----------------------
            st.subheader("📌 Hypothesis Explanation (Detailed)")
            st.markdown(f"""
**1️⃣ Null Hypothesis (H0):**  
The mean of **{num_col}** is the same across all groups of **{cat_col}**.  

**2️⃣ Alternative Hypothesis (H1):**  
At least one group mean of **{num_col}** is different.  

**3️⃣ Reasoning:**  
ANOVA compares **variance between groups** vs **variance within groups**.  
- F-value = Variance between groups / Variance within groups  
- Larger F → more likely to reject H0  

**4️⃣ Interpretation:**  
Based on the calculated p-value, we make a final decision.
""")
            
            # -----------------------
            # Results
            # -----------------------
            st.subheader("📈 ANOVA Results")
            st.write(f"**F-Value:** {f_stat:.4f}")
            st.write(f"**P-Value:** {p_val:.4f}")
            
            # Final decision explanation
            decision = decision_explanation(p_val)
            st.markdown(f"**Final Decision:** {decision}")
            if "Reject" in decision:
                st.success("✅ Null Hypothesis Rejected")
            else:
                st.info("ℹ️ Null Hypothesis Accepted")
            
            # -----------------------
            # Visualizations
            # -----------------------
            st.subheader("📊 Visualizations")
            fig, ax = plt.subplots(1, 2, figsize=(12,5))
            sns.boxplot(x=cat_col, y=num_col, data=data, ax=ax[0])
            ax[0].set_title("Boxplot")
            sns.violinplot(x=cat_col, y=num_col, data=data, ax=ax[1])
            ax[1].set_title("Violin Plot")
            st.pyplot(fig)
            
            # Save Output & Figure
            output_df = pd.DataFrame({
                "Categorical Column": [cat_col],
                "Numerical Column": [num_col],
                "F-Value": [f_stat],
                "P-Value": [p_val],
                "Decision": [decision]
            })
            save_output(output_df, filename="one_way_anova_output.csv")
            save_figure(fig, filename="one_way_anova_plot.png")
    
    # -----------------------
    # Two-Way ANOVA
    # -----------------------
    elif selected_anova == "Two-Way ANOVA":
        cat_col1 = st.selectbox("Select First Categorical Column", cat_columns, key="cat1")
        cat_col2 = st.selectbox("Select Second Categorical Column", cat_columns, key="cat2")
        num_col2 = st.selectbox("Select Numerical Column", num_columns, key="num2")
        
        if st.button("Run Two-Way ANOVA"):
            anova_table = perform_two_way_anova(data, cat_col1, cat_col2, num_col2)
            
            # -----------------------
            # Hypothesis Detailed Explanation
            # -----------------------
            st.subheader("📌 Hypothesis Explanation (Detailed)")
            st.markdown(f"""
**1️⃣ Null Hypotheses (H0):**  
- No main effect of **{cat_col1}** on **{num_col2}**  
- No main effect of **{cat_col2}** on **{num_col2}**  
- No interaction effect between **{cat_col1}** and **{cat_col2}** on **{num_col2}**

**2️⃣ Alternative Hypotheses (H1):**  
- At least one main effect or interaction is significant  

**3️⃣ Reasoning:**  
Two-Way ANOVA examines two factors simultaneously.  
- F-value = variance between groups / variance within groups  
- P-value tells if effect is significant  

**4️⃣ Interpretation:**  
Check p-values for each term to decide whether to reject or accept H0.
""")
            
            # -----------------------
            # Results with Decision
            # -----------------------
            st.subheader("📈 ANOVA Table")
            anova_table_sorted = anova_table.sort_values("F", ascending=False)
            st.dataframe(anova_table_sorted)
            
            # Add decision column
            decision_list = []
            for pv in anova_table_sorted['PR(>F)']:
                decision_list.append(decision_explanation(pv))
            anova_table_sorted['Decision'] = decision_list
            
            st.subheader("📌 Final Decision for Each Term")
            st.dataframe(anova_table_sorted)
            
            # -----------------------
            # Visualization
            # -----------------------
            st.subheader("📊 Visualization")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.boxplot(x=cat_col1, y=num_col2, hue=cat_col2, data=data, ax=ax)
            ax.set_title("Two-Way ANOVA Boxplot")
            st.pyplot(fig)
            
            # Save Output & Figure
            save_output(anova_table_sorted, filename="two_way_anova_output.csv")
            save_figure(fig, filename="two_way_anova_plot.png")   