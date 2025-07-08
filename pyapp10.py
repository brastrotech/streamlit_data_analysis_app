import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import seaborn as sns
import io

st.set_page_config(page_title="Data Analysis App", layout="centered")
st.title("📊 Streamlit Data Analyzer")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")
    st.write("Preview of data:")
    st.dataframe(df.head())

    if st.checkbox("Change column data types manually"):
        for col in df.columns:
            option = st.selectbox(f"Select type for `{col}`", ["Auto", "string", "float", "int", "datetime"], key=col)
            if option == "float":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif option == "int":
                df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
            elif option == "datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif option == "string":
                df[col] = df[col].astype(str)

    st.markdown("---")

    # Analysis Option
    option = st.selectbox("Choose analysis type", ["Plotting", "Statistical Summary"])

    if option == "Plotting":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()

        x_col = st.selectbox("Select X-axis", options=all_cols)
        y_col = st.selectbox("Select Y-axis", options=numeric_cols)

        # Axis and title input
        x_label = st.text_input("X-axis label", value=x_col)
        y_label = st.text_input("Y-axis label", value=y_col)
        plot_title = st.text_input("Plot title", value=f"{y_label} vs {x_label}")

        # Averaging
        apply_avg = st.checkbox("Average large datasets (for fast plotting)", value=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        x = df[x_col]
        y = df[y_col]

        max_points = 1000
        if apply_avg and len(x) > max_points:
            avg_factor = len(x) // max_points
            x = x.groupby(np.arange(len(x)) // avg_factor).mean()
            y = y.groupby(np.arange(len(y)) // avg_factor).mean()

        ax.plot(x, y, 'o-', markersize=2, label='Data')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)

        # Curve fitting options
        fit_option = st.selectbox("Curve fitting (optional)", ["None", "Polynomial", "Exponential"])
        if fit_option != "None":
            def poly_func(x, a, b, c): return a * x ** 2 + b * x + c
            def exp_func(x, a, b): return a * np.exp(b * x)

            try:
                x_fit = x.astype(float)
                y_fit = y.astype(float)

                if fit_option == "Polynomial":
                    popt, _ = curve_fit(poly_func, x_fit, y_fit)
                    ax.plot(x_fit, poly_func(x_fit, *popt), 'r--', label='Poly Fit')

                elif fit_option == "Exponential":
                    popt, _ = curve_fit(exp_func, x_fit, y_fit, maxfev=5000)
                    ax.plot(x_fit, exp_func(x_fit, *popt), 'g--', label='Exp Fit')

                ax.legend()
            except Exception as e:
                st.warning(f"Curve fitting failed: {e}")

        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Plot as PNG",
            data=buf.getvalue(),
            file_name="plot.png",
            mime="image/png"
        )

    elif option == "Statistical Summary":
        st.write("📋 **Descriptive Statistics:**")
        st.dataframe(df.describe(include='all'))

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Correlation
        st.markdown("📈 **Correlation Matrix (Pearson):**")
        corr_matrix = df[numeric_cols].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

        # Regression
        st.markdown("📊 **Linear Regression Between Two Columns:**")
        col1 = st.selectbox("Select X for regression", options=numeric_cols, key="reg_x")
        col2 = st.selectbox("Select Y for regression", options=numeric_cols, key="reg_y")

        X = df[[col1]].dropna()
        Y = df[col2].loc[X.index]

        if not X.empty and not Y.empty:
            reg = LinearRegression()
            reg.fit(X, Y)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r2 = reg.score(X, Y)

            st.write(f"**Regression Equation:** {col2} = {slope:.4f} × {col1} + {intercept:.4f}")
            st.write(f"**R² Score:** {r2:.4f}")

            fig_reg, ax_reg = plt.subplots()
            ax_reg.scatter(X, Y, label="Data", s=10)
            ax_reg.plot(X, reg.predict(X), color='red', label="Regression Line")
            ax_reg.set_xlabel(col1)
            ax_reg.set_ylabel(col2)
            ax_reg.set_title(f"Linear Regression: {col2} vs {col1}")
            ax_reg.legend()
            st.pyplot(fig_reg)

else:
    st.info("Please upload a CSV or Excel file to begin.")
