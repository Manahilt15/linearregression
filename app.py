import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Streamlit app
def main():
    st.title("Linear Regression App")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())

        # Select x and y variables
        columns = data.columns.tolist()
        x_var = st.selectbox("Select X variable:", columns)
        y_var = st.selectbox("Select Y variable:", columns)

        if st.button("Perform Linear Regression"):
            # Extracting selected columns
            X = data[[x_var]]
            y = data[y_var]

            # Perform linear regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # Display results
            st.write("Intercept:", model.intercept_)
            st.write("Coefficient:", model.coef_[0])

            # Plotting the regression line
            fig, ax = plt.subplots()
            ax.scatter(X, y, color="blue", label="Actual Data")
            ax.plot(X, y_pred, color="red", label="Regression Line")
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
