import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Page setting
# -----------------------------
st.set_page_config(
    page_title="Property Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f7f7f7;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    possible_paths = [
        Path("property_price_data.csv"),
        Path("property_price_data(1).csv"),
        Path("/mnt/data/property_price_data.csv"),
        Path("/mnt/data/property_price_data(1).csv"),
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    return None

# -----------------------------
# Helper functions
# -----------------------------
def get_basic_info(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (df.isnull().mean() * 100).round(2).values,
        "Unique Values": df.nunique().values
    })


def prepare_model_data(df):
    data = df.copy()

    if "SalePrice" not in data.columns:
        st.error("SalePrice column is missing. Please check the dataset.")
        st.stop()

    y = np.log1p(data["SalePrice"])
    X = data.drop(columns=["SalePrice"])

    if "Prop_Id" in X.columns:
        X = X.drop(columns=["Prop_Id"])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return X, y, preprocessor, numeric_cols, categorical_cols


@st.cache_resource
def train_model(df, model_name):
    X, y, preprocessor, numeric_cols, categorical_cols = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_name == "Ridge Regression":
        model = Ridge(alpha=10)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    pred_log = pipe.predict(X_test)
    y_test_actual = np.expm1(y_test)
    pred_actual = np.expm1(pred_log)

    mae = mean_absolute_error(y_test_actual, pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))
    r2 = r2_score(y_test_actual, pred_actual)

    results = pd.DataFrame({
        "Actual Price": y_test_actual,
        "Predicted Price": pred_actual
    })
    results["Error"] = results["Actual Price"] - results["Predicted Price"]

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return pipe, metrics, results, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def get_feature_importance(pipe, numeric_cols, categorical_cols):
    try:
        preprocessor = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["model"]

        feature_names = []
        feature_names.extend(numeric_cols)

        if len(categorical_cols) > 0:
            encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(cat_feature_names)

        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance_values = np.abs(model.coef_)
        else:
            return None

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_values
        }).sort_values("Importance", ascending=False).head(20)

        return importance_df
    except Exception:
        return None


def make_prediction_input(df):
    X = df.drop(columns=["SalePrice"], errors="ignore").copy()
    X = X.drop(columns=["Prop_Id"], errors="ignore")

    input_row = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            input_row[col] = X[col].median()
        else:
            input_row[col] = X[col].mode().iloc[0] if not X[col].mode().empty else "None"

    return input_row, X


def format_price(value):
    return f"${value:,.0f}"

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏠 Property Dashboard")
st.sidebar.write("Use this dashboard to explore the data and predict property prices.")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
df = load_data(uploaded_file)

if df is None:
    st.warning("Please upload the property price CSV file or keep it in the same folder as app.py.")
    st.stop()

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Data Overview",
        "EDA Charts",
        "Missing Values",
        "Model Performance",
        "Predict Price",
        "Insights"
    ]
)

# -----------------------------
# Home page
# -----------------------------
if page == "Home":
    st.markdown('<div class="main-title">Property Price Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">This dashboard shows data analysis, model results, and price prediction for properties.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Average Price", format_price(df["SalePrice"].mean()) if "SalePrice" in df.columns else "NA")
    col4.metric("Highest Price", format_price(df["SalePrice"].max()) if "SalePrice" in df.columns else "NA")

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Project Aim")
    st.write(
        "The aim of this project is to understand property data and build a machine learning model "
        "that can predict the sale price of a property based on different features."
    )

# -----------------------------
# Data overview
# -----------------------------
elif page == "Data Overview":
    st.title("Data Overview")
    st.write("Here I checked the basic structure of the dataset.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", df.duplicated().sum())

    st.subheader("Column Information")
    info_df = get_basic_info(df)
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

# -----------------------------
# EDA charts
# -----------------------------
elif page == "EDA Charts":
    st.title("EDA Charts")
    st.write("This section helps to understand the pattern in property prices and other features.")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if "SalePrice" in df.columns:
        st.subheader("SalePrice Distribution")
        fig = px.histogram(df, x="SalePrice", nbins=40, title="Distribution of SalePrice")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox("Select numerical feature", numeric_cols, index=numeric_cols.index("GrLivArea") if "GrLivArea" in numeric_cols else 0)
    with col2:
        color_col = st.selectbox("Color by category", [None] + categorical_cols, index=0)

    if "SalePrice" in df.columns:
        fig = px.scatter(
            df,
            x=x_col,
            y="SalePrice",
            color=color_col,
            title=f"{x_col} vs SalePrice"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Price by Category")
    if categorical_cols and "SalePrice" in df.columns:
        cat_col = st.selectbox("Select category column", categorical_cols, index=categorical_cols.index("Neighborhood") if "Neighborhood" in categorical_cols else 0)
        avg_price = df.groupby(cat_col)["SalePrice"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_price, x=cat_col, y="SalePrice", title=f"Average SalePrice by {cat_col}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    if len(numeric_cols) > 1:
        selected_corr_cols = st.multiselect(
            "Select columns for correlation",
            numeric_cols,
            default=[col for col in ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "YearBuilt"] if col in numeric_cols]
        )
        if len(selected_corr_cols) >= 2:
            corr = df[selected_corr_cols].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Missing values
# -----------------------------
elif page == "Missing Values":
    st.title("Missing Value Analysis")
    st.write("Here I checked which columns have missing values.")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        st.success("No missing values found in the dataset.")
    else:
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
        fig = px.bar(missing_df, x="Column", y="Missing %", title="Missing Value Percentage")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simple Missing Value Treatment Used")
    st.write(
        "For model building, numerical missing values are filled with median values. "
        "Categorical missing values are filled with 'None'. This is useful because some missing values "
        "mean that the feature is not available, like no pool, no alley, or no fireplace."
    )

# -----------------------------
# Model performance
# -----------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Here I trained a regression model and checked how well it predicts property prices.")

    model_name = st.selectbox(
        "Select Model",
        ["Ridge Regression", "Random Forest", "Gradient Boosting"]
    )

    with st.spinner("Training model..."):
        pipe, metrics, results, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = train_model(df, model_name)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", format_price(metrics["MAE"]))
    col2.metric("RMSE", format_price(metrics["RMSE"]))
    col3.metric("R² Score", round(metrics["R2"], 3))

    st.subheader("Actual vs Predicted Price")
    fig = px.scatter(
        results,
        x="Actual Price",
        y="Predicted Price",
        title="Actual Price vs Predicted Price"
    )
    fig.add_shape(
        type="line",
        x0=results["Actual Price"].min(),
        y0=results["Actual Price"].min(),
        x1=results["Actual Price"].max(),
        y1=results["Actual Price"].max(),
        line=dict(dash="dash")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Error")
    fig = px.histogram(results, x="Error", nbins=40, title="Distribution of Prediction Error")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")
    importance_df = get_feature_importance(pipe, numeric_cols, categorical_cols)
    if importance_df is not None:
        fig = px.bar(importance_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Top Important Features")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model.")

# -----------------------------
# Predict price
# -----------------------------
elif page == "Predict Price":
    st.title("Predict Property Price")
    st.write("Enter property details below and the model will predict the sale price.")

    model_name = st.selectbox(
        "Select Model for Prediction",
        ["Gradient Boosting", "Random Forest", "Ridge Regression"]
    )

    pipe, metrics, results, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = train_model(df, model_name)

    input_row, X = make_prediction_input(df)

    st.subheader("Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "OverallQual" in X.columns:
            input_row["OverallQual"] = st.slider("Overall Quality", 1, 10, int(input_row["OverallQual"]))
        if "GrLivArea" in X.columns:
            input_row["GrLivArea"] = st.number_input("Living Area (sq ft)", min_value=0, value=int(input_row["GrLivArea"]))
        if "TotalBsmtSF" in X.columns:
            input_row["TotalBsmtSF"] = st.number_input("Basement Area (sq ft)", min_value=0, value=int(input_row["TotalBsmtSF"]))
        if "YearBuilt" in X.columns:
            input_row["YearBuilt"] = st.number_input("Year Built", min_value=1800, max_value=2030, value=int(input_row["YearBuilt"]))

    with col2:
        if "GarageCars" in X.columns:
            input_row["GarageCars"] = st.slider("Garage Cars", 0, 5, int(input_row["GarageCars"]))
        if "GarageArea" in X.columns:
            input_row["GarageArea"] = st.number_input("Garage Area", min_value=0, value=int(input_row["GarageArea"]))
        if "FullBath" in X.columns:
            input_row["FullBath"] = st.slider("Full Bathrooms", 0, 5, int(input_row["FullBath"]))
        if "BedroomAbvGr" in X.columns:
            input_row["BedroomAbvGr"] = st.slider("Bedrooms", 0, 10, int(input_row["BedroomAbvGr"]))

    with col3:
        if "Neighborhood" in X.columns:
            input_row["Neighborhood"] = st.selectbox("Neighborhood", sorted(X["Neighborhood"].dropna().unique().tolist()))
        if "MSZoning" in X.columns:
            input_row["MSZoning"] = st.selectbox("MS Zoning", sorted(X["MSZoning"].dropna().unique().tolist()))
        if "PropStyle" in X.columns:
            input_row["PropStyle"] = st.selectbox("Property Style", sorted(X["PropStyle"].dropna().unique().tolist()))
        if "KitchenQual" in X.columns:
            input_row["KitchenQual"] = st.selectbox("Kitchen Quality", sorted(X["KitchenQual"].dropna().unique().tolist()))

    more_options = st.expander("More options")
    with more_options:
        col4, col5, col6 = st.columns(3)
        with col4:
            if "ExterQual" in X.columns:
                input_row["ExterQual"] = st.selectbox("Exterior Quality", sorted(X["ExterQual"].dropna().unique().tolist()))
            if "BsmtQual" in X.columns:
                input_row["BsmtQual"] = st.selectbox("Basement Quality", sorted(X["BsmtQual"].dropna().unique().tolist()))
        with col5:
            if "GarageType" in X.columns:
                input_row["GarageType"] = st.selectbox("Garage Type", sorted(X["GarageType"].fillna("None").unique().tolist()))
            if "CentralAir" in X.columns:
                input_row["CentralAir"] = st.selectbox("Central Air", sorted(X["CentralAir"].dropna().unique().tolist()))
        with col6:
            if "SaleCondition" in X.columns:
                input_row["SaleCondition"] = st.selectbox("Sale Condition", sorted(X["SaleCondition"].dropna().unique().tolist()))
            if "TotRmsAbvGrd" in X.columns:
                input_row["TotRmsAbvGrd"] = st.slider("Total Rooms", 1, 15, int(input_row["TotRmsAbvGrd"]))

    if st.button("Predict Price"):
        input_df = pd.DataFrame([input_row])
        prediction_log = pipe.predict(input_df)[0]
        prediction = np.expm1(prediction_log)
        st.success(f"Predicted Property Price: {format_price(prediction)}")
        st.caption("This is an estimated price based on the trained machine learning model.")

# -----------------------------
# Insights
# -----------------------------
elif page == "Insights":
    st.title("Business Insights")
    st.write("These are the main points I found from the analysis and model.")

    st.subheader("Important Points")
    st.write("1. Properties with higher overall quality usually have higher sale prices.")
    st.write("2. Bigger living area is strongly related to higher property price.")
    st.write("3. Garage size and basement area also affect the price.")
    st.write("4. Location or neighborhood can make a big difference in the property value.")

    st.subheader("Recommendations")
    st.write("Sellers can improve property value by improving quality, kitchen condition, living space, and garage area.")
    st.write("Buyers can compare properties using important features like area, quality, year built, and location.")

    st.subheader("Limitations")
    st.write("The model is trained only on the given dataset, so it may not work perfectly for every city or market.")
    st.write("More recent market data and location details can improve the prediction in future.")
