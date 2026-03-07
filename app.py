import streamlit as st
import pandas as pd
import model

st.set_page_config(page_title="Recommendation Engine", layout="wide")

st.title("Predictive Recommendation Dashboard")
st.divider()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Configuration")

    uploaded = st.file_uploader("Upload CSV Dataset", type="csv")

    method = st.selectbox(
        "Algorithm",
        [
            "User-Based Collaborative Filtering",
            "Item-Based Collaborative Filtering",
            "Market Basket Analysis",
        ],
    )

# ---------------------------
# Load Dataset
# ---------------------------
if uploaded:
    try:
        data = pd.read_csv(uploaded)
    except:
        st.error("Unable to read the CSV file.")
        st.stop()
else:
    # Default dataset
    data = pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U2", "U2", "U3", "U3", "U4", "U4"],
            "item_id": [
                "Product_A",
                "Product_B",
                "Product_A",
                "Product_C",
                "Product_B",
                "Product_D",
                "Product_A",
                "Product_D",
            ],
            "rating": [5, 4, 4, 2, 5, 4, 3, 5],
        }
    )

# Validate dataset
required_cols = {"user_id", "item_id"}
if not required_cols.issubset(data.columns):
    st.error("Dataset must contain at least 'user_id' and 'item_id'")
    st.stop()

if "rating" not in data.columns:
    data["rating"] = 1


# ---------------------------
# Cache Heavy Computation
# ---------------------------
@st.cache_data(show_spinner="Computing similarity matrix...")
def run_engine(df, method):

    if method == "Market Basket Analysis":
        rules = model.run_market_basket(df)
        return rules, None

    target = "user" if "User-Based" in method else "item"

    pivot, sim_df = model.compute_similarity(df, target=target)

    return pivot, sim_df


# ---------------------------
# Layout
# ---------------------------
col_input, col_output = st.columns([1, 2], gap="large")


# ---------------------------
# Input Panel
# ---------------------------
with col_input:

    st.subheader("Controls")

    engine_data, sim_df = run_engine(data, method)

    if "Collaborative" in method:

        user_list = engine_data.index.tolist()

        selected_user = st.selectbox("Select User", user_list)

        k = st.slider("Number of Recommendations", 1, 10, 5)

        if st.button("Generate Recommendations"):

            if "User-Based" in method:

                recs = model.recommend_user_based(
                    selected_user,
                    engine_data,
                    sim_df,
                    k,
                )

            else:

                recs = model.recommend_item_based(
                    selected_user,
                    engine_data,
                    sim_df,
                    k,
                )

            st.session_state["results"] = recs

    else:

        item_list = sorted(data["item_id"].unique())

        selected_item = st.selectbox("Select Anchor Item", item_list)

        if st.button("Run Market Basket Analysis"):

            rules = engine_data[
                engine_data["antecedent"] == selected_item
            ]

            st.session_state["results"] = rules["consequent"].tolist()


# ---------------------------
# Output Panel
# ---------------------------
with col_output:

    st.subheader("Recommendations")

    if "results" in st.session_state:

        results = st.session_state["results"]

        if results:

            df_results = pd.DataFrame(
                results, columns=["Recommended Items"]
            )

            st.dataframe(df_results, use_container_width=True)

        else:

            st.info("No recommendations found.")

    else:

        st.caption("Click the button to generate recommendations.")

    st.divider()

    with st.expander("Dataset Preview"):

        st.dataframe(data.head(20), use_container_width=True)
