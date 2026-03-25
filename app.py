import streamlit as st
import pandas as pd
from model import RecommenderEngine

st.set_page_config(page_title="AI Recommender Studio", layout="wide", page_icon="📊")

# --- Sample Data for Demonstration ---
def load_sample_data():
    # Expanded dataset: 20 users, 10 items, varied ratings (1-5)
    data = {
        'User': [
            'Alice','Alice','Alice','Bob','Bob','Bob','Charlie','Charlie','Charlie','David','David',
            'Eve','Eve','Eve','Frank','Frank','Grace','Grace','Heidi','Heidi','Ivan','Ivan',
            'Judy','Judy','Mallory','Mallory','Niaj','Niaj','Olivia','Olivia','Peggy','Peggy',
            'Sybil','Sybil','Trent','Trent','Victor','Victor','Walter','Walter'
        ],
        'Item': [
            'Elden Ring','Halo','Zelda','Elden Ring','Tetris','COD','Halo','FIFA','Minecraft',
            'COD','FIFA','Elden Ring','Halo','Zelda','Halo','Minecraft','Zelda','Mario Kart',
            'Elden Ring','Mario Kart','FIFA','COD','Zelda','Mario Kart','Tetris','Minecraft',
            'Halo','COD','Elden Ring','Zelda','FIFA','Mario Kart','Tetris','Halo',
            'Zelda','COD','Elden Ring','FIFA','Minecraft','Mario Kart'
        ],
        'Rating': [
            5,4,5,5,3,2,4,5,4,4,5,5,5,4,4,5,5,5,5,4,2,3,4,5,3,4,4,5,5,5,2,5,3,4,5,2,5,4,5,5
        ]
    }
    return pd.DataFrame(data)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Strategy Selector
    strategy = st.radio(
        "Recommendation Strategy:",
        ["User-Based CF", "Item-Based CF", "Market Basket Analysis"]
    )
    
    st.divider()
    
    # Data Source Selector
    data_option = st.selectbox("Data Source:", ["Use Built-in Sample", "Upload My Own CSV"])
    
    if data_option == "Upload My Own CSV":
        file = st.file_uploader("Upload CSV", type="csv")
        if file:
            df = pd.read_csv(file)
        else:
            st.info("Please upload a CSV file to proceed.")
            st.stop()
    else:
        df = load_sample_data()
        st.success("Sample data loaded!")

    # Template Download
    template = pd.DataFrame(columns=['user_id', 'item_id', 'rating']).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV Template", data=template, file_name="template.csv")

# --- Main App Interface ---
st.title("🎯 AI Recommendation Engine")
st.markdown(f"Currently using: **{strategy}**")

# Initialize Engine
engine = RecommenderEngine(df)

# UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Selection")
    users = df.iloc[:, 0].unique()
    selected_user = st.selectbox("Target User:", users)
    num_recs = st.slider("Results count:", 1, 10, 5)
    
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)

with col2:
    st.subheader("Generated Recommendations")
    
    if st.button(f"Run {strategy}"):
        with st.spinner("Processing math..."):
            if strategy == "User-Based CF":
                recs = engine.get_user_based(selected_user, n=num_recs)
                note = "Logic: Finding people who rate items similarly to you."
            elif strategy == "Item-Based CF":
                recs = engine.get_item_based(selected_user, n=num_recs)
                note = "Logic: Finding items that are similar to what you've liked."
            else:
                recs = engine.get_market_basket(selected_user, n=num_recs)
                note = "Logic: People who bought what you bought also picked these."

            if not recs.empty:
                st.info(note)
                # Convert results to a clean table
                res_df = pd.DataFrame({
                    "Recommended Item": recs.index,
                    "Strength Score": recs.values.round(2)
                })
                st.table(res_df)
                st.balloons()
            else:
                st.warning("Not enough data to find recommendations for this user.")

st.divider()
st.caption("Developed by Abhishek Jha | Collaborative Filtering & Association Rules Engine")
