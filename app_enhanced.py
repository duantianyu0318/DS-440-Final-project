import pandas as pd
import streamlit as st
from datetime import datetime

from src.data_utils import get_stock_data, preprocess_data
from src.model_utils import build_model, plot_results, summarize_results, train_model

st.set_page_config(
    page_title='Enhanced Stock Prediction Platform',
    page_icon='📈',
    layout='wide'
)

# ---------------------------
# Session State Initialization
# ---------------------------
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

if 'favorite_tickers' not in st.session_state:
    st.session_state.favorite_tickers = []

if 'last_summary_df' not in st.session_state:
    st.session_state.last_summary_df = None

if 'detail_results' not in st.session_state:
    st.session_state.detail_results = {}

# ---------------------------
# Sidebar Navigation
# ---------------------------
with st.sidebar:
    page = st.radio("Navigation", ["Home", "Analysis", "User Dashboard"])

    st.markdown("---")

    if page == "Analysis":
        st.header('Configuration')
        tickers_text = st.text_input(
            'Stock tickers (comma-separated)',
            'AAPL, MSFT, TSLA'
        )
        period = st.selectbox(
            'Historical period',
            ['6mo', '1y', '2y', '5y'],
            index=2
        )
        window = st.slider(
            'Sliding window size',
            min_value=5,
            max_value=30,
            value=10,
            step=1
        )
        model_type = st.selectbox(
            'Prediction model',
            ['Linear Regression', 'Random Forest']
        )

        if st.button('Run Analysis', type='primary'):
            st.session_state.run_analysis = True

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.title('📈 Stock Prediction Platform')

    st.write(
        'This version improves the original stock predictor with a cleaner layout, '
        'multi-ticker analysis, model selection, configurable history windows, '
        'and a user dashboard for saved preferences and search history.'
    )

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.info('Cleaner front-end layout and navigation')
    info_col2.info('Multiple tickers with configurable period and window')
    info_col3.info('User Dashboard with search history and favorites')

    st.subheader('How to use this build')
    st.markdown(
        """
        1. Open the **Analysis** page from the sidebar.  
        2. Enter one or more stock tickers.  
        3. Choose a historical period, sliding window size, and model.  
        4. Click **Run Analysis** to generate predictions.  
        5. Open **User Dashboard** to review search history and saved favorites.  
        """
    )

    st.markdown(
        """
        ### Project overview
        This platform supports stock prediction and comparison across multiple tickers,
        while also adding user-oriented functionality such as search history tracking
        and favorite ticker management.
        """
    )

# ---------------------------
# Analysis Page
# ---------------------------
elif page == "Analysis":
    st.title("Stock Analysis Dashboard")

    if st.session_state.run_analysis:
        tickers = [t.strip().upper() for t in tickers_text.split(',') if t.strip()]
        summaries = []
        detail_results = {}

        for ticker in tickers:
            data = get_stock_data(ticker, period=period)
            bundle = preprocess_data(data, window=window)
            model = build_model(model_type)
            model = train_model(model, bundle['X_train'], bundle['y_train'])
            summary = summarize_results(model, bundle)

            summaries.append({
                'Ticker': ticker,
                'Latest Close ($)': round(summary['latest_close'], 2),
                'Predicted Next Close ($)': round(summary['next_close'], 2),
                'RMSE': round(summary['rmse'], 2),
                'MAE': round(summary['mae'], 2),
                'R²': round(summary['r2'], 3),
            })
            detail_results[ticker] = summary

        summary_df = pd.DataFrame(summaries)

        # Save latest results into session state
        st.session_state.last_summary_df = summary_df
        st.session_state.detail_results = detail_results

        # Save search history
        history_entry = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Tickers": ", ".join(tickers),
            "Period": period,
            "Window": window,
            "Model": model_type
        }
        st.session_state.search_history.append(history_entry)

        st.subheader('Prediction Summary')
        metric_cols = st.columns(min(3, len(summary_df)))
        for idx, row in summary_df.head(3).iterrows():
            delta = row['Predicted Next Close ($)'] - row['Latest Close ($)']
            metric_cols[idx].metric(
                label=f"{row['Ticker']} next close",
                value=f"${row['Predicted Next Close ($)']:.2f}",
                delta=f"{delta:.2f} vs latest"
            )

        st.dataframe(summary_df, width="stretch", hide_index=True)

        tab1, tab2, tab3 = st.tabs(['Detailed Chart', 'Method Summary', 'Front-End Notes'])

        with tab1:
            selected_ticker = st.selectbox(
                'Select ticker for chart view',
                summary_df['Ticker'].tolist()
            )
            fig = plot_results(
                detail_results[selected_ticker]['result_df'],
                selected_ticker,
                model_type
            )
            st.pyplot(fig, clear_figure=True)
            st.caption('Chart compares holdout-set actual prices against model predictions.')

        with tab2:
            st.markdown(
                f"""
                - **Data source:** Yahoo Finance  
                - **Preprocessing:** Sliding window of **{window}** closing prices  
                - **Model:** **{model_type}**  
                - **Evaluation metrics:** RMSE, MAE, and R²  
                - **Improvement over simple version:** multi-ticker support, user controls, richer outputs, and cleaner visual organization  
                """
            )

        with tab3:
            st.markdown(
                """
                **Front-end improvements included in this build:**
                1. Sidebar-based controls instead of a single text input  
                2. Multi-stock analysis in one run  
                3. Better result communication with metrics, summary table, and chart tab  
                4. User Dashboard page for tracking history and favorites  
                """
            )

    else:
        st.info("Set your parameters in the sidebar, then click Run Analysis.")

# ---------------------------
# User Dashboard Page
# ---------------------------
elif page == "User Dashboard":
    st.title("👤 User Dashboard")
    st.write("Review your recent activity, manage favorite tickers, and view your latest prediction results.")

    # Favorites Section
    st.subheader("Favorite Tickers")
    fav_col1, fav_col2 = st.columns([2, 1])

    with fav_col1:
        new_favorite = st.text_input("Add a ticker to favorites", placeholder="e.g. NVDA")

    with fav_col2:
        if st.button("Add Favorite"):
            if new_favorite:
                ticker_clean = new_favorite.strip().upper()
                if ticker_clean not in st.session_state.favorite_tickers:
                    st.session_state.favorite_tickers.append(ticker_clean)

    if st.session_state.favorite_tickers:
        st.write("Saved favorites:")
        st.write(", ".join(st.session_state.favorite_tickers))
    else:
        st.info("No favorite tickers saved yet.")

    st.markdown("---")

    # Search History Section
    st.subheader("Search History")
    if st.session_state.search_history:
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(history_df, width="stretch", hide_index=True)
    else:
        st.info("No search history available yet. Run an analysis first.")

    st.markdown("---")

    # Recent Results Section
    st.subheader("Recent Prediction Summary")
    if st.session_state.last_summary_df is not None:
        st.dataframe(st.session_state.last_summary_df, width="stretch", hide_index=True)
    else:
        st.info("No recent prediction results available yet.")
