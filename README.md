# DS-440-Final-project
Stock Market Prediction Platform using Python, Machine Learning, and Streamlit.

# Stock Market Prediction Platform

This project is a Stock Market Prediction Platform developed for IST 440.

## Features
- Real-time stock data retrieval using Yahoo Finance API
- Data preprocessing (normalization, sliding window)
- Machine learning prediction (Linear Regression, Random Forest)
- Visualization of actual vs predicted values
- User-friendly interface using Streamlit

## Technologies
- Python
- Streamlit
- scikit-learn
- yfinance

## How to Run
1. Download Python and set up the environment.
   https://www.python.org/downloads/

2. Open a terminal or command prompt.

3. Click "Code," then download the ZIP file to your desktop.
   Return to the desktop, click on the downloaded file, then right-click on the file and select "Open in Terminal."

4. In the terminal install dependencies:
   pip install -r requirements.txt

   If requirements.txt is missing:
   pip install streamlit pandas numpy scikit-learn matplotlib yfinance
   or
   python -m pip install streamlit pandas numpy scikit-learn matplotlib yfinance

5. In the terminal run the app:
   streamlit run app_enhanced.py
   or
   python -m streamlit run app_enhanced.py

## Authors
- Tianyu Duan
- Songqi Lin
