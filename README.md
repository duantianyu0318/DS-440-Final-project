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

3. Use the following code to run the code.
   git clone https://github.com/duantianyu0318/DS-440-Final-project.git
   cd DS-440-Final-project

4. Open the terminal in the project directory and create a virtual environment.
   python -m venv venv
   venv\Scripts\activate

5. In the terminal install dependencies:
   pip install -r requirements.txt

   If requirements.txt is missing:
   pip install streamlit pandas numpy scikit-learn matplotlib yfinance

6. In the terminal run the app:
   streamlit run app_enhanced.py

## Authors
- Tianyu Duan
- Songqi Lin
