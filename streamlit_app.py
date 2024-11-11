# Installing dependencies
# Installing dependencies
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt.efficient_frontier import EfficientFrontier
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

st.title("Portfolio Optimization Tool💸")
st.markdown("<a name='basic-description'></a>", unsafe_allow_html=True)
st.markdown("""
    **Welcome to the Portfolio Theory Educational Tool!**

    This app is designed to help you learn about modern portfolio theory and explore portfolio optimization techniques (Modern Portfolio Theory and Mean-Variance Optimization). With this tool, you can:

    - **Select a list of stocks** and explore their performance metrics and historical prices.
    - **Choose the stocks** you'd like to include in your portfolio.
    - **Optimize your portfolio** using one of three methods:
        - **Markowitz** (minimizing volatility)
        - **Max Sharpe Ratio** (maximizing the Sharpe ratio)
        - **Min Conditional Value at Risk (CVaR)** (minimizing CVaR)
    - After optimization, you can **choose a method for discrete allocation**, where you can input your total investment amount, see how many shares of each stock to buy, and view any leftover money.
    
    If you don't know anything about portfolio theory I recommend you [this lecture](https://www.youtube.com/watch?v=ywl3pq6yc54), its pretty good!
""")
# Defining Base functions

# Function to get stock data
# Función para obtener los indicadores deseados sobre las acciones elegidas usando las APIs de Yahooo Finance
@st.cache_data
def getStocks(tickers):
    # Defining indicators
    fields = {
        'Company': 'shortName',
        'Ticker': 'symbol',
        'Exchange': 'exchange',
        'Country': 'country',
        'Sector': 'sector',
        'Industry': 'industry',
        'Dividend Yield': 'dividendYield',
        'Earnings Per Share': 'trailingEps',
        'Price To Earnings': 'trailingPE',
        'Operations Margin': 'operatingMargins',
        'Net Margin': 'profitMargins',
        'ROA': 'returnOnAssets',
        'ROE': 'returnOnEquity',
        'EBITDA': 'ebitda',
        'Beta': 'beta'
        }

    # Creating a list to hold stock data
    data_stock = []

    # Iterating over tickers to extract desired information
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extracting desired information
        row = {field: info.get(yf_field, 'N/A') for field, yf_field in fields.items()}

        data_stock.append(row)

    # Creating a DataFrame
    df_info = pd.DataFrame(data_stock)

    # Calculating expected returns and standard deviations
    df_ret = pd.DataFrame({})

    # Obtaining the daily closing prices from the las 10 years
    ohlc = yf.download(tickers, period='10y', interval='1d')
    data_prices = ohlc['Adj Close'].dropna(how='all')

    # Iterating over the stock list
    for col_name, col_data in data_prices.items():
        # Calculating daily returns
        data = col_data.pct_change()
        # Calculating expected returns and standard deviations
        yearDays = 252
        yearlyRet = (1 + data.mean()) ** yearDays - 1
        yearlyStd = data.std() * np.sqrt(yearDays)
        # Calculating the Sharpe Ratio of the stocks
        rf = 0.04220
        sharpe = (yearlyRet-rf)/yearlyStd
        # Adding results to the dataframe
        result = {'Ticker':col_name,'Expected Return':yearlyRet,'Std Deviation':yearlyStd,'Sharpe Ratio':sharpe}
        df_ret = pd.concat([df_ret,pd.DataFrame([result])], ignore_index=True)

    # Merging dataframes
    df = pd.merge(df_info,df_ret,on='Ticker')

    # Formatting columns as percentages
    percentage_cols = ['Dividend Yield', 'Operations Margin', 'Net Margin', 'ROA', 'ROE', 'Expected Return', 'Std Deviation']
    for col in percentage_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') * 100
            df[col] = df[col].apply(lambda x: '{:.2f}%'.format(x) if pd.notnull(x) else 'N/A')

    # Setting 'Ticker' as the index and adding 'Selected' column
    df['Selected'] = True
    df = df[['Selected'] + [col for col in df.columns if col != 'Selected']]
    df.set_index('Ticker', inplace=True)

    # Returning the tha dataframe with the data
    return df

@st.cache_data
# Function to obtain historic daily closing prices from a list of tickers
def stocksPriceReturn(tickers):
    # Obtaining price data
    ohlc = yf.download(tickers, period='max', interval='1d')
    prices = ohlc['Adj Close'].dropna(how='all')
    # Returning output dataframe
    return prices

# Function to optimize portfolio using different objectives (min volatility, max sharpe, min cvar)
def portOptimize(returns, cov_matrix, prices, optimization_type='min_volatility', risk_free_rate=0.0422):
    """
    Optimize the portfolio using the Markowitz Efficient Frontier.

    Parameters:
    - returns: Expected returns of the assets.
    - cov_matrix: Covariance matrix of the assets' returns.
    - prices: Historical price data of the assets.
    - optimization_type: Type of optimization to perform ('min_volatility', 'max_sharpe', 'min_cvar').
    - risk_free_rate: The risk-free rate for Sharpe ratio calculation.

    Returns:
    - tickers: List of tickers in the portfolio.
    - weights_val: Portfolio weights as percentages.
    - perf: Dictionary with portfolio performance metrics.
    """
    
    # Initialize the EfficientFrontier object
    ef = EfficientFrontier(returns, cov_matrix, weight_bounds=(0, 1))
    
    # Perform the chosen optimization
    if optimization_type == 'min_volatility':
        ef.min_volatility()  # Minimize portfolio volatility
    elif optimization_type == 'max_sharpe':
        ef.max_sharpe(risk_free_rate=risk_free_rate)  # Maximize Sharpe ratio
    elif optimization_type == 'min_cvar':
        ef_VaR = EfficientCVaR(returns, cov_matrix)  # Initialize EfficientCVaR for CVaR optimization
        ef_VaR.min_cvar()  # Minimize Conditional VaR
    else:
        raise ValueError("Invalid optimization type. Choose from 'min_volatility', 'max_sharpe', or 'min_cvar'.")
    
    # Get optimal portfolio weights
    weights = ef.clean_weights() if optimization_type != 'min_cvar' else ef_VaR.clean_weights()
    
    # Performance metrics for 'min_volatility' and 'max_sharpe'
    if optimization_type in ['min_volatility', 'max_sharpe']:
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        # Calculate CVaR for 'min_volatility' and 'max_sharpe' methods
        weights_arr = np.array(list(weights.values()))
        rets = expected_returns.returns_from_prices(prices).dropna()
        rets_sum = (rets * weights_arr).sum(axis=1)
        var = rets_sum.quantile(0.05)  # 5% quantile for VaR
        cvar = rets_sum[rets_sum <= var].mean()  # Conditional VaR
        perf = dict({
            'Expected Returns': str(round(perf[0] * 100, 2)) + '%',
            'Volatility': str(round(perf[1] * 100, 2)) + '%',
            'Sharpe Ratio': str(round(perf[2], 2)),
            'Conditional VaR': str(round(cvar * 100, 2)) + '%'
        })
    else:
        # For 'min_cvar', use the CVaR value directly from the EfficientCVaR object
        cvar = ef_VaR.portfolio_performance()[1]  # CVaR from portfolio_performance()

        # Set the weights in the EfficientFrontier object to get other performance metrics
        ef_mincvar = EfficientFrontier(returns, cov_matrix)
        ef_mincvar.set_weights(weights)  # Set the weights from min_cvar optimization
        ReCVaR, StdCVaR, SharpeCVaR = ef_mincvar.portfolio_performance(risk_free_rate=risk_free_rate)
        
        # Format the performance metrics
        perf = dict({
            'Expected Returns': str(round(ReCVaR * 100, 2)) + '%',
            'Volatility': str(round(StdCVaR * 100, 2)) + '%',
            'Sharpe Ratio': str(round(SharpeCVaR, 2)),
            'Conditional VaR': str(round(cvar * 100, 2)) + '%'
        })
    
    # Get tickers and weights
    tickers = list(weights.keys())
    weights_val = list(weights.values())
    
    # Round the weights to two decimal places
    for index, weight in enumerate(weights_val):
        weights_val[index] = round(weight * 100, 2)
    
    # Return the results
    return tickers, weights_val, perf, weights


# Function to display portfolio dashboard with specified titles and layout
def display_portfolio_dashboard(tickers, weights, perf, optimization_type, fig_key):
    # Determine the title based on the optimization type
    if optimization_type == 'min_volatility':
        dashboard_title = "Markowitz Portfolio Performance Dashboard"
    elif optimization_type == 'max_sharpe':
        dashboard_title = "Max. Sharpe Ratio Portfolio Performance Dashboard"
    elif optimization_type == 'min_cvar':
        dashboard_title = "Min. Conditional VaR Portfolio Performance Dashboard"
    else:
        dashboard_title = "Portfolio Performance Dashboard"

    # Create a collapsible container for displaying key metrics
    with st.expander(dashboard_title, expanded=False):
        # Create a Plotly donut chart to display weights
        fig = px.pie(
            names=tickers,
            values=weights,
            title="Portfolio Allocation",
            hole=0.6,  # Donut chart
            labels={'names': 'Tickers', 'values': 'Weights (%)'}
        )
        fig.update_traces(textinfo='percent+label')

        # Display the donut chart in the container
        st.plotly_chart(fig, fig_key)

        st.markdown("**Key Portfolio Metrics**")
        # Create a 2x2 grid layout for displaying the metrics
        cols = st.columns(2)
        metric_names = list(perf.keys())
        metric_values = list(perf.values())
        
        for i, col in enumerate(cols * 2):  # Repeat columns to fill a 2x2 grid
            if i < len(metric_names):
                # Display each metric in a card-like div
                with col:
                    st.markdown(
                        f"""
                        <div style='background-color: #35d257; padding: 15px; border-radius: 5px; margin-bottom: 10px; color:white; font-size: 18px;'>
                            {metric_names[i]}: <strong>{metric_values[i]}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

def display_allocation_dashboard(weights, selected_prices):
    """
    Displays a dashboard for portfolio allocation based on the output of markoOptimize().
    
    Args:
        marko_output (tuple): The output from the markoOptimize function, 
                              including (tickers, weights_val, perf, weights_arr).
        selected_prices (DataFrame): DataFrame of the selected stock prices.
    """

    # Input box for the investment amount
    investment_amount = st.number_input("Enter the total investment amount", min_value=1000.0, step=1000.0)

    # Get the latest prices of the selected stocks
    latest_prices = get_latest_prices(selected_prices)

    # Create a DiscreteAllocation object
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = da.greedy_portfolio()

    # Display allocation details and leftover money
    st.markdown(
                        f"""
                        <div style='background-color: #35d257; padding: 15px; border-radius: 5px; margin-bottom: 10px; color:white; font-size: 18px;'>
                            Leftover Money: <strong>${leftover:.2f}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # Prepare data for the bar plots
    allocation_values = [allocation[ticker] * latest_prices[ticker] for ticker in allocation]
    allocation_labels = list(allocation.keys())

    # Plot the number of shares
    fig_shares = go.Figure(data=go.Bar(
        x=list(allocation.values()),
        y=allocation_labels,
        orientation='h',
        marker=dict(color='green')
    ))
    fig_shares.update_layout(
        title="Number of Shares to Buy per Stock",
        xaxis_title="Number of Shares",
        yaxis_title="Ticker"
    )
    st.plotly_chart(fig_shares, key="shares_plot")

    # Plot the amount spent on each stock
    fig_spent = go.Figure(data=go.Bar(
        x=allocation_values,
        y=allocation_labels,
        orientation='h',
        marker=dict(color='blue')
    ))
    fig_spent.update_layout(
        title="$ Amount Spent per Stock",
        xaxis_title="Amount Spent ($)",
        yaxis_title="Ticker"
    )
    st.plotly_chart(fig_spent, key="spent_plot")

# Streamlit Development section

# Input widget for tickers as a comma-separated string
st.subheader("Stock Selection 🔍")
st.markdown("<a name='stock-selection'></a>", unsafe_allow_html=True)
st.markdown("""
To build your portfolio, the first step is to choose which stocks you want to include. While stock selection is a detailed process that involves many factors, here you can easily select stocks by entering their tickers (unique identifiers) **separated by commas**. Once you've entered your list, you'll be able to see some key performance metrics about each stock.

We use data from the **Yahoo Finance API** to provide you with real-time stock information. Be sure to explore the [Yahoo Finance platform](https://finance.yahoo.com/) and choose stocks that align with your investment goals and interests.
            
P.S. With the Dropdown menu you can filter the columns in the table, and with the 'Selected' column you can choose which stocks to include in your portfolio.
""")

tickers_input = st.text_input("Enter tickers separated by commas", placeholder="e.g.: AAPL,MSFT,TSLA")
tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]

# Check if the tickers list is empty
if not tickers:
    st.warning("Please enter at least one ticker to display data.")
    st.stop()  # Stop execution until valid input is provided

# Load DataFrame in session state if not already loaded or if tickers list changes
if 'prev_tickers' not in st.session_state or st.session_state.prev_tickers != tickers:
    st.session_state.prev_tickers = tickers
    st.session_state.df_stocks = getStocks(tickers)

# Collapsible section for selecting columns to display
with st.expander("Select columns to display"):
    selected_columns = ['Selected']  # Always include 'Selected'
    columns = [col for col in st.session_state.df_stocks.columns if col != 'Selected']
    
    # Creating a 6x3 grid for checkboxes
    num_columns = 3
    num_rows = 6
    col_chunks = [columns[i:i + num_columns] for i in range(0, len(columns), num_columns)]

    for chunk in col_chunks:
        cols = st.columns(num_columns)
        for col_obj, col_name in zip(cols, chunk):
            with col_obj:
                if st.checkbox(f"Show {col_name}", value=True):
                    selected_columns.append(col_name)

# Display and allow editing of the 'Selected' column only
disabled_columns = [col for col in st.session_state.df_stocks.columns if col != 'Selected']

# Filter the DataFrame to show only selected columns
df_stocks_edit = st.data_editor(st.session_state.df_stocks[selected_columns], disabled=disabled_columns)

# Update the session state with the edited DataFrame
st.session_state.df_stocks.update(df_stocks_edit)

# Get the list of tickers where 'Selected' is True
selected_tickers = df_stocks_edit[df_stocks_edit['Selected'] == True].index.tolist()

# Obtaining historic daily closing prices from the selected tickers
selected_prices = stocksPriceReturn(selected_tickers)

# Resample to get monthly closing prices (using the last available price of each month)
monthly_prices = selected_prices.resample('ME').last()

# Display historical prices and allow filtering by tickers
if selected_tickers:
    st.markdown("<h2 style='font-size:22px;'>Historic Stock Prices 📈</h2>", unsafe_allow_html=True)
    st.markdown("""
    Here you can see the historic prices of the stocks you selected!
                
    You can filter the years displayed in the plot with the slider and which stocks are shown with the dropdown menu.
    """)
    # Slider to filter years displayed in the line plot
    min_year = monthly_prices.index.min().year
    max_year = monthly_prices.index.max().year
    year_range = st.slider("Select year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    # Filter the data based on the selected year range using proper datetime filtering
    filtered_prices = monthly_prices.loc[
        (monthly_prices.index >= f"{year_range[0]}-01-01") & (monthly_prices.index <= f"{year_range[1]}-12-31")
    ]

    # Ensure the filtered_prices DataFrame has the tickers after filtering
    filtered_prices = filtered_prices.dropna(axis=1, how='all')

    # Checkbox for filtering which tickers to show in the chart
    with st.expander("Filter Tickers to Display"):
        displayed_tickers = []
        cols = st.columns(3)
        for i, ticker in enumerate(filtered_prices.columns):  # Iterate only over columns that are present after filtering
            with cols[i % 3]:
                if st.checkbox(f"{ticker}", value=True):
                    displayed_tickers.append(ticker)

    # Plot the filtered data
    if displayed_tickers:
        st.line_chart(filtered_prices[displayed_tickers])
    else:
        st.write("Select at least one ticker to display.")

# Obtaining the expected reutrns of the selected tickers through the CAPM method
capm_rets = expected_returns.capm_return(selected_prices)

# Getting exponential covariance Matrix of the selected stocks
cov_matrix = risk_models.risk_matrix(selected_prices, method='exp_cov')

# Convert the index to a timezone-unaware datetime to avoid the UTC comparison error
selected_prices.index = selected_prices.index.tz_localize(None)
# Getting the data from the last 5 years
last_5_years_data = selected_prices[selected_prices.index >= pd.to_datetime("today") - pd.DateOffset(years=5)]
# Getting the daily returns from the last 5 years
last_5_years_rets = last_5_years_data.pct_change()
# Getting the correlation matrix
corr_matrix = last_5_years_rets.corr()

# Display the covariance and correlation matrices side by side
st.subheader("Risk and Returns Models 🤖")
st.markdown("<a name='risk-and-returns-models'></a>", unsafe_allow_html=True)
st.markdown("""
Once you've chosen the stocks for your portfolio, the next step is to model the risk and returns. These models provide the foundation for optimizing your portfolio, guiding your investment decisions.

There are various methods and extensive research dedicated to finding the best ways to model these aspects. For simplicity and effectiveness, we have chosen two well-regarded models:

- **[CAPM (Capital Asset Pricing Model)](https://www.investopedia.com/terms/c/capm.asp)**: CAPM is a widely used approach for estimating the expected return of an asset based on its risk in relation to the overall market. It assumes that investors need to be compensated for both the time value of money and the risk taken. This model helps you estimate returns by considering the relationship between the expected market return and the risk (or beta) of each stock in your portfolio.
  
- **[Exponential Covariance](https://reasonabledeviations.com/2018/08/15/exponential-covariance/)**: To model risk, we use the Exponential Covariance matrix. This method gives more weight to recent data, allowing for a more responsive and realistic measure of risk that adjusts to market changes. It ensures that the model remains up-to-date with the latest market conditions, providing a more accurate view of potential risk compared to standard covariance matrices.

These models help balance the potential for returns with the level of risk, serving as the basis for the optimization methods available in this app. If you want to learn more, read the following [article](https://www.financestrategists.com/wealth-management/investment-management/portfolio-optimization/)

In the dropdown menu you can see the Correlation and Covariance matrices of the stocks you selected!
""")

# Creating collapsible containers
with st.expander("Correlation and Covariance Matrices"):
    # Create a Plotly heatmap for the correlation matrix
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        colorbar_title="Correlation"
    ))
    fig_corr.update_layout(
        title="Correlation Matrix",
        xaxis_title="Tickers",
        yaxis_title="Tickers"
    )

    # Create a Plotly heatmap for the covariance matrix
    fig_cov = go.Figure(data=go.Heatmap(
        z=cov_matrix.values,
        x=cov_matrix.columns,
        y=cov_matrix.index,
        colorscale='Viridis',
        colorbar_title="Covariance"
    ))
    fig_cov.update_layout(
        title="Covariance Matrix",
        xaxis_title="Tickers",
        yaxis_title="Tickers"
    )

    # Display the plots
    st.plotly_chart(fig_corr, key="correlation_chart")
    st.plotly_chart(fig_cov, key="covariance_chart")

# Displaying the optimal portfolios
st.subheader("Portfolio Optimization 🔝")
st.markdown("<a name='portfolio-optimization'></a>", unsafe_allow_html=True)
st.markdown("""
Now comes the fun part!
            
In this section we will create three optimal portfolios with the stocks you picked. 
            
Press the dropdown menus to display a performance dashboard for each optimization method listed below. You'll be able to see each optimal portfolio allocation, expected returns, volatitlity, Sharpe Ratio and Conditional VaR.
""")

st.markdown("<h2 style='font-size:24px;'>Markowitz Portfolio Optimization Method 📊</h2>", unsafe_allow_html=True)
st.markdown("""
This is the classic approach based on Modern Portfolio Theory (MPT), introduced by Harry Markowitz. The goal here is to minimize the overall volatility (or risk) of the portfolio while maintaining a specific level of expected return. By considering the covariance between asset returns, this method helps build a diversified portfolio that maximizes returns for a given risk level. It's perfect for investors seeking a stable and well-diversified investment strategy.
""")

# Generating the Markowitz Optimal Portfolio
ticks_marko, weights_marko, perf_marko, w_arr_marko = portOptimize(capm_rets, cov_matrix, selected_prices, optimization_type='min_volatility')
# Displaing the Dashborad for the Markowtiz Optimal Portfolio
display_portfolio_dashboard(ticks_marko, weights_marko, perf_marko, 'min_volatility', 1)

st.markdown("<h2 style='font-size:24px;'>Maximum Sharpe Ratio Optimization Method 📊</h2>", unsafe_allow_html=True)
st.markdown("""
This method aims to build a portfolio that maximizes the Sharpe Ratio, which is the ratio of excess return (return above the risk-free rate) to the portfolio's risk (standard deviation). A higher Sharpe Ratio indicates better risk-adjusted returns. This method is ideal for those looking to maximize their potential return for every unit of risk taken, ensuring an efficient use of capital.
""")

# Generating the Max Sharpe Ratio Optimal Portfolio
ticks_sharpe, weights_sharpe, perf_sharpe, w_arr_sharpe = portOptimize(capm_rets, cov_matrix, selected_prices, optimization_type='max_sharpe')
# Displaing the Dashborad for the Max Sharpe Ratio Optimal Portfolio
display_portfolio_dashboard(ticks_sharpe, weights_sharpe, perf_sharpe, 'max_sharpe', 2)

st.markdown("<h2 style='font-size:24px;'>Minimum Conditional VaR Optimization Method 📊</h2>", unsafe_allow_html=True)
st.markdown("""
CVaR, also known as Expected Shortfall, is a risk measure that captures the average loss of a portfolio beyond a specified confidence level (e.g., 5%). The minimum CVaR method focuses on minimizing the worst-case average losses, providing a cushion against extreme negative outcomes. This approach is especially suitable for risk-averse investors who want to protect their portfolio from significant downturns.
""")

# Generating the Min CVaR Optimal Portfolio
ticks_cvar, weights_cvar, perf_cvar, w_arr_cvar = portOptimize(capm_rets, cov_matrix, selected_prices, optimization_type='min_cvar')
# Displaing the Dashborad for the Min CVaR Optimal Portfolio
display_portfolio_dashboard(ticks_cvar, weights_cvar, perf_cvar, 'min_cvar', 3)

# Displaying the Discrete Allocation of the selected portfolio
st.subheader("Discrete Allocation 🔃")
st.markdown("<a name='discrete-allocation'></a>", unsafe_allow_html=True)
st.markdown("""
When you create an optimized portfolio, the result often provides weights or proportions for each stock relative to the total investment amount. However, in real life, you can't invest in fractional shares (unless you're using a platform that supports them). **Discrete allocation** translates these proportions into the actual number of whole shares to purchase for each stock while staying as close as possible to the target weights.
            
In this dropdown menu you have a discrete allocation tool where you can choose between the three optimization methods used, then you can write your investment amount and check out how many shares of each stock you'd have to buy and how much money you'd have to invest in each stock.
""")

# Create a collapsible section for discrete allocation
with st.expander("Portfolio Discrete Allocation"):
    # Radio buttons to choose the optimization method
    method = st.radio("Select Optimization Method", ("Min Volatility", "Max Sharpe Ratio", "Min CVaR"))

    # Display appropriate results based on the selected method
    if method == "Min Volatility":
        display_allocation_dashboard(w_arr_marko, selected_prices)
    elif method == "Max Sharpe Ratio":
        display_allocation_dashboard(w_arr_sharpe, selected_prices)
    elif method == "Min CVaR":
        display_allocation_dashboard(w_arr_cvar, selected_prices)

# Conclusion section
st.markdown("<a name='conclusion'></a>", unsafe_allow_html=True)
st.subheader("Conclusion and Further Learning 🎉")
st.markdown("""
Congrats on building your first Investemnt Portfolio, You now have a basic understanding of how to select stocks, model risk and returns, optimize portfolios, and perform discrete allocation.

If you want to deepen your knowledge and explore additional resources, consider the following:

**Online Courses**:
  - [Coursera's "Portfolio and Risk Management"](https://www.coursera.org/learn/portfolio-risk-management)
  - [Udemy's "Portfolio Management & Risk Management"](https://www.udemy.com/course/investment-analysis-portfolio-management/?couponCode=ST20MT111124A)
  - [edX's "Principles of Valuation: Risk and Return"](https://www.edx.org/executive-education/university-of-cape-town-investment-management?index=product&queryID=a1a7a1a48993d6674cd10ed5ed0df1b7&position=1&results_level=first-level-results&term=portfolio+management&objectID=course-a90b2d64-f4e3-4e07-82a8-2af60e8a6548&campaign=Investment+Management&source=2u&product_category=executive-education&placement_url=https%3A%2F%2Fwww.edx.org%2Fsearch)

**Websites and Forums**:
  - [Investopedia](https://www.investopedia.com) for comprehensive definitions and tutorials.
  - [Seeking Alpha](https://seekingalpha.com) for financial analysis and community discussions.
  - [r/investing on Reddit](https://www.reddit.com/r/investing/) for insights and shared experiences from a community of investors.

Continuous learning and staying informed about the latest advancements in finance, investment strategies, and portfolio theory can help you refine your approach and achieve better results. We hope this app has provided you with a strong starting point. Happy investing!
""")


# Sidebar Menu for navigation
st.sidebar.title("Navigation 🧭")
st.sidebar.markdown("[Basic Description](#basic-description)")
st.sidebar.markdown("[Stock Selection](#stock-selection)")
st.sidebar.markdown("[Risk and Returns Models](#risk-and-returns-models)")
st.sidebar.markdown("[Portfolio Optimization](#portfolio-optimization)")
st.sidebar.markdown("[Discrete Allocation](#discrete-allocation)")
st.sidebar.markdown("[Conclusion](#conclusion)")

# Footer section inside the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Portfolio Optimization Tool**  
This app was created for educational purposes.
                    
Data is sourced from [Yahoo Finance](https://finance.yahoo.com/).
                    
Developed by [Ibrajin Zeitun](https://github.com/IbrajinZ).
""")
