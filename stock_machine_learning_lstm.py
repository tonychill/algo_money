import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
from lumibot.backtesting import PolygonDataBacktesting
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable

# from credentials import ALPACA_CONFIG, POLYGON_CONFIG

# ALPACA Configuration
ALPACA_CONFIG = {
    "API_KEY": os.getenv("ALPACA_API_KEY", "default_api_key"),  # Provide default value or handle None
    "API_SECRET": os.getenv("ALPACA_API_SECRET", "default_api_secret"),
    "PAPER": os.getenv("ALPACA_IS_PAPER", True),
}
# POLYGON Configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # This should work, but had to set it manually in the terminal or in the zshrc file
print(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))
has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "cpu" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"

# LSTM model definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # This should give you a shape of (batch_size, output_size)
        return out
    
# Training function
def train_model(model, optimiser, loss_fn, X_train, y_train, n_epochs):
    y_train = y_train.view(-1, 1)
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        optimiser.zero_grad()
        outputs = model(X_train)
        if outputs.shape != y_train.shape:
            outputs = outputs.view(*y_train.shape)

        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimiser.step()
        end_time = time.time()

        iteration_time = end_time - start_time
        iterations_per_second = 1 / iteration_time

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{n_epochs}, Loss: {loss.item():.6f}, Iterations/s: {iterations_per_second:.2f}')
            
def train_from_df(df, compute_frequency, n_epochs, learning_rate, model):
    # Get the close price in the future
    df["close_future"] = df["close"].shift(-compute_frequency)
    
    # Get the last row of data to use for prediction
    last_row = df.copy().iloc[[-1]]
    
    # Drop the last row of data as it will be used for prediction
    data_train = df.dropna()
    
    # Standardize and normalize data
    X_train_scaled = ss.fit_transform(data_train.drop(columns=['close_future']))
    y_train_scaled = mm.fit_transform(data_train[['close_future']])

    # Convert to tensors
    X_train_tensors = Variable(torch.Tensor(X_train_scaled)).to(device)
    y_train_tensors = Variable(torch.Tensor(y_train_scaled)).to(device)

    # Reshape for LSTM input
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    y_train_tensors_final = torch.reshape(y_train_tensors, (y_train_tensors.shape[0], 1, y_train_tensors.shape[1]))

    # Assuming this is inside your training loop right before you call train_model
    # print(f"Shape of training data (features): {X_train_tensors_final.shape}")
    # print(f"Shape of training data (target): {y_train_tensors_final.shape}")
    # print(f"Model expected input size: {model.lstm.input_size}")
    # print(f"Model expected hidden size: {model.hidden_size}")
    # print(f"Model expected number of layers: {model.num_layers}")
    # print(f"Model expected output size: {model.fc.out_features}")
    
    # Train the model
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, optimiser, loss_fn, X_train_tensors_final, y_train_tensors_final, n_epochs)
    
    return last_row
            

# Standardize the training data and normalize the target variable
ss = StandardScaler()
mm = MinMaxScaler()
loss_fn = nn.MSELoss()
    

class StockMachineLearningLstm(Strategy):

    # trading_info = pd.DataFrame(columns=['Date of Prediction', 'Expected Price Change', 'Expected Price Change Pct', 'Expected Change Threshold Up', 'Current Price', 'Limit', 'Stop Loss', 'Order'])
    parameters = {
        "asset": Asset(symbol="NVDA", asset_type="stock"),
        "compute_frequency": 15,  # The time (in minutes) that we should retrain our model and make a prediction
        # "compute_frequency": 1440,  # 1440 minutes = 1 day
        #2000 initial lookback period = 2000 * 15 = 30,000 minutes = 20.833 days
        "initial_lookback_period": 4000,  # Increasing this will improve accuracy but will take longer to train
        "initial_epochs": 100,  # The number of epochs to train the model for initially
        "iteration_epochs": 100,  # The number of epochs to train the model for on each iteration
        "learning_rate_initial": 0.001,  # The learning rate for the model initially
        "learning_rate_iteration": 0.003,  # The learning rate for the model on each iteration
        "pct_portfolio_per_trade": 0.45,  # What percentage of the portfolio to trade in each trade. Eg. If the portfolio is worth $100k and this is 0.5, then each trade will be worth $50k
        # "price_change_threshold_up": 0.05,  # The difference between predicted price and the current price that will trigger a buy order (in percentage change).
        # "price_change_threshold_down": -0.08,  # The difference between predicted price and the current price that will trigger a sell order (in percentage change).
        "price_change_threshold_up": 0.003,  # The difference between predicted price and the current price that will trigger a buy order (in percentage change).
        "price_change_threshold_down": -0.003,  # The difference between predicted price and the current price that will trigger a sell order (in percentage change).
        "max_pct_portfolio_long": 1,  # The maximum that the strategy will buy as a percentage of the portfolio (eg. if this is 0.8 - or 80% - and our portfolio is worth $100k, then we will stop buying when we own $80k worth of the symbol)
        "max_pct_portfolio_short": 0.3,  # The maximum that the strategy will sell as a percentage of the portfolio (eg. if this is 0.8 - or 80% - and our portfolio is worth $100k, then we will stop selling when we own $80k worth of the symbol)
        "take_profit_factor": 1,  # Where you place your limit order based on the prediction, eg. if the prediction is 1.05 and this is 1.1, then the limit order will be placed at 1.05 * 1.1 = 1.155
        "stop_loss_factor": 0.5,  # Where you place your stop order based on the prediction, eg. if the prediction is 1.05 and this is 0.9, then the stop order will be placed at 1.05 * 0.9 = 0.945
        "cache_directory": "default_directory"  # Default value or dynamically assigned
    }
    trading_info = pd.DataFrame(columns=['Date of Prediction', 'Expected Price Change', 'Expected Price Change Pct', 'Expected Change Threshold Up', 'Current Price', 'Limit', 'Stop Loss', 'Order'])    

    def save_trading_info(self):
        # Construct the file path by combining the cache directory with the customized filename
        file_name = f'trading_info_{self.asset_symbol}_freq{self.compute_freq}.csv'
        file_path = os.path.join(self.cache_directory, file_name)

        # Save the DataFrame to the specified file path
        self.trading_info.to_csv(file_path, index=True)

    def initialize(self):
        # Extract asset symbol and compute frequency from parameters
        self.asset_symbol = self.parameters['asset'].symbol
        self.asset = self.parameters['asset']  # Initialize the asset attribute
        self.compute_freq = self.parameters['compute_frequency']
        self.learning_rate_iteration = self.parameters['learning_rate_iteration']

        # Dynamically create the cache_directory
        self.cache_directory = f"ml_cache_lstm_{self.asset_symbol}_freq{self.compute_freq}"

        # Ensure the cache directory exists
        if not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory)

        # Get parameters 
        asset = self.parameters["asset"]
        initial_lookback_period = self.parameters["initial_lookback_period"]
        initial_epochs = self.parameters["initial_epochs"]
        learning_rate_initial = self.parameters["learning_rate_initial"]
        
        # Set the initial variables or constants
        compute_frequency = self.parameters["compute_frequency"]

        # Built in Variables
        self.sleeptime = f"{compute_frequency}M"

        # Variable initial states
        self.last_compute = None
        self.cache_df = None
        
        # TODO: This should be saved to a model or something
        
        # Get the historical prices
        df = self.get_data(
            asset, self.quote_asset, compute_frequency * initial_lookback_period 
        )
        # df = self.get_data(
        #     asset, self.quote_asset, initial_lookback_period 
        # )
        
        # Initialize the LSTM model
        hidden_size = 50
        num_layers = 1
        output_size = 1
        num_features = len(df.columns)
        
        # Initialize the LSTM model
        self.model = LSTM(num_features, hidden_size, num_layers, output_size).to(device)
        
        # Train the model
        train_from_df(df, compute_frequency, initial_epochs, learning_rate_initial, self.model)

    def on_trading_iteration(self):
        # Get parameters for this iteration
        asset = self.parameters["asset"]
        compute_frequency = self.parameters["compute_frequency"]
        pct_portfolio_per_trade = self.parameters["pct_portfolio_per_trade"]
        price_change_threshold_up = self.parameters["price_change_threshold_up"]
        price_change_threshold_down = self.parameters["price_change_threshold_down"]
        max_pct_portfolio_long = self.parameters["max_pct_portfolio_long"]
        max_pct_portfolio_short = self.parameters["max_pct_portfolio_short"]
        take_profit_factor = self.parameters["take_profit_factor"]
        stop_loss_factor = self.parameters["stop_loss_factor"]
        cache_directory = self.parameters["cache_directory"]
        iteration_epochs = self.parameters["iteration_epochs"]
        learning_rate_iteration = self.parameters["learning_rate_iteration"]

        # Get the current time
        dt = self.get_datetime()

        # Get the historical prices
        price_df = self.get_data(
            # TODO: This does not need to get so much data because it should be trained already at init (remove lookpack_period)
            asset, self.quote_asset, compute_frequency + 1
        )

        # The current price of the asset
        last_price = self.get_last_price(asset)

        # Get how much we currently own of the asset, do we have a position?
        position = self.get_position(asset)
        # Print the position
        print(f"Position: {position}")
        if position is not None:
            shares_owned = float(position.quantity)
            asset_value = shares_owned * last_price
        else:
            shares_owned = 0
            asset_value = 0
        # Print the asset value and shares owned
        print(f"Asset Value: {asset_value}")
        print(f"Shares Owned: {shares_owned}")
        # Reset the prediction
        prediction = None
        cache_filepath = f"{self.cache_directory}/{self.compute_freq}_lrit{self.learning_rate_iteration}_{self.asset.symbol}.csv"
        
        # Set up caching if we are backtesting
        if self.is_backtesting:
            # Check if file exists, if not then create it
            if os.path.isfile(cache_filepath):
                self.log_message("Cache file exists")
                if self.cache_df is None:
                    self.cache_df = pd.read_csv(cache_filepath)
                    self.cache_df["datetime"] = pd.to_datetime(
                        self.cache_df["datetime"]
                    )
                    self.cache_df = self.cache_df.set_index("datetime")

                # Get the prediction from the cache
                # current_prediction = self.cache_df.loc[dt]
                current_prediction = self.cache_df.loc[self.cache_df.index == dt]

                if current_prediction is not None and len(current_prediction) == 1:
                    prediction = current_prediction["prediction"][0]
            else:
                if not os.path.exists(cache_directory):
                    os.mkdir(cache_directory)
                self.cache_df = pd.DataFrame(columns=["prediction"])
                self.cache_df.index.name = "datetime"
        
        # If we don't have a prediction then we need to compute it using the ML model
        if prediction is None:
            last_row = train_from_df(price_df, compute_frequency, iteration_epochs, learning_rate_iteration, self.model)
            
            # Predict on the next window of test data
            X_test_scaled = ss.transform(last_row.drop(columns=['close_future']))
            X_test_tensors = Variable(torch.Tensor(X_test_scaled)).to(device)
            X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))
            
            self.model.eval()
            with torch.no_grad():
                test_prediction = self.model(X_test_tensors_final)
                test_prediction = test_prediction.cpu().data.numpy()
                test_prediction = mm.inverse_transform(test_prediction)
                prediction = test_prediction[0][0]

            # Add the prediction to the cache
            df = pd.DataFrame([prediction], columns=["prediction"], index=[dt])
            df.index.name = "datetime"
            self.cache_df = pd.concat([self.cache_df, df])
            self.cache_df.sort_index(inplace=True)
            
            # Get the most rexcent close price
            close_price = price_df.iloc[-1]["close"]
            
            # Set the "close" column in self.cache_df to the most recent close price
            self.cache_df.loc[self.cache_df.index == dt, "close"] = close_price
            
            self.cache_df.to_csv(cache_filepath)

        # Calculate the percentage change that the model predicts
        expected_price_change = prediction - last_price
        expected_price_change_pct = expected_price_change / last_price
        
        # Print current directory
        print(os.getcwd())
        
        # Code to calculate or retrieve these values...
        always_recorded_row = {
            'Date of Prediction': dt,
            'Portfolio Value': self.portfolio_value,
            'Expected Price Change': expected_price_change,
            'Expected Price Change Pct': expected_price_change_pct,
            'Expected Change Threshold Up': price_change_threshold_up,
            'Current Price': last_price,
            'Limit': np.nan,  # Placeholder for limit value
            'Stop Loss': np.nan,  # Placeholder for stop loss value
            'Order': None,  # Placeholder for order
            'Cash': self.cash
        }
        # trading_info = trading_info.concatenate(always_recorded_row, ignore_index=True)
        # trading_info = trading_info.concat([pd.always_recorded_row], ignore_index=True)
        # self.trading_info = pd.concat([self.trading_info, pd.DataFrame([always_recorded_row])], ignore_index=True)


        # Our machine learning model is predicting that the asset will increase in value
        if expected_price_change_pct > price_change_threshold_up:
            #print expected price change
            print("\n")
            #Date of prediction
            print(f"Date of Prediction: {dt}")
            print(f"Expected Price Change: {expected_price_change}")
            #print expected price change pct
            print(f"Expected Price Change Pct: {expected_price_change_pct}")
            #print expected change threshold up
            print(f"Expected Change Threshold Up: {price_change_threshold_up}")
            max_allocation = self.parameters["max_pct_portfolio_long"] * self.portfolio_value
            available_allocation = max_allocation - asset_value
            value_to_trade = min(self.portfolio_value * pct_portfolio_per_trade, available_allocation)
            quantity = int(value_to_trade / last_price)


            # Define maximum allocation for a single asset
            max_allocation = self.portfolio_value * self.parameters["max_pct_portfolio_long"]

            # Only execute a buy order if we have not reached our maximum allocation
            # Print portfolio value and max allocation and asset value
            #print available allocation
            print(f"Available Allocation: {available_allocation}")
            print(f"Portfolio Value: {self.portfolio_value}")
            print(f"Max Allocation: {max_allocation}")
            print(f"Asset Value: {asset_value}")
            print(f"Quantity: {quantity}")
            print(f"Value to Trade: {value_to_trade}")
            if asset_value < available_allocation:
                # Market order
                main_order = self.create_order(
                    asset, quantity, "buy", 
                )
                self.submit_order(main_order)
                #print buy order
                print(f"Buy Order: {main_order}")
                # OCO order
                expected_price_move = abs(
                    last_price * expected_price_change_pct
                )
                limit = last_price + (expected_price_move * take_profit_factor)
                # stop_loss = last_price - (expected_price_move * stop_loss_factor)
                # stop_loss = last_price * 0.95
                stop_loss = .05
                #print current price
                #print next line character
                print("\n")
                print(f"Current Price: {last_price}")
                print(f"Limit: {limit}, Stop Loss: {stop_loss}")
                # order = self.create_order(
                #     asset,
                #     quantity,
                #     "sell",
                #     take_profit_price=limit,
                #     stop_loss_price=stop_loss,
                #     position_filled=True,
                #     type="oco",
                # )
                # self.submit_order(order)
                # Update row_data with specific values
                always_recorded_row['Limit'] = limit
                always_recorded_row['Stop Loss'] = stop_loss
                always_recorded_row['Order'] = str(main_order)# + str(order)
            

        # Our machine learning model is predicting that the asset will decrease in value
        elif expected_price_change_pct < price_change_threshold_down:
            max_position_size = max_pct_portfolio_short * self.portfolio_value
            value_to_trade = self.portfolio_value * pct_portfolio_per_trade
            quantity = int(value_to_trade / last_price)

            # Print portfolio value and max allocation and asset value
            print(f"Portfolio Value: {self.portfolio_value}")
            print(f"Asset Value: {asset_value}")
            # print value to trade and max position size
            print(f"Value to Trade: {value_to_trade}")
            print(f"Max Position Size: {max_position_size}")
            # Check that we are not selling too much of the asset
            if (asset_value - value_to_trade) > -max_position_size:
                # Market order
                main_order = self.create_order(
                    asset, quantity, "sell",
                )
                self.submit_order(main_order)
                #print sell order
                print(f"Sell Order: {main_order}")

                # OCO order
                expected_price_move = abs(
                    last_price * expected_price_change_pct
                )
                limit = last_price - (expected_price_move * take_profit_factor)
                stop_loss = last_price + (expected_price_move * stop_loss_factor)
                #print current price
                print("\n")
                print(f"Current Price: {last_price}")
                print(f"Limit: {limit}, Stop Loss: {stop_loss}")
                # order = self.create_order(
                #     asset,
                #     quantity,
                #     "buy",
                #     take_profit_price=limit,
                #     stop_loss_price=stop_loss,
                #     position_filled=True,
                #     quote=self.quote_asset,
                #     type="oco",
                # )
                # self.submit_order(order)
                # Update row_data with specific values
                always_recorded_row['Limit'] = limit
                always_recorded_row['Stop Loss'] = stop_loss
                always_recorded_row['Order'] = str(main_order) #+ str(order)
                #update recording row with cash
                #always_recorded_row['Cash'] = self.cash
        self.trading_info = pd.concat([self.trading_info, pd.DataFrame([always_recorded_row])], ignore_index=False)
        #if you have reached the end date then save the trading info
        print(dt)
        print(f"Cash: {self.cash}")
        self.save_trading_info()
            
    def get_data(self, asset, quote_asset, window_size):
        """Gets pricing data from our data source, then calculates a bunch of technical indicators

        Args:
            asset (Asset): The asset that we want the data for
            window_size (int): The amount of data points that we want to get from our data source (in minutes)

        Returns:
            pandas.DataFrame: A DataFrame with the asset's prices and technical indicators
        """
        data_length = window_size + 40

        bars = self.get_historical_prices(
            asset, data_length, "minute", quote=quote_asset
        )
        data = bars.df

        times = data.index.to_series()
        current_datetime = self.get_datetime()
        data["timeofday"] = (times.dt.hour * 60) + times.dt.minute
        data["timeofdaysq"] = ((times.dt.hour * 60) + times.dt.minute) ** 2
        data["unixtime"] = data.index.astype(np.int64) // 10**9
        data["unixtimesq"] = data.index.astype(np.int64) // 10**8
        data["time_from_now"] = current_datetime.timestamp() - data["unixtime"]
        data["time_from_now_sq"] = data["time_from_now"] ** 2

        data["delta"] = np.append(
            None,
            (np.array(data["close"])[1:] - np.array(data["close"])[:-1])
            / np.array(data["close"])[:-1],
        )
        data["rsi"] = ta.momentum.rsi(data["close"])
        data["ema"] = ta.trend.ema_indicator(data["close"])
        data["cmf"] = ta.volume.chaikin_money_flow(
            data["high"], data["low"], data["close"], data["volume"]
        )
        data["vwap"] = ta.volume.volume_weighted_average_price(
            data["high"], data["low"], data["close"], data["volume"]
        )
        data["bollinger_high"] = ta.volatility.bollinger_hband(data["close"])
        data["bollinger_low"] = ta.volatility.bollinger_lband(data["close"])
        data["macd"] = ta.trend.macd(data["close"])
        # data["adx"] = ta.trend.adx(data["high"], data["low"], data["close"])
        ichimoku = ta.trend.IchimokuIndicator(data["high"], data["low"])
        data["ichimoku_a"] = ichimoku.ichimoku_a()
        data["ichimoku_b"] = ichimoku.ichimoku_b()
        data["ichimoku_base"] = ichimoku.ichimoku_base_line()
        data["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
        data["stoch"] = ta.momentum.stoch(data["high"], data["low"], data["close"])

        # This was causing the problem. It was adding NaN values to the dataframe
        # data["kama"] = ta.momentum.kama(data["close"])
        
        # If we have a column called "otc" then we need to remove it
        if "otc" in data.columns:
            data = data.drop(columns=["otc"])

        data = data.dropna()

        data = data.iloc[-window_size:]

        return data


if __name__ == "__main__":
    is_live = False

    if is_live:
        ####
        # Run the strategy
        ####

        broker = Alpaca(ALPACA_CONFIG)

        strategy = StockMachineLearningLstm(
            broker=broker,
        )

        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        ####
        # Backtest
        ####

        backtesting_start = datetime(2020, 1, 1)
        backtesting_end = datetime(2023, 12, 15)

        ####
        # Get and Organize Data
        ####

        strategy = StockMachineLearningLstm.backtest(
            PolygonDataBacktesting,
            # YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            benchmark_asset="NVDA",
            polygon_api_key=POLYGON_API_KEY,
            polygon_has_paid_subscription=True,
        )
        



