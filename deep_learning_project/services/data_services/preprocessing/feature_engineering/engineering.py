import pandas as pd


class FeatureEngineer:
    def __init__(self, data):
        """
        Initialize the FeatureEngineer with a dataset.

        Parameters:
        data (dict or DataFrame): The data to be used for feature engineering.
        """
        self.df = pd.DataFrame(data)

    def create_date_feature(self):
        """
        Convert the 'date' column to datetime format.
        """
        self.df['date'] = pd.to_datetime(self.df['date'], format='%d.%m.%Y', dayfirst=True)

    def add_time_features(self):
        """
        Add time-based features such as month, day of the week, day of the year,
        day of the month, week of the year, and season to the DataFrame.
        """
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['day_of_month'] = self.df['date'].dt.day
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['season'] = self.df['date'].dt.month % 12 // 3 + 1

    def add_lag_features(self):
        """
        Add lag features for the 'amount' column, including 1-day, 7-day, and 30-day lags,
        as well as rolling mean features for 7, 14, and 30 days.
        """
        self.df['amount_lag_1'] = self.df['amount'].shift(1)
        self.df['amount_lag_7'] = self.df['amount'].shift(7)
        self.df['amount_lag_30'] = self.df['amount'].shift(30)
        self.df['amount_rolling_mean_7'] = self.df['amount'].rolling(window=7).mean()
        self.df['amount_rolling_mean_14'] = self.df['amount'].rolling(window=14).mean()
        self.df['amount_rolling_mean_30'] = self.df['amount'].rolling(window=30).mean()

    def add_aggregate_features(self):
        """
        Add aggregate features such as total and average 'amount' per day, week, and month.
        """
        self.df['daily_total_amount'] = self.df.groupby('date')['amount'].transform('sum')
        self.df['weekly_total_amount'] = self.df.groupby('week_of_year')['amount'].transform('sum')
        self.df['monthly_total_amount'] = self.df.groupby('month')['amount'].transform('sum')
        self.df['daily_avg_amount'] = self.df.groupby('date')['amount'].transform('mean')
        self.df['weekly_avg_amount'] = self.df.groupby('week_of_year')['amount'].transform('mean')
        self.df['monthly_avg_amount'] = self.df.groupby('month')['amount'].transform('mean')

    def add_change_rate_features(self):
        """
        Add percentage change features for the 'amount' column on a daily, weekly, and monthly basis.
        """
        self.df['daily_amount_pct_change'] = self.df['amount'].pct_change()
        self.df['weekly_amount_pct_change'] = self.df['amount'].pct_change(7)
        self.df['monthly_amount_pct_change'] = self.df['amount'].pct_change(30)

    def add_price_relationship_features(self):
        """
        Add features related to the relationship between 'price' and 'amount', including
        price to amount ratio and lagged price values.
        """
        self.df['price_amount_ratio'] = self.df['price'] / self.df['amount']
        self.df['price_lag_1'] = self.df['price'].shift(1)
        self.df['price_lag_7'] = self.df['price'].shift(7)
        self.df['price_lag_30'] = self.df['price'].shift(30)

    def add_season_country_category_features(self):
        """
        Add aggregate features based on season, country, and main category, such as
        average and total 'amount' per season, country, and main category.
        """
        self.df['season_avg_amount'] = self.df.groupby('season')['amount'].transform('mean')
        self.df['season_total_amount'] = self.df.groupby('season')['amount'].transform('sum')
        self.df['country_avg_amount'] = self.df.groupby('country_id')['amount'].transform('mean')
        self.df['country_total_amount'] = self.df.groupby('country_id')['amount'].transform('sum')
        self.df['main_category_avg_amount'] = self.df.groupby('main_category')['amount'].transform('mean')
        self.df['main_category_total_amount'] = self.df.groupby('main_category')['amount'].transform('sum')
        self.df['season_country_avg_amount'] = self.df.groupby(['season', 'country_id'])['amount'].transform('mean')
        self.df['season_country_total_amount'] = self.df.groupby(['season', 'country_id'])['amount'].transform('sum')
        self.df['season_main_category_avg_amount'] = self.df.groupby(['season', 'main_category'])['amount'].transform(
            'mean')
        self.df['season_main_category_total_amount'] = self.df.groupby(['season', 'main_category'])['amount'].transform(
            'sum')
        self.df['country_main_category_avg_amount'] = self.df.groupby(['country_id', 'main_category'])[
            'amount'].transform('mean')
        self.df['country_main_category_total_amount'] = self.df.groupby(['country_id', 'main_category'])[
            'amount'].transform('sum')

    def round_decimal_values(self):
        """
        Round numerical columns to 2 decimal places.
        """
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].round(2)

    def fill_missing_values(self):
        """
        Fill missing values in the DataFrame with 0.
        """
        categorical_cols = ['shop', 'item', 'item_category_id', 'main_category', 'country_id', 'season', 'day_of_week',
                            'day_of_year', 'day_of_month', 'week_of_year']
        self.df[categorical_cols] = self.df[categorical_cols].astype('float64')
        self.df = self.df.fillna(0)

    def engineer_features(self):
        """
        Perform all feature engineering steps: creating date features, adding time features,
        adding lag features, adding aggregate features, adding change rate features,
        adding price relationship features, adding season, country, and category features,
        filling missing values, and rounding decimal values.

        Returns:
        DataFrame: The DataFrame with engineered features.
        """
        self.create_date_feature()
        self.add_time_features()
        self.add_lag_features()
        self.add_aggregate_features()
        self.add_change_rate_features()
        self.add_price_relationship_features()
        self.add_season_country_category_features()
        self.fill_missing_values()
        self.round_decimal_values()
        return self.df
