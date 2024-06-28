import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose



# Loading the Chicago crimes data from 2005 to 2007
df_1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)
# Loading the Chicago crimes data from 2008 to 2011
df_2 = pd.read_csv('Chicago_crimes_2008_to_2011.csv', error_bad_lines=False)
# Loading the Chicago crimes data from 2012 to 2017
df_3 = pd.read_csv('Chicago_crimes_2012_to_2017.csv', error_bad_lines=False)

# Displaying the shape of the dataframes
print(df_1.shape, df_2.shape, df_3.shape)

# Concatenating all the datasets into one for a complete analysis
chicago_df = pd.concat([df_1, df_2, df_3])
print(chicago_df.shape)

# Checking the first few rows of the dataset
print(chicago_df.head())

# Checking the last few rows of the dataset
print(chicago_df.tail(10))

# Identifying the missing data points using a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(chicago_df.isnull(), cbar=False, cmap='YlGnBu')
plt.show()

# Dropping irrelevant columns
chicago_df.drop(['Unnamed: 0', 'Case Number', 'ID', 'IUCR', 'X Coordinate', 'Y Coordinate', 'Updated On', 
                 'Year', 'FBI Code', 'Beat', 'Ward', 'Community Area', 'Location', 'District', 'Latitude', 
                 'Longitude'], inplace=True, axis=1)

# Converting 'Date' to datetime format
chicago_df['Date'] = pd.to_datetime(chicago_df['Date'], format='%m/%d/%Y %I:%M:%S %p')

# Setting 'Date' as the index
chicago_df.set_index(pd.DatetimeIndex(chicago_df['Date']), inplace=True)

# Count of each type of crime
crime_counts = chicago_df['Primary Type'].value_counts()
print(crime_counts)

# Top 15 types of crimes
top_15_crimes = crime_counts.iloc[:15]
print(top_15_crimes)

# Visualizing the top 15 types of crimes
plt.figure(figsize=(15,10))
sns.countplot(y='Primary Type', data=chicago_df, order=top_15_crimes.index)
plt.title('Top 15 Types of Crimes in Chicago')
plt.show()

# Visualizing the top 15 locations of crimes
top_15_locations = chicago_df['Location Description'].value_counts().iloc[:15]
plt.figure(figsize=(15,10))
sns.countplot(y='Location Description', data=chicago_df, order=top_15_locations.index)
plt.title('Top 15 Locations of Crimes in Chicago')
plt.show()

# Visualizing the number of crimes per year
plt.figure(figsize=(10,5))
plt.plot(chicago_df.resample('Y').size())
plt.title('Crime Count per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
plt.show()

# Visualizing the number of crimes per month
plt.figure(figsize=(10,5))
plt.plot(chicago_df.resample('M').size())
plt.title('Crime Count per Month')
plt.xlabel('Months')
plt.ylabel('Number of Crimes')
plt.show()

# Visualizing the number of crimes per day
plt.figure(figsize=(10,5))
plt.plot(chicago_df.resample('D').size())
plt.title('Crime Count per Day')
plt.xlabel('Days')
plt.ylabel('Number of Crimes')
plt.show()

# Visualizing the number of crimes per quarter
plt.figure(figsize=(10,5))
plt.plot(chicago_df.resample('Q').size())
plt.title('Crime Count per Quarter')
plt.xlabel('Quarters')
plt.ylabel('Number of Crimes')
plt.show()

# Resampling the data on monthly level
chicago_prophet = chicago_df.resample('M').size().reset_index()

# Renaming the columns
chicago_prophet.columns = ['Date', 'Crime Count']

# Renaming columns for Prophet compatibility
chicago_prophet_df_final = chicago_prophet.rename(columns={'Date': 'ds', 'Crime Count': 'y'})
# Instantiating Prophet model
m = Prophet()

# Fitting the model on the data
m.fit(chicago_prophet_df_final)

# Making future predictions for 720 periods (days)
future = m.make_future_dataframe(periods=720)
forecast = m.predict(future)

# Visualizing the forecast
fig1 = m.plot(forecast, xlabel='Date', ylabel='Crime Rate')
plt.show()

# Plotting forecast components
fig2 = m.plot_components(forecast)
plt.show()
# Decomposing the time series
decomposition = seasonal_decompose(chicago_prophet_df_final['y'], model='additive', period=12)
decomposition.plot()
plt.show()

# Identifying anomalies using Prophet
anomalies = forecast[forecast['yhat_upper'] < chicago_prophet_df_final['y']]
plt.figure(figsize=(10,5))
plt.plot(chicago_prophet_df_final['ds'], chicago_prophet_df_final['y'], label='Actual')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
plt.scatter(anomalies['ds'], anomalies['yhat_upper'], color='red', label='Anomaly')
plt.xlabel('Date')
plt.ylabel('Crime Count')
plt.legend()
plt.show()
# Crime trends for the top 5 crime types
top_crimes = chicago_df['Primary Type'].value_counts().index[:5]
plt.figure(figsize=(15,10))
for crime in top_crimes:
    crime_trend = chicago_df[chicago_df['Primary Type'] == crime].resample('M').size()
    plt.plot(crime_trend.index, crime_trend, label=crime)
plt.title('Crime Trends for Top 5 Crime Types')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()
