# Are International Tuna Prices Impacted by Environmental Variables? <br>
*Hugo Cesar Camou Campoy, Josh Kennedy, Reed Abraham, Roopa Bharadwaj* <br>
Machine Learning for Business Applications 1 <br>
Tepper School of Business, Carnegie Mellon University <br>
May 5, 2021

**The aim of this project is to predict international prices of skipjack tuna given a series of environmental and economic factors.**

## Introduction & Overview

International and local efforts are crucial to guarantee the balance between the sustainability of the catch and the worth of the industry. Tuna is most consumed fish and the second most important fish by wild capture in the world (with 5.2 million metric tons in 2018 ), and the industry around it contributes more that 40 billion dollars to the global economy per year. Even when catch has been increasing year after year, tuna prices have plummeted since 2012 , destroying in the process 1.8 billion dollars in value, not to mention that increased catch threatens the sustainability of the activity. This is aggravated by a lack of international coordination: there is not one single sanctioning body that concentrates efforts on a global context. For example, in the Pacific Ocean, the fastest growth and main producing region of tuna, three different international associations (IATTC , WCPFC and the CCSBT ) establish the norms for the catch, sometimes with overlap in the areas. Even in a regional scale, lack of coordination is evident: in this year, IATTC did not establish international catch quotas for the eastern Pacific, after its members failed to reach consensus . An accurate and unbiased prediction of prices, paired with other environmental and production models can provide the confidence to work on a global context, and the necessary context to determine the optimal regulatory framework.

Price prediction for commodities in general and food supplies in particular is a topic of common interest. Academic research has intensively proposed price and production (catch) prediction models using traditional statistical analysis (e.g. Onour, Ibrahim and Sergi, Bruno, Modeling and forecasting volatility in the global food commodity prices (January 1, 2011)), financial valuation approaches (e.g. Chen, Yu-Chin and Rogoff, Kenneth S. and Rossi, Barbara, Predicting Agri-Commodity Prices: An Asset Pricing Approach (May 10, 2010)), random forests and vector machines (e.g. Dabin Zhang, Shanyin Cheng, Liwen Ling and Qiang Xia, Forecasting Agricultural Commodity Prices Using Model Selection Framework With Time Series Features and Forecast Horizons (February 4, 2020)), and machine learning (e.g. Jabez Harris, A Machine Learning Approach to Forecasting Consumer Food Prices (August 2017)) with different degrees of success. Currently, no method or model is universally accepted as a reliable and standard predictor.

Machine Learning is an adequate tool to develop a pricing model, and can potentially surpass the prediction accuracy of other methods. Traditional statistical analysis relies on the assumption of invariability in time, which does not hold in the tuna industry context. Juvenile depletion caused by excess catch, global warming affectations in the life cycle of tuna, and changes in food consumption preferences can all impact pricing. A machine learning model can deal with these circumstances by continuously getting new information and updating its predictions automatically. In this way, an ML model can remain current for the next prediction horizon.

## Data Collection
#### Starting Point & Data Collection
Our approach began without many assumptions as to whether Tuna Prices can be modeled or predicted. We wanted to begin without any biases as to what factors might influence the price.

As such, we started by generating a list of the broad areas of data we believed might be predictive in our analysis and model. The initial list included a variety of sources & hypotheses:

*Environmental Data: Was the lifecycle of Tuna somehow impacted by changing global conditions?*
 - Water Temperatures & Variances
 - Land Temperatures & Variances
 - Ocean Currents & Wave Patterns
 - Sea Winds
 - Sea Ice Levels
 - Sea Level Pressures
 - Precipitation

*Fishing & Harvesting Data: Certainly there must be a relationship between the levels of fishing & harvesting, but to what extent?*
 - Pounds of Tuna Harvested
 - Pounds of Substitute Fish Harvested
 - Fishing License Statistics
 - Government Fishing Regulations

*Commercial Activity Data: Producers and consumers work hand-in-hand to impact prices, but what drives prices?*

 - Producer Price Indices for Seafood-adjacent Industries
 - Price Indices of Tuna substitutes (Shrimp, Chicken, etc. .)
 - Import & Export Prices Indices
 - Sushi & Seafood Restaurant Performance
 - Consumer Preferences

#### Data Collection Process
In practice, much of the tuna & fishing specific data was found to be proprietary and sparse. Available fishing data was dispersed across each government's networks and there was no global organization to consolidate and distribute the data. 

The richest datasets were related to environmental factors. The National Oceanic and Atmospheric Administration (NOAA) provides robust datasets related to temperatures. The European Commission funds a "Climate Data Store" (Copernicus) that provides a wealth of data.

Federal Reserve Economic Data (FRED) provided a number of datasets related to commercial & economic data.

## Model Data Sources
As a result of the available data, the majority of our model inputs are monthly aggregations of environmental factors. A detailed summary of the various inputs can be found here:

**NOAA:** Monthly land-ocean temperature datasets were compiled via ASCII Time Series Data Access.
> https://www.ncdc.noaa.gov/noaa-merged-land-ocean-global-surface-temperature-analysis-noaaglobaltemp-v5

**Copernicus:** Our broad Monthly Sea Dataset was compiled via the "Climate Data Store" in GRIB format. We then used the Pygrib package to extract, transform, and join to our NOAA dataset. 
> https://cds.climate.copernicus.eu/cdsapp#!/home

**NSIDC:** Our Monthly Sea Ice Dataset was compiled via FTP. 
> https://nsidc.org/data/g02135

**FRED:** A number of features were collected from FRED.
> https://fred.stlouisfed.org/

 - Seafood Product Preparation & Packaging Producer Price Index
 - Fish and Seafood Markets Producer Price Index
 - Global Fish Price Index
 - Global Shrimp Price Index
 - U.S. Fish & Shellfish Import/Export Price Indices

All of these sources were filtered and joined together via custom Python E/T job.

## Linear Forecasting Baseline

### Time Series Forecasting

"A time series is usually modelled through a stochastic process `Y(t)`, i.e. a sequence of random variables. In a forecasting setting we find ourselves at time t and we are interested in estimating `Y(t+h)`, using only information available at time t."[[1]](#1)

The usage of time series models here is twofold:
* Obtain an understanding of the underlying forces and structure that produced the data
* Fit a model and proceed to forecast.

"Time series analysis provides a ton of techniques to better understand a dataset. The most useful of these is the splitting of time series into 4 parts:
1. **Level**: The base value for the series if it were a straight line.
1. **Trend**: The linear increasing or decreasing behavior of the series over time.
1. **Seasonality**: The repeating patterns or cycles of behavior over time.
1. **Noise:** The variability in the observations that cannot be explained by the model."[[2]](#2)

All-time series generally have a level, noise, while trend and seasonality are optional. The main features of many time series are trends and seasonal variation. 

These components combine in some way to provide the observed time series. For example, they may be added together to form a model such as:

`Y = levels + trends + seasonality + noise`

### Automatic Time Series Decomposition

Statsmodel python library provides a function seasonal_compose() to automatically decompose a time series. We used additive model as our quick peek at the time series shows linear trend and seasonality.

![pic1](images/roopa1.png)
**Fig. 1** - *Automatic Time Series Decomposition*

We also generated a model using manual polynomial fitting to find the seasonalities in our data.

![pic1](images/roopa2.png)
**Fig. 2** - *Polynomial fit to find seasonalities*

This shows that seasonalities are present in our data.

### Stationarity

"Stationarity is an important characteristic of time series. A time series is stationary if it has constant mean and variance over time. Most models work only with stationary data as this makes it easier to model."[[7]](#7)

#### Check for Stationarity

We use autocorrelation and partial autocorrelation plots to show how correlated are values at time t with the next values in time `t+1, t+2, ..., t+n`.

"If the data would be non-stationary the autocorrelation values will be highly correlated with distant points in time showing possible seasonalities or trends. Stationary series autocorrelation values will quickly decrease over time t. This shows us that information is carried over time and then the series will not constant over time."[[7]](#7)

We also used augmented Dickey-Fuller test, which is a type of statistical test called a unit root test. "The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend. There are a number of unit root tests and the Augmented Dickey-Fuller may be one of the more widely used. It uses an autoregressive model and optimizes an information criterion across multiple different lag values.

The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary." [[7]](#7)

![pic1](images/roopa3.png)
**Fig. 3** - *Dickey-Fuller test, Auto and Partial Correlation, Rolling mean and standard deviation*


With a p value of 0.23, we apply methods to make our target series stationary like log scale transformation or smoothing.

### Methods for Time Series Forecasting

"There are many methods that we could have used for time series forecasting and there is not a clear winner. Model selection depended on how the data looked. Some models were more robust against outliers but performed worse than the more sensible ones.

When looking at our data the main split was whether we had extra regressors (features) to our time series or just the series. Based on this we started exploring different methods for forecasting and their performance in different metrics."[[7]](#7)

We split our data into test training sets having 85 months of training data and 36 months of testing data.

#### Univariate Time Series Analysis

In this section we will focus on time series forecasting methods capable of only looking at the target variable. This means no other regressors (more variables) can be added into the model. We tried three univariate time series models as follows -

* Auto Regression (AR)
* Autoregressive integrated moving average (ARIMA)
* Seasonal Autoregressive moving average (SARIMA)

"In an Auto Regressive model, the forecasts correspond to a linear combination of past values of the variable. In a Moving Average model, the forecasts correspond to a linear combination of past forecast errors.

Basically, the ARIMA models combine these two approaches. Since they require the time series to be stationary, differencing (Integrating) the time series was a necessary step i.e. considering the time series of the differences instead of the original one.

The SARIMA model (Seasonal ARIMA) extends the ARIMA by adding a linear combination of seasonal past values and/or forecast errors."[[7]](#7) 
The following plots show the predictions on or 36 months test data by using Auto Regression (AR) and SARIMA models.

##### **Auto Regression (AR)**
![pic1](images/roopa4.png)
**Fig. 4** - *Auto Regression model*

##### **Seasonal Autoregressive Integrated Moving-Average (SARIMA)**
![pic1](images/roopa5.png)
**Fig. 5** -*Seasonal Autoregressive Integrated Moving-Average (SARIMA) model*

#### Multivariate Time Series Analysis

In this section we attempted to use methods like Random Forest and XGBoost for time series forecasting. 
##### **Random Forest (RF)**

"Random forest is an ensemble of decision tree algorithms. A number of decision trees are created where each tree is created from a different sample. It can be used for both classification and regression. In our case the final prediction is the average prediction across the decision trees (we used 5)."[[4]](#4)

![pic1](images/roopa6.png)
**Fig. 6** - *Random Forest model*

##### **XGBoost**

"XGBoost (Extreme Gradient Boost) provides a high-performance implementation of gradient boosted decision trees. Rather than training all of the models in isolation of one another like random forest, XG Boost trains models in succession"[[4]](#4)

![pic1](images/roopa7.png)
**Fig. 7** *XGBoost model*
### Evaluation Metrics

There are many measures that can be used to analyze the performance of our prediction so we will be using the top 4 most used metrics for time series forecasting. 

* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)
* Root Mean Squared Error (RMSE)
* R2 Coefficient of Determination (r2)

![pic1](images/roopa8.png)
**Fig. 8** - *Result metrics*

"For any data, that a Random Forest/XGBoost has not seen before, at best, it can predict an average of training values that it has seen before. If the Validation set consists of data points that are greater or less than the training data points, a Random Forest will provide us with Average results as it is not able to Extrapolate and understand the growing/decreasing trend in our data. 

Therefore, a Random Forest model does not scale very well for time-series data and might need to be constantly updated in Production or trained with some Random data that lies outside our range of Training set."[[5]](#5)

Answering questions like “What would the price of SkipJack Tuna be for next Year?” becomes really difficult when using Random Forests.
### Conclusions

Fitting a Linear Model or a Neural Net, in this case, might be sufficient to predict data which has increasing or decreasing trends. 
## Model Selection and ML Thought Process
@Reed
#### Data Characteristics / Input Generation
`Placeholder text`
#### Time Step Asynchronicity
`Placeholder text`
#### High Colinearity
`Placeholder text`
#### GeoSpatial & Time Series Combination
`Placeholder text`
#### Feature Count vs. Sample Size
`Placeholder text `

## ML Technologies of Interest
#### Modern RNN Implementations
`Placeholder text`
#### Multi Modal Deep Learning
`Placeholder text`

## Model Build, Test, and Analysis (rename header?)
#### Feature Selection
Lasso @Josh
#### Baseline Results - MLP
@Reed @Hugo
 - Show unsatisfactory
 - With + without normalization
 - Too many parameters, not enough depth
 - does not adequately capture periodicity

#### Classification Transform
@Hugo
 - Failure (Hugo comments?)

#### LeNet Adaptation
@Hugo
 - Actually not Lenet just did a really good job of pre-processing the data with PCA
 - 7 harmonics of various parameters included in PCA so 'pseudo-history' is included in model

#### LSTM
@Josh
 - Standard implementation of LSTM attempting to forward predict last time sample
 - Works exceptionally well

#### LSTM, Rolling Window
@Reed
 - Attempt to limit LSTM to predicting only using prior 12 months of data
 - Limit possibility that network is being trained on the macro trends
 - Failure - implementation too complex, unable to validate true behavior or implement well enough to attempt meta-parameter opimization

#### Temporal Fusion Transformer (TFT)
@Reed
 - Link to paper
 - Recent architecture featured in "PyTorch Forecasting"
 - Combines attributes of RNN, LSTM, and CNN - capable of combining time series data as well as categoricals (say: months)
 - Promising initial results but implementation in PyTorch Forecasting was questionable and detailed / vetted implementation was beyond the scope of the project

#### Future Work
 - speculation on what we could do with infinite time horizon

## Model Results
@All after model sections are written
 - Expectations
 - Results from model(s)
 - Which model is best?

## Summary
 - Overall broad summary (how does this relate back to the problem space?
 - Broad findings
 - How can this be used in the real world?

## References and Citations
#### 3rd Party Python Packages & Functions
> [[7]](#7) Jaime Ferrando Huertas, https://github.com/jiwidi/time-series-forecasting-with-python

#### Dataset Citations
> [[1]](#1) Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2019): ERA5 monthly averaged data on single levels from 1979 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). (Accessed on [01-MAY-2021]), https://10.24381/cds.f17050d7

> [[1]](#1) Zhang, H.-M., B. Huang, J. Lawrimore, M. Menne, Thomas M. Smith, NOAA Global Surface Temperature Dataset (NOAAGlobalTemp), Version 4.0. NOAA National Centers for Environmental Information. doi: https://10.7289/V5FN144H [01-MAY-2021].

> [[1]](#1) Fetterer, F., K. Knowles, W. N. Meier, M. Savoie, and A. K. Windnagel. 2017, updated daily. Sea Ice Index, Version 3. Boulder, Colorado USA. NSIDC: National Snow and Ice Data Center. doi: https://doi.org/10.7265/N5K072F8. [01-MAY-2021].

> [[1]](#1) International Monetary Fund, Global price of Fish [PSALMUSDM], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PSALMUSDM, May 2, 2021.

> [[1]](#1) U.S. Bureau of Labor Statistics, Producer Price Index by Industry: Seafood Product Preparation and Packaging: Fresh and Frozen Seafood Processing [PCU3117103117102], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PCU3117103117102, May 2, 2021.

> [[1]](#1) U.S. Bureau of Labor Statistics, Producer Price Index by Industry: Specialty Food Stores: Fish and Seafood Markets [PCU445200445200102], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PCU445200445200102, May 2, 2021.

> [[1]](#1) U.S. Bureau of Labor Statistics, Tuna, Light, Chunk, Per Lb. (453.6 Gm) in U.S. City Average [APU0000707111], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/APU0000707111, May 2, 2021.

> [[1]](#1) U.S. Bureau of Labor Statistics, Import Price Index (End Use): Fish and Shellfish [IR01000], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IR01000, May 2, 2021.

> [[1]](#1) U.S. Bureau of Labor Statistics, Export Price Index (End Use): Fish and Shellfish [IQ01000], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IQ01000, May 2, 2021.

> [[1]](#1) International Monetary Fund, Global price of Shrimp [PSHRIUSDM], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PSHRIUSDM, May 2, 2021.

> [[1]](#1)Davide Bruba. "An overview of time series forecasting models", 
https://towardsdatascience.com/an-overview-of-time-series-forecasting-models-a2fa7a358fcb

> [[2]](#2)Athul Anish. “Time Series Analysis”,
https://medium.com/swlh/time-series-analysis-7006ea1c3326

> [[3]](#3)Statworx Blog, “Time series forecasting with random forest”,
https://medium.com/@statworx_blog/time-series-forecasting-part-i-e30a16bac58a

> [[4]](#4)Indraneel Dutta Baruah, Analytics Vidya. “Combining Time Series Analysis with Artificial Intelligence: the future of forecasting”,
https://medium.com/analytics-vidhya/combining-time-series-analysis-with-artificial-intelligence-the-future-of-forecasting-5196f57db913

> [[5]](#5)Aman Arora, “Why Random Forests can’t predict trends and how to overcome this problem?”,
https://medium.datadriveninvestor.com/why-wont-time-series-data-and-random-forests-work-very-well-together-3c9f7b271631

> [[6]](#6)Jason Browniee, “How to Decompose Time Series Data into Trend and Seasonality”
https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/




## Source Code
We have published all of our source code to a public Github repo:
[CMU MSBA Team 4 Code Repository](https://github.com/josh-kennedy-7/cmu_msba_team_4_2022)
