# Are International Tuna Prices Impacted by Environmental Variables? <br>

*Hugo Cesar Camou Campoy, Josh Kennedy, Reed Abraham, Roopa Bharadwaj* <br>
Machine Learning for Business Applications 1 <br>
Tepper School of Business, Carnegie Mellon University <br>
May 5, 2021

**The aim of this project is to predict international prices of skipjack tuna given a series of environmental and economic factors.**

## **TOP TO DO LIST**

1. Citations pass
   1. Do citations have unique numbers
   2. Do the citations link to the appropriate reference
2. Figures pass
   1. Are all figures labeled and captioned
   2. Do figure numbers ascend properly
3. Tense pass
   1. Is everything uniformly 3rd person?
   2. Numbers: do we write out single digits "one" or use numerics? "1"
4. Conclusion draft
5. Pass for redundancy to check whether we are redundant in anything
6. Feature count pass
   1. we have a bunch of places that call out different features
   2. Do we need a section in the data overview that talks about different data sets?
      1. Raw
      2. flat / standardized
      3. Lasso fit
      4. PCA
      5. flat / standardized with harmonics
      6. PCA with harmonics?

## Introduction & Overview

International and local efforts are crucial to guarantee the balance between the sustainability of the catch and the worth of the industry. Tuna is most consumed fish and the second most important fish by wild capture in the world (with 5.2 million metric tons in 2018 ), and the industry around it contributes more that 40 billion dollars to the global economy per year. Even when the catch has been increasing year after year, tuna prices have plummeted since 2012 , destroying in the process 1.8 billion dollars in value, not to mention that the increased catch threatens the sustainability of the activity.

This is aggravated by a lack of international coordination: there is not one single sanctioning body that concentrates efforts on a global context. For example, in the Pacific Ocean, the fastest growth and main producing region of tuna, three different international associations (IATTC[[1]](#1), WCPFC[[2]](#2) and the CCSBT[[3]](#3)) establish the norms for the catch, sometimes with overlap in the areas. Even in a regional scale, lack of coordination is evident: in this year, the IATTC did not establish international catch quotas for the eastern Pacific, after its members failed to reach consensus . An accurate and unbiased prediction of prices, paired with other environmental and production models can provide the confidence to work on a global scale, and the necessary context to determine the optimal regulatory framework.

The Inter-American-Tropical-Tuna-Commission (IATTC), Western & Central Pacific Fisheries Commission (WCPFC), and the Commission for the Conservation of Southern Bluefin Tuna (CCSBT) establish the norms for the catch, sometimes with overlap in the areas. Even in a regional scale, lack of coordination is evident: in this year, the IATTC did not establish international catch quotas for the eastern Pacific, after its members failed to reach consensus . An accurate and unbiased prediction of prices, paired with other environmental and production models can provide the confidence to work on a global context, and the necessary context to determine the optimal regulatory framework.

Price prediction for commodities in general and food supplies in particular is a topic of common interest. Academic research has intensively proposed price and production (catch) prediction models using traditional statistical analysis [[4]](#4), financial valuation approaches [[5]](#5), random forests and vector machines [[6]](#6), and machine learning [[7]](#7) with different degrees of success. Currently, no method or model is universally accepted as a reliable and standard predictor.


Machine Learning is an adequate tool to develop a pricing model, and can potentially surpass the prediction accuracy of other methods. Traditional statistical analysis relies on the assumption of invariability in time, which does not hold in the tuna industry context. Juvenile depletion caused by excess catches, global warming effects on the life cycle of tuna, and changes in food consumption preferences can all impact pricing. A machine learning model can deal with these circumstances by continuously getting new information and updating its predictions automatically. In this way, an ML model can remain current for the next prediction horizon.

## Data Collection

### Starting Point & Data Collection

The data collection process was designed for agnosticism to biases towards assumed factors of influence to the price. The process generated a list of the candidate areas of data with predictive capabilities. The initial list included a variety of sources & hypotheses:

#### *Environmental Data: Was the lifecycle of Tuna somehow impacted by changing global conditions?*

- Water Temperatures & Variances
- Land Temperatures & Variances
- Ocean Currents & Wave Patterns
- Sea Winds
- Sea Ice Levels
- Sea Level Pressures
- Precipitation

#### *Fishing & Harvesting Data: Certainly there must be a relationship between the levels of fishing & harvesting, but to what extent?*

- Pounds of Tuna Harvested
- Pounds of Substitute Fish Harvested
- Fishing License Statistics
- Government Fishing Regulations

#### *Commercial Activity Data: Producers and consumers work hand-in-hand to impact prices, but what drives prices?*

- Producer Price Indices for Seafood-adjacent Industries
- Price Indices of Tuna substitutes (Shrimp, Chicken, etc. .)
- Import & Export Prices Indices
- Sushi & Seafood Restaurant Performance
- Consumer Preferences

### Data Collection Process

In practice, much of the tuna & fishing specific data was found to be proprietary and sparse. Available fishing data was dispersed across each government's networks and there was no global organization to consolidate and distribute the data.

The richest datasets were related to environmental factors. The National Oceanic and Atmospheric Administration (NOAA) provides robust datasets related to temperatures. The European Commission funds a "Climate Data Store" (Copernicus) that provides a wealth of data.

Federal Reserve Economic Data (FRED) provided a number of datasets related to commercial & economic data.

## Model Data Sources

As a result of the available data, the majority of our model inputs are monthly aggregations of environmental factors. A detailed summary of the various inputs can be found here:

**Copernicus:** Our broad Monthly Sea Dataset was compiled via the "Climate Data Store" in GRIB format. We then used the Pygrib package to extract, transform, and join to our NOAA dataset. [[5]](#5)

**NOAA:** Monthly land-ocean temperature datasets were compiled via ASCII Time Series Data Access. [[6]](#6)

**NSIDC:** Our Monthly Sea Ice Dataset was compiled via FTP. [[7]](#7)

**FRED:** A number of features were collected from FRED. [[8]](#8)

- Seafood Product Preparation & Packaging Producer Price Index [[9]](#9)
- Fish and Seafood Markets Producer Price Index [[10]](#10)
- Global Fish Price Index [[11]](#11)
- Global Shrimp Price Index [[12]](#12)
- U.S. Fish & Shellfish Import/Export Price Indices [[13]](#13)

All of these sources were filtered and joined together via a [custom Python data cleaning script.](https://github.com/josh-kennedy-7/cmu_msba_team_4_2022/blob/main/data/dataset_clean_generate_script.py)

**#TODO Debate merits of swapped data section up here**
## Data Characteristics and Input Considerations

@Reed

### Data Characteristics
Summary statistics & context for Skipjack Tuna can be found below:

<img src="images/tuna_statistics.png" alt="drawing" width="400" style="float:left"><br>

<img src="images/tuna_price_over_time.png" alt="drawing" width="800" style="float:left"><br>

#### Non-Uniform Data length

Climate data generally had a longer total history than market data. Market data itself had an inconsistent total history due to the diversity of sources.

To simplify model implementation the basic data set was truncated at the minimum history length. Techniques such as padding or multi modality were not investigated due to time constaints, but would be a logical next step.

#### High Colinearity

Heterogenous data sources and a "more-is-better" collection approach yielded an initial dataset with high colinearity.

<img src="images/colinearity_example.png" alt="drawing" width="400"/><br>
**Fig. n** - *Subset of data graphically representing colinarity.*

All climate data included multiple statistics for each time step. Economic data included common metrics such as maximums, minimums, and variances within the reporting period. While useful for human analysis it is unlikely many of these fields contributed meaningfully to our models. This was quantified through variable selection methods and dimensionality reduction attempts.

#### Feature Count vs. Sample Size

Clipping the data at the minimum available length yielded 121 months of data versus 430 covariates targeting a single output variable (the monthly price of skipjack tuna). A 4 to 1 covariate to history length is unfavorable (**#TODO find some citation on recommended data length**) for deep learning applications.

The width vs. depth of our data points to a set of preliminary directions:

- Using dimensionality reduction methods
- Avoiding deep or nested networks (**#TODO - does this statement disqualify any of our good models?**)
- Exploiting other information contained in the series structure or pattern

#### GeoSpatial & Time Series Combination

All features were time series non-categorical data except for the calendar month and calendar year. Climate data sources had associated latitude and longitude metadata. Data structural characteristics were of key concern to mitigate the unfavorable data length to covariate count. Two strategies were identified:

- Capture repeatable time series characteristics.
- Include geospatial relative or absolute positions as covariates.

The latter (geospatial covariates) was identified as being significantly harder than the former (time series characteristics) and de-prioritized due to project schedule requirements. Climate data were flattened and not associated with relative position aside from covariate names.

Temporal characteristics were maintained and explored both through use of networks with memory (recurrent neural networks, long short term memory) and including time harmonics in the same example for input into multi-layer perceptrons.

### Feature Selection and Preprocessing

#### **Data Synthetization** 

Data synthetization was done with a PCA encoding that kept the maximum possible amount of components in the whole dataset (including the target), and then random noise was injected into the decoder. However, given that the maximum decoding matrix size achievable was 121 X 121, 364 features were lost in the process (producing a loss in variance explanation), and therefore the output was not similar enough to the original dataset to be used as a training set. A manual selection that removed the additional 364 features before applying the PCA encoding and decoding could have solved the problem, however due to time constraints, this approach was not attempted.

#### Ridge Regularization

**#TODO: Final feature count verification + time history count verify**
**#TODO: check with Josh re. how he wants to shift 1st to 3rd person here**

Since we began with 433 features with unknown, but certain, relationships, we knew that feature selection would be important to our model. Building a correlation matrix, we can easily see certain elements that would detract from the model (see figure X for colinearity example). By implementing a Ridge regression for regularization, we are able to identify 210 features that could be removed from the data.
> Inspiration and methodology from Akash Dubey

#### Principal Components

PCA was another method used for parameter selection. The number of features not only was big when compared to the number of examples, but also had redundant information. This allowed to compress the information these covariates provided and reduce them to 16 while only losing 21% of the explained variability. This also provided the oportunity to include new covariates that could potentially add to the predictability of the model without the concern of the number of features.

PCA was effective as a selector of the best variables for the model, but it came on a late phase on our experimentation. Some earlier models that had experienced a poor performance without pre-processing improved once PCA covariate selection was included. However, due to time constraints its inclusion was not exhaustive to all of the experiment branches that were developed.

## Linear Forecasting Baseline

Traditionally most time series analysis are univariate in approach. Other variables are not incorporated because the features themselves may have predicted values which will propagate to the time series variable being predicted. We wanted to test this approach of time series forecasting to to see if they perform better on predictions than those using features.

### Time Series Forecasting

The usage of time series models here is twofold:

- Obtain an understanding of the underlying forces and structure that produced the data
- Fit a model and proceed to forecast.

Time series analysis is the splitting of time series into 4 parts:
1. **Level**: Long-term gradual changes in the series.
1. **Trend**: The increase or decrease in data over a period of time.
1. **Seasonality**: When time series is affected by seasonal factors, a seasonal pattern occurs.
1. **Noise:** The variability in the observations that cannot be explained by the model.

These components combine in some way to provide the observed time series. For example, they may be added together to form a model such as:

`Y = levels + trends + seasonality + noise`

### Automatic Time Series Decomposition

The Statsmodel python library provides a function `seasonal_compose()` to automatically decompose a time series. The additive model was used for preliminary assessment of time series linear trending and seasonality.

![pic1](images/roopa1.png)
**Fig. 1** - *Automatic Time Series Decomposition*

Seasonalities were also assessed using a manual polynomial fit.

![pic1](images/roopa2.png)
**Fig. 2** - *Polynomial fit to find seasonalities*

We can see how the model to find a seasonality fits well to our data.

### Stationarity

As a test for checking Stationarity, we used both Autocorrelation and partial autocorrelation plots as well as *Dickey-Fuller* test. The purpose of using a Dickey-fuller test was to see how strongly our time series was defined by a trend.

The null hypothesis of the test is that the time series can be represented by a unit root and has some time-dependent structure. The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary. [[7]](#7)

![pic1](images/roopa3.png)
**Fig. 3** - *Dickey-Fuller test, Auto and Partial Correlation, Rolling mean and standard deviation*

With a p value of 0.23 indicates this series is a candidate for methods to make the target series stationary such as log scale transformation or smoothing.

### Methods for Time Series Forecasting

When looking at our data the main split was whether we had extra regressors (features) to our time series or just the series. Based on this we started exploring different methods for forecasting and their performance in different metrics.[[7]](#7)

We split our data into test training sets having 85 months of training data and 36 months of testing data.

#### Univariate Time Series Analysis

Three univariate time series models were fitted to the data to assess performance:

- Auto Regression (AR)
- Autoregressive integrated moving average (ARIMA): Combination of moving average and auto-regression model.
- Seasonal Autoregressive moving average (SARIMA): Extends ARIMA model by adding seasonal past values and/or forecast erros.

The following plots show the predictions on or 36 months test data by using Auto Regression (AR) and SARIMA models.

##### **Auto Regression (AR)**

![pic1](images/roopa4.png)
**Fig. 4** - *Auto Regression model*

##### **Seasonal Autoregressive Integrated Moving-Average (SARIMA)**

![pic1](images/roopa5.png)
**Fig. 5** -*Seasonal Autoregressive Integrated Moving-Average (SARIMA) model*

#### Multivariate Time Series Analysis

Random Forest and XGBoost multivariate methods were also applied to the data to assess performance.


##### **Random Forest (RF)**

**#TODO: Style choice - are we ok with the direct quote? Generally citations are attached to a relevant paraphrase**
"Random forest is an ensemble of decision tree algorithms. A number of decision trees are created where each tree is created from a different sample. It can be used for both classification and regression. In our case the final prediction is the average prediction across the decision trees (we used 5)."[[22]](#22)

![pic1](images/roopa6.png)
**Fig. 6** - *Random Forest model*

##### **XGBoost**

**#TODO: Style choice - are we ok with the direct quote? Generally citations are attached to a relevant paraphrase**
"XGBoost (Extreme Gradient Boost) provides a high-performance implementation of gradient boosted decision trees. Rather than training all of the models in isolation of one another like random forest, XG Boost trains models in succession"[[22]](#22)

![pic1](images/roopa7.png)
**Fig. 7** *XGBoost model*
### Evaluation Metrics

There are many measures that can be used to analyze the performance of our prediction so we will be using the top 4 most used metrics for time series forecasting.

- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Error (RMSE)
- R2 Coefficient of Determination (r2)

![pic1](images/roopa8.png)
**Fig. 8** - *Result metrics*

**#TODO: Style choice - are we ok with the direct quote? Generally citations are attached to a relevant paraphrase**
"For any data, that a Random Forest/XGBoost has not seen before, at best, it can predict an average of training values that it has seen before. If the Validation set consists of data points that are greater or less than the training data points, a Random Forest will provide us with Average results as it is not able to Extrapolate and understand the growing/decreasing trend in our data. 

Therefore, a Random Forest model does not scale well for time-series data and might need to be constantly updated in Production or trained with some Random data that lies outside our range of Training set."[[22]](#22)

Answering questions like “What would the price of SkipJack Tuna be for next Year?” becomes really difficult when using Random Forests.

### Conclusions

Fitting a Linear Model or a Neural Net, in this case, might be sufficient to predict data which has increasing or decreasing trends.

## Machine Learning / Deep Learning Model Results

### Baseline Results - MLP

An attempt was made at training a flat model without filtering the features and with no pre-processing. The model failed to pick any signal, as can be seen in the following plot:

![pic1](images/Baseline.png)

**Fig. 9** - *Confusion Matrix and Estimated Error for k = 5*

### Classification Transform

The problem was transformed from a continuous to a discrete output to try to improve the performance of the model. This experiment had two variations:

1. Distribute the prices in equal sized buckets (with trials for 3, 4 and 5).
2. Binary: provide directionality in terms of price increase or decrease with respect to the previous period.

The dispersion and range of prices within any given training and testing set was very similar so to avoid recalculating the buckets on each trial the whole set was used. Since the dataset was shuffled, the risk of bias remained very low, however other temporality concerns arose (to be discussed later in the report).
**#TODO: Sounds like a good candidate to be elaborated upon in the data description lead up**

In the price bucket variety classification accuracy decreased as the number of buckets increased. At the same time an estimated RMSE loss was reduced. The estimated RMSE was based on the difference of the averages of the buckets instead of the difference of the average of the bucket and the actual price, making the estimated RMSE lower that the actual RMSE. This can be seen in the following confusion matrix:

![pic1](images/ConfusionMatrixB5.png)

**Fig. 10** - *Confusion Matrix and Estimated Error for k = 5*

![pic1](images/ConfusionMatrixB4.png)

**Fig. 11** - *Confusion Matrix and Estimated Error for k = 4*

![pic1](images/ConfusionMatrixB3.png)

**Fig. 12** - *Confusion Matrix and Estimated Error for k = 3*

The price directionality variety model was sub-par. The model selected only one of the labels for the entire dataset. Additionally it was not consistent in the election since both labels have an occurrence probability of 50%.

The three main limitations of a discrete approach to the problem were the implication that price was limited to a fixed range known in advance, inaccurate classification, and the rigidity of a discrete prediction. Furthermore, the decrease of the estimated RMSE as the number of buckets increased is a strong indicator to keep the model as a continuous approach.

Finally, since trials were made shuffling the whole set, the model was filling voids in the past instead of predicting the future. This realization was taken into account in the next models so that data was split by time rather than by volume.

### **CNN Adaptation**

A challenge to train with the available data was that the number of features (484: 482 environmental covariates, and the index consisting of year and month) was greater than the number of examples (121). Furthermore, the existing number of features did not show sufficient explanatory power in previous models. To deal with this, three different alternatives were explored, and those that were successful were merged into a model:

#### **Selecting the most significant covariates.**

A PCA encoder was used to select the most meaningful components, and train the model using them instead of the actual features. Several trials were made with different cutoff parameters, and the final decision was to keep the most relevant 16 components (with a loss of 21% in the variance explanation). 

#### **Adding covariates that could complement the existing information.** 

During exploratory analysis, it was found that tuna prices followed a cyclical pattern, and that prices at a given point in time are related to adjacent historic prices. Time cyclicity was included into the model by breaking the signal into Fourier harmonics. To ensure that low frequencies remained the most relevant, the first harmonic parameters for time offset and length of the period were determined by minimizing MSE. Using those parameters, the six lowest frequencies were calculated and incorporated to the dataset.

![pic1](images/LowFreqHarmonic.png)

**Fig. 13** - *Adjustment of Lowest Frequency Harmonic*

The advantage of using Machine Learning instead of applying Fourier Series directly, was that coefficients for each harmonic could be determined in context with the rest of the covariates.
Additionally to the harmonics, two price related covariates were added to the dataset: the price average for the last 6 periods and the change in price between `t-2` and `t-1`.

#### **Establishing a network that could generalize a large set of features.** 

A CNN based on LeNet’s architecture [[1]](#1) was used to train the model. The input for this model were the resulting 16 main components after applying PCA plus the additional 8 variables. This was arranged in a 3 X 8 input matrix. The temporal split between the train and the test sets was made at 65/35% to ensure that the cycle described by the first harmonic was completely included in the training set. Data was randomized only for the train set after the split. The results of this model were better than the previous attempts, and a RMSE of 397 was obtained (for context, the average price was $1,577), with a correlation of 0.76.  

![pic1](images/PredictLeNet.png)

**Fig. 14** - *Performance of LeNet Network*

This model relies on the assumption that the price cyclicity observed will continue in the future, i.e. that it was not a matter of chance. The full cycle encompasses roughly five years, and could be produced by multi-annual weather patterns such as El Niño or La Niña. However, the recurrence of this pattern in the future is uncertain (and outside the scope of this project), and should be further analyzed to assure the applicability of the model in a general context.

- Actually not Lenet just did a really good job of pre-processing the datawith PCA
- 7 harmonics of various parameters included in PCA so 'pseudo-history' is included in model

### LSTM

The target variable's seasonality and time-based nature points to a LSTM network as well suited. Strong results were found when applying a standard implementation of this type of model: loss of 0.00509 over 3,000 epochs with lr of 0.03

![pic1](images/lstm_output.png)

### LSTM, Rolling Window

Success of the first LSTM inspired a secondary LSTM approach. The objective was to structure input data to capture seasonality without creating dependence on the structure of the data's macro-trend.

Examples were restricted to 12-length sequences, 4 sequences per batch for training. Test data was the most recent 12-period sequence, validation was the 12-period sequence prior to that.

Results were inconclusive as the project ended prior to completing the implementation.

### Temporal Fusion Transformer (TFT)

Data complexity led to an investigation of open source libraries and tools designed to perform time series analysis on targets with many covariates. PyTorch Forecasting [[26]](#26), an extension of PyTorch Lightning [[25]](#25) emerged as a candidate for open source application.

PyTorch Forecasting's advantages included:

- Pre-written building-blocks for net design including structures for custom data objects, data loaders, and a pre-structured model trainer and evaluator object.
- Boilerplates for complex model structures such as the "Temporal Fusion Transformer" (TFT).
- Built-in plotting, evaluation, and hyper-parameter methods.

PyTorch Forecasting's Disadvantages Included:

- Low-visibility into pre-written routines and methods.
- Coding and structural conventions and practices which were time-consuming to learn.

The Temporal Fusion Transformer is a recently introduced neural network architecture that combines RNN and CNN design elements. [[27]](#27)

![pic1](images/model_results_tft_diagram_frompaper.png)

**Fig. n** - : *TFT Architecture* [[27]](#27)

TFT advantages include:

- Built-in variable selection
- Capability to fuse categorical and time-series data
- Capability to differentiate between unknown and known future data for a given prediction horizon (e.g. future prices versus the month for the next 3 cycles).

TFT implementation used example code from PyTorch Forecasting libraries. Experimental forecasting horizons included 1, 2, 4, and 60 periods (months). Experimental maximum sequence lengths included 6, 12, 20, 24, and 60 periods (months). Training and Validation data were split by witholding the most recent forecasting horizon from the data set (e.g. the last 4 months) and then training on random sequence length selections within the earlier data.

The TFT's recurrent block was examined using a hyper-parameter optimizer. Hidden network size was varied between 12 and 32, the attention head size was varied between 1 and 4, gradient clip was between 0.01 and 0.5, and the dropout between 0.05 and 0.3. Frequently larger networks were found optimal (~20 hidden depth) but computational resource limitations precluded their full use or investigation. Assured access to more powerful GPUs would have enabled more exploration.

![pic1](images/model_results_tft_output_4ahead.png)

**Fig. n** - *TFT Results Predicting Ahead 4 Periods Using 20 Previous*

As typical with forecasting neural networks accuracy suffered as a strong function of forecast horizon and generally improved with training on greater sequence lengths.

![pic1](images/model_results_tft_output_1ahead.png)

**Fig. n** - *TFT Results Predicting Ahead 1 Period Using 23 Previous*

TFT implementation was not robust and hard to validate and evaluate. Use of the single, latest time period for validation offered limited insights into TFT prediction capabilities. Attempts to create additional validation points were difficult within the module architecture and would require more time.

True implementation would require examination of Pytorch Forecasting's libraries to prove faithful representation of the design. Additionally the TFT's tested accuracy was highly dependent on hyper-parameters. No decomposition or normalization was performed on TFT inputs in order to test the claimed integrated variable selection and scaling routines of PyTorch Forecasting. The TFT's capability to schedule certain weights by defining "categories" of inputs was not tested.

While inconsistent the TFT's capabilities and PyTorch Lightning's features suggest them as prime candidates for study given more time for validation and implementation.

### Future Work

Pricing projection based on diverse inputs is an area of research interest. Given more time a number of novel technologies and techniques could be applied to this problem.

#### Multi Modal Neural Networks

This project would be well suited to apply a multi modal neural network. The diversity of data sources in length and general characteristics (geospatial vs. abstract) mean there are probably advantages towards separately applying a variety of neural networks to different sources of the data.

Geospatial Multi Modal applications in general are an area of active research.[[28]](#28)

#### Spherical Convolutional Neural Networks

The structural information inherent in the climate information based on the data location was ignored for all models in this project. Traditional convolutional neural networks would not capture the structural information appropriately due to the 2-dimensional rectilinear structure of their inputs. For example 180 degrees West would be interpreted as maximum distance from 180 degrees East.

There are several examples of CNN input structures modified for spherical geometry data. [[31]](#31) There are additional examples of spherical CNNs being modified for geospatial data. [[#30]](#29) Optimal modeling of the climate data would include a pass through a spherical CNN.

## Results and Summary
### Tabulated Results
| Model | RMSE | Remarks |
|-|-|-|
| Best Linear Time-series |  | XGBoost |
| Flat MLP | 1.4e4 | Bad |
| Categorical CNN |  | Converged to Continuous |
| MLP + Harmonics |  |  |
| LSTM, Conventional |  |  |
| LSTM, Rotating | N/A | Never converged |
| TFT, 1-Ahead | ~200 | Module code unverified |

### Remarks

Successful models held several common characteristics:

- Incorporated time characteristics whether by hidden layers or flattened information from previous time steps in data.
- Heavily pre-processed input data to reduce colinarity
- <third thing that I can't think of right now>

Flattening the data and hoping for the best led to performance worse than traditional statistical models. Given more time it would have been value-added to explore every combination of 

### Conclusions

**#TODO: Write with Team**

## References and Citations

### Introduction
### Industry References and Definitions
> <a id="1">[1]</a> Inter-American Tropical Tuna Commission, https://www.iattc.org/

> <a id="2">[2]</a> Western and Central Pacific Fisheries Commission, https://www.wcpfc.int/

> <a id="3">[3]</a> Commission for the Conservation of Southern Bluefin Tuna, https://www.ccsbt.org/

> <a id="4">[4]</a> Onour, Ibrahim and Sergi, Bruno, Modeling and forecasting volatility in the global food commodity prices (January 1, 2011)

> <a id="5">[5]</a> Chen, Yu-Chin and Rogoff, Kenneth S. and Rossi, Barbara, Predicting Agri-Commodity Prices: An Asset Pricing Approach [May 10, 2010].

> <a id="6">[6]</a> Dabin Zhang, Shanyin Cheng, Liwen Ling and Qiang Xia, Forecasting Agricultural Commodity Prices Using Model Selection Framework With Time Series Features and Forecast Horizons [February 4, 2020].

> <a id="7">[7]</a> Jabez Harris, A Machine Learning Approach to Forecasting Consumer Food Prices [August 2017].


### 3rd Party Python Packages, Methodology, & Functions

> <a id="2">[2]</a> Chen, Yu-Chin and Rogoff, Kenneth S. and Rossi, Barbara, Predicting Agri-Commodity Prices: An Asset Pricing Approach (May 10, 2010)

> <a id="3">[3]</a> Dabin Zhang, Shanyin Cheng, Liwen Ling and Qiang Xia, Forecasting Agricultural Commodity Prices Using Model Selection Framework With Time Series Features and Forecast Horizons (February 4, 2020)

> <a id="4">[4]</a> Jabez Harris, A Machine Learning Approach to Forecasting Consumer Food Prices (August 2017)
### Dataset Citations

> <a id="5">[5]</a> Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2019): ERA5 monthly averaged data on single levels from 1979 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). (Accessed on [01-MAY-2021]), https://10.24381/cds.f17050d7


> <a id="6">[6]</a> Zhang, H.-M., B. Huang, J. Lawrimore, M. Menne, Thomas M. Smith, NOAA Global Surface Temperature Dataset (NOAAGlobalTemp), Version 4.0. NOAA National Centers for Environmental Information. doi: https://10.7289/V5FN144H [01-MAY-2021].


> <a id="7">[7]</a> Fetterer, F., K. Knowles, W. N. Meier, M. Savoie, and A. K. Windnagel. 2017, updated daily. Sea Ice Index, Version 3. Boulder, Colorado USA. NSIDC: National Snow and Ice Data Center. doi: https://doi.org/10.7265/N5K072F8. [01-MAY-2021].


> <a id="8">[8]</a> International Monetary Fund, Global price of Fish [PSALMUSDM], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PSALMUSDM, May 2, 2021.


> <a id="9">[9]</a> U.S. Bureau of Labor Statistics, Producer Price Index by Industry: Seafood Product Preparation and Packaging: Fresh and Frozen Seafood Processing [PCU3117103117102], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PCU3117103117102, May 2, 2021.


> <a id="10">[10]</a> U.S. Bureau of Labor Statistics, Producer Price Index by Industry: Specialty Food Stores: Fish and Seafood Markets [PCU445200445200102], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PCU445200445200102, May 2, 2021.


> <a id="11">[11]</a> U.S. Bureau of Labor Statistics, Tuna, Light, Chunk, Per Lb. (453.6 Gm) in U.S. City Average [APU0000707111], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/APU0000707111, May 2, 2021.


> <a id="12">[12]</a> U.S. Bureau of Labor Statistics, Import Price Index (End Use): Fish and Shellfish [IR01000], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IR01000, May 2, 2021.


> <a id="13">[13]</a> U.S. Bureau of Labor Statistics, Export Price Index (End Use): Fish and Shellfish [IQ01000], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IQ01000, May 2, 2021.


> <a id="14">[14]</a> International Monetary Fund, Global price of Shrimp [PSHRIUSDM], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PSHRIUSDM, May 2, 2021.

### 3rd Party Python Packages, Methodology, & Functions

> <a id="15">[15]</a> Jaime Ferrando Huertas, https://github.com/jiwidi/time-series-forecasting-with-python

> <a id="16">[16]</a> Akash Dubey. "Feature Selection Using Regularisation", 
<https://towardsdatascience.com/feature-selection-using-regularisation-a3678b71e499>

> <a id="17">[17]</a> Jason Brownlee. "Time Series Forecasting with the Long Short-Term Memory Network in Python", 
<https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
### Time Series Citations

> <a id="18">[18]</a> Davide Bruba. "An overview of time series forecasting models", 
<https://towardsdatascience.com/an-overview-of-time-series-forecasting-models-a2fa7a358fcb>

> <a id="19">[19]</a> Athul Anish. “Time Series Analysis”,
https://medium.com/swlh/time-series-analysis-7006ea1c3326

> <a id="20">[20]</a> Statworx Blog, “Time series forecasting with random forest”,
https://medium.com/@statworx_blog/time-series-forecasting-part-i-e30a16bac58a

> <a id="21">[21]</a> Indraneel Dutta Baruah, Analytics Vidya. “Combining Time Series Analysis with Artificial Intelligence: the future of forecasting”,
https://medium.com/analytics-vidhya/combining-time-series-analysis-with-artificial-intelligence-the-future-of-forecasting-5196f57db913

> <a id="22">[22]</a> Aman Arora, “Why Random Forests can’t predict trends and how to overcome this problem?”,
https://medium.datadriveninvestor.com/why-wont-time-series-data-and-random-forests-work-very-well-together-3c9f7b271631

> <a id="23">[23]</a> Jason Browniee, “How to Decompose Time Series Data into Trend and Seasonality”
https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/

### Model Development Citations

> <a id="24">[24]</a> Yann Le Cun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, Gradient-Based Learning Applied to Document Recognition, IEEE [November 1998]

> <a id="25">[25]</a> PyTorch Lightning
https://pytorch-lightning.readthedocs.io/en/latest/

> <a id="26">[26]</a> PyTorch Forecasting
https://pytorch-forecasting.readthedocs.io/en/latest/index.html

> <a id="27">[27]</a> Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister, "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting";
https://arxiv.org/pdf/1912.09363.pdf

> <a id="28">[28]</a> Jing Gao, Peng Li, Zhikui Chen, Jianing Zhang; "A Survey on Deep Learning for Multimodal Data Fusion." Neural Comput 2020; 32 (5): 829–864. doi:
https://direct.mit.edu/neco/article/32/5/829/95591/A-Survey-on-Deep-Learning-for-Multimodal-Data

> <a id="29">[29]</a> DeepSphere: Efficient spherical Convolutional Neural Network with HEALPix sampling for cosmological applications
https://arxiv.org/abs/1810.12186

> <a id="30">[30]</a> Deep Learning for Spatio - Temporal Data Mining: A Survey
https://arxiv.org/pdf/1906.04928.pdf

> <a id="31">[31]</a> Taco S. Cohen, Mario Geiger, Jonas Köhler, Max Welling: "Spherical CNNs"; ICLR 2018.
https://openreview.net/pdf?id=Hkbd5xZRb



## Source Code

We have published all of our source code to a public Github repo:

[CMU MSBA Team 4 Code Repository](https://github.com/josh-kennedy-7/cmu_msba_team_4_2022)
