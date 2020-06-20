# Table of Contents
- Exploratory Notebooks
- Report Notebook
- Project Presentation
- Data download
- src/ directory with project source code
- figures/ directory with project visuals
- Data references
- Project Conda environment

# Context of Project
Our project task was to create a linear regression model providing insight into what influenced the sale price of homes in King County. Specifically, we wanted to explore the effect having a porch, being on the waterfront, and the presence of nuisances (power lines, traffic noise, airport noise, etc) had on home sale price. In order to accomplish the project task we created a multiple linear regression model to determine what influence each of these had on the sale price of a home. In addition to these three claims we also explored the effects of location and home quality.

Our analysis was done with the intention of providing insight on housing price indicators for potential homebuyers in King County. To taylor our analysis more towards the average homebuyer we focused on the less extreme home sale prices by removing necessary outliers from our dataset.

The data used in this analysis can be found at [King County Department of Assessments](https://info.kingcounty.gov/assessor/DataDownload/default.aspx). The specific data used will be introduced later.

# Project Plan
As discussed earlier our model was to provide insight on housing price indicators for potential homebuyers. In order to do this we needed to create the most accurate model possible. After exploring the dataset, we found a strong correlation between the total square footage of a home and it's sale price. We decided to create a baseline model that used total square footage to predict the sale price of a home. We then used this model to judge the effectiveness of our contious iterations off of.

After creating our baseline model and following every model iteration, we needed to test the assumptions of linear regression to ensure our analysis was statistically significant. We performed a rainbow fit test to check the linearity between the features and target variables, the Jarque-Bera test to check for normality of the residuals from the resulting model, the Breuch-Pagan test to check for homoscedasticity, and calculated variance inflation factor to check for multicollinearity between the model's features. We then used these assumptions to improve the next iteration of our model.

# Preview of Results

# Data
We utilized three datasets for our analysis, all of which can be found at [King County Department of Assessments](https://info.kingcounty.gov/assessor/DataDownload/default.aspx). The specific datasets used are as follows:
- [Real Property Sales](https://aqua.kingcounty.gov/extranet/assessor/Real%20Property%20Sales.zip)
- [Residential Buildings](https://aqua.kingcounty.gov/extranet/assessor/Residential%20Building.zip)
- [Parcel](https://aqua.kingcounty.gov/extranet/assessor/Parcel.zip)

The Real Property Sales dataset provided a information on property sale transactions in King County such as sale date and price. The Residential Buildings dataset provided general information on the homes that were sold in King County such as total square footage and number of bedrooms. The Parcel dataset gave information on the property a given home was located, such as if it was located on a waterfront or any nuisances were present. All three datasets contained a 'Major' and 'Minor' column which were the identifying variables for the data. These were the columns used to combine the datasets together. The specific columns we used from each dataset are as follows (not including the 'Major' and 'Minor' columns):
Real Property Sales:
- SalePrice

Residential Buildings:
- SqFtTotLiving
- SqFtOpenPorch
- SqFtEnclosedPorch
- BldgGrade
- Bedrooms
- BathFullCount
- BathHalfCount
- SqFtGarageAttached
- NbrLivingUnits

Parcel:
- TrafficNoise
- PowerLines
- OtherNuisances
- TidelandShoreland
- Township
- SqFtLot
- LakeWashington
- WfntLocation

As stated earlier the datasets contains information on all property sales in King County. For the purpose of our analysis we focused only on those from 2019, so our resulting dataframe was filtered to contain only the information on home sales from 2019.

