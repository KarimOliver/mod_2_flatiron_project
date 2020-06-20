import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import statsmodels.stats.api as sms
import scipy.stats as stats
import statsmodels.stats.diagnostic as at


def create_initial_dataframes():
    ''' Creates initial dataframes from downloaded .csv files

    The project required data from three different datasets. Utilizing 
    the data_download notebook we stored the .csv files under the src/
    directory. This function stores those .csv files in dataframes.

    Returns:
        ps_df: dataframe made from Real Property Sales file
        b_df: dataframe made from Residential Building files
        p_df: dataframe made from Parcel file
    '''
    ps_df = pd.read_csv('../../data/EXTR_RPSale.csv')
    b_df = pd.read_csv('../../data/EXTR_ResBldg.csv')
    p_df = pd.read_csv('../../data/EXTR_Parcel.csv', encoding='latin-1')

    return ps_df, b_df, p_df

def add_major_leading_zeros(majors):
    ''' Add's leading zeroes to 'Major'

    Take's in the 'Major' column of a dataframe and add's
    leading zeroes if necessary. 'Major' code should have a
    length of 6, if it's shorter than that this function
    add's zeroes in front to make it length 6.

    Args:
        majors: pandas series of 'Major' codes 
    
    Returns:
        f_majors: list of 'Major' codes formatted with
                  leading zeroes
    '''
    # Empty list to store formatted major codes in
    f_majors = []
    
    # loop through current majors and add leading zeros
    for major in majors:
        #convert major to str and split
        s = [num for num in str(major)]
        # length of current major code
        l = len(s)
        # list with 6 zeros
        f_major = ['0', '0', '0', '0', '0', '0']
        # take current major and add to end of list
        f_major[6-l:] = s
        #join list into str
        f_major = ''.join(f_major)
        # convert to int append to formatted majors list
        f_majors.append(f_major)
    
    return f_majors

def add_minor_leading_zeros(minors):
    ''' Add's leading zeroes to 'Minor'

    Take's in 'Minor' column of a dataframe and add's
    leading zeros if necessary. 'Minor' code should have a
    length of 4, if it's shorter than that this function
    add's zeroes in front to make it length 4

    Args:
        minors: pandas series of 'Minor' codes

    Returns:
        f_minors: list of 'Minor' codes formatted with
                  leading zeroes
    '''
    # Empty list to store formatted minor codes in
    f_minors = []
    
    # loop through current minors and add leading zeros
    for minor in minors:
        #convert minor to str and split
        s = [num for num in str(minor)]
        # length of current minor code
        l = len(s)
        # list with 4 zeros
        f_minor = ['0', '0', '0', '0']
        # take current minor and add to end of list
        f_minor[4-l:] = s
        #join list into str
        f_minor = ''.join(f_minor)
        # convert to int append to formatted minors list
        f_minors.append(f_minor)
    
    return f_minors

def add_leading_zeros(majors, minors):
    ''' Call's add_leading_zeroes functions

    Takes in two pandas series, 'Major' and 'Minor'
    columns and calls add_major_leading_zeros and 
    add_minor_leading_zeros respectively.

    Args:
        majors: pandas series of 'Major' codes
        minors: pandas series of 'Minor' codes

    Returns:
        formatted 'Major' and 'Minor' codes

    '''
    return add_major_leading_zeros(majors), add_minor_leading_zeros(minors)

def merge_dataframes(df1, df2, df3, type, cols):
    ''' Creates one dataframe from three passed in dataframes

    Function takes in three dataframes, type of merge, and
    columns to merge them on. It then returns a single dataframe
    merged from the three arguement dataframes.

    Args:
        df1, df2, df3: three dataframes to merge together
        type: the type of merge to execute
        cols: columns to merge dataframes on
    
    Returns:
        df: merged dataframe

    '''
    df = df1.merge(df2, how=type, on=cols).merge(df3, how=type, on=cols)
    return df

def encode_column(col):
    ''' Use's LabelEncoder to create binary column

    Take's in a categorical column and uses LabelEncoder
    to create a numeric column

    Args:
        col: column to be encoded
    
    Returns:
        encoded column

    '''
    le = LabelEncoder()

    encoded = le.fit_transform(list(col))
    return pd.Series(encoded)

def create_dataframe():
    ''' Creates dataframe required for our analysis

    Function calls other functions to create, filter, and merge the data
    required for our analysis into a single dataframe. Filters for sale's
    from 2019 only. Filters for the specific columms we need from each 
    individual dataframe. Add's two binary columns, 'has_porch' and 'on_water'
    and encodes necessary columns

    Returns:
        df: formatted dataframe for analysis

    '''
    ps_df, b_df, p_df = create_initial_dataframes()

    # Declare columns to be used
    ps_columns = ['Major', 'Minor', 'SalePrice', 'PropertyClass']
    b_columns = ['Major', 'Minor', 'SqFtTotLiving', 'SqFtOpenPorch', 'SqFtEnclosedPorch', 'BldgGrade', 'Bedrooms', 'BathFullCount', 'BathHalfCount']
    p_columns = ['Major', 'Minor', 'TrafficNoise', 'PowerLines', 'OtherNuisances', 'TidelandShoreland', 'Township', 'SqFtLot', 'WfntLocation', 'WfntAccessRights']

    # Filter for 2019
    year = [True if int(d[6:]) == 2019 else False for d in ps_df['DocumentDate']]
    ps_df = ps_df[year]

    # Format leading zeros on major and minor codes
    ps_df['Major'], ps_df['Minor'] = add_leading_zeros(ps_df['Major'], ps_df['Minor'])
    b_df['Major'], b_df['Minor'] = add_leading_zeros(b_df['Major'], b_df['Minor'])
    p_df['Major'], p_df['Minor'] = add_leading_zeros(p_df['Major'], p_df['Minor'])

    # Create new dataframe from selected columns of each original dataframe
    df = merge_dataframes(ps_df[ps_columns], b_df[b_columns], p_df[p_columns], 'inner', ['Major', 'Minor'])

    # Add columns for model creation
    has_porch = [1 if ((op > 0) | (ep > 0)) else 0 for op, ep in zip(df['SqFtOpenPorch'], df['SqFtEnclosedPorch'])]
    df['has_porch'] = has_porch
    on_water = [1 if x>0 else 0 for x in df['TidelandShoreland']]
    df['on_water'] = on_water

    # Encode powerlines and othernuisances columns
    df['PowerLines'] = encode_column(df['PowerLines'])
    df['OtherNuisances'] = encode_column(df['OtherNuisances'])

    return df

def plot_dist(x):
    ''' creates distplot and boxplot of distribution

    Function takes in a sequence and produces two visuals.
    A seaborn distplot and a seaborn boxplot that allows for
    visual analysis of given distribution.

    Args:
        x: data array
    
    Returns:
        produces distplot and boxplot of given distribution
    '''

    fig, ax = plt.subplots(2, 1, figsize=(10,12))
    sns.distplot(x, ax=ax[0])
    sns.boxplot(x, ax=ax[1])

def z_score(x, mean, std):
    '''Computes z-score from given mean, standard deviationa, and data point

    Function calculates z-score from given mean,
    standard deviation, and data point passed in.

    Args:
        x: data point to compute z-score of
        mean: mean of population
        std: standard deviation of population
    
    Returns:
        z: z-score of given data point

    '''
    z = (x-mean)/std
    return z

def corr_heatmap(df):

    sns.heatmap(df.corr())

def create_model(features, target):
    '''Creates model from given features and target

    Function takes in feature variables and target variable
    and creates a linear regression model utilizing 
    statsmodels.api OLS function.

    Args:
        features: feature variables for model
        target: target variable for model
    
    Returns:
        model: linear regression model

    '''
    model = sm.OLS(target, features).fit()
    return model

def test_assumptions(df, model, ivar):
    '''Test linear regression assumptions of given model

    Function takes in a linear regression model, model dataframe,
    and a list of independent variables and performs the 4 assumption
    test for linear regression. Performs a rainbow fit test to check
    for linearity, jarque bera test to check normality, breusch-pagan
    test to check for homoscedasticity, and the variance inflation
    factor to check for independence. Prints output to screen.

    Args:
        df: dataframe used to create model
        model: linear regression model
        ivar: list of feature variables

    '''

    # model residuals
    resids = model.resid

    # df with only features
    idv_df = df[ivar]

    # Plot qq-plot for normality and scatterplot for homoscedasticity
    sm.graphics.qqplot(resids, dist=stats.norm, line='45', fit=True)

    fig, ax = plt.subplots()
    ax.scatter(resids, model.predict())

    # Rainbow fit test to check for linearity
    rb_test = at.linear_rainbow(model)

    #print results of rainbow fit test
    print('Rainbow test statistic: {}\nRainbow test p-value: {}'.format(rb_test[0], rb_test[1]))

    # Jarque-Bera (JB) test to check for normality
    jb_test = sms.jarque_bera(resids)

    #Print results of JB test
    print('JB test statistic: {}\nJB test p-value: {}'.format(jb_test[0], jb_test[1]))

    # Breusch Pagan test for homoscedasticity and scatter plot of resids and predicted values
    bp_test = at.het_breuschpagan(resids, idv_df)
    print('Breusch Pagan test statistic: {}\nBreusch Pagan p-value: {}'.format(bp_test[0], bp_test[1]))

    # Variance Inflation Factor (VIF) to check for independence
    vif_features = pd.DataFrame()
    vif_features['vif'] = [vif(idv_df.values, i) for i in range(idv_df.shape[1])]
    vif_features['features'] = idv_df.columns
    print('VIF: {}'.format(vif_features.vif.mean()))