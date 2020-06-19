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

    The project required data from three different datasets. 

    '''
    ps_df = pd.read_csv('../../data/EXTR_RPSale.csv')
    b_df = pd.read_csv('../../data/EXTR_ResBldg.csv')
    p_df = pd.read_csv('../../data/EXTR_Parcel.csv', encoding='latin-1')

    return ps_df, b_df, p_df

def add_major_leading_zeros(majors):
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
    return add_major_leading_zeros(majors), add_minor_leading_zeros(minors)

def merge_dataframes(df1, df2, df3, type, cols):
    df = df1.merge(df2, how=type, on=cols).merge(df3, how=type, on=cols)
    return df

def encode_column(col):
    le = LabelEncoder()

    encoded = le.fit_transform(list(col))
    return pd.Series(encoded)

def create_dataframe():
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

    fig, ax = plt.subplots(2, 1, figsize=(10,12))
    sns.distplot(x, ax=ax[0])
    sns.boxplot(x, ax=ax[1])

def z_score(x, mean, std):
    z = (x-mean)/std
    return z

def corr_heatmap(df):
    sns.heatmap(df.corr())

def create_model(features, target):
    model = sm.OLS(target, features).fit()
    return model

def test_assumptions(df, model, ivar):
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
    # vif_features = pd.DataFrame()
    # vif_features['vif'] = [vif(idv_df.values, i) for i in range(idv_df.shape[1])]
    # vif_features['features'] = idv_df.columns
    # print('VIF: {}'.format(vif_features.vif.mean()))