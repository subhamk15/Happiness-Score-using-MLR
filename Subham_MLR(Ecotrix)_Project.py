#!/usr/bin/env python
# coding: utf-8

# # <span style='color:Red; font-family:Helvetica; font-size:1em'> PREDICTING HAPPINESS SCORE OF A COUNTRY USING SEVERAL FACTORS BY REGRESSION ANALYSIS </span>
# # <span style='color:Black; background :yellow; font-family:Helvetica; font-size:2em'>  Assignment Report </span>
# # <span style='color:Blue; font-family:Helvetica; font-size:1em'>  Econometrics Modelling and Forecasting </span>
# 

# ## <span style='color:Black ; font-family:Helvetica; font-size:1em'>  Submitted by --> SUBHAM KUMAR </span>
# ## <span style='color:Black ; font-family:Helvetica; font-size:1em'>  MOR: Sem 2 </span>
# ## <span style='color:Black ; font-family:Helvetica; font-size:1em'>  South Campus </span>
# ## <span style='color:Black ; font-family:Helvetica; font-size:1em'>  Roll No. - 21/1649 </span>

# # <center style='color:Brown ; font-family:Helvetica; font-size:1em'>Brief Report</center>

# <span style='color: black ; font-family:Helvetica; font-size:1em'>The World Happiness Report is a landmark survey of the state of global happiness that ranks 156 countries by how happy their citizens perceive themselves to be. The data we have taken for our report is from 2021 which maps countries happiness based on the factors such as GDP, Health life expectancy and so on.
# 
# In this project, we will predict the happiness score based on different factors. Below are the dependent and Independent variables used for the regression model.
#     
# Dependent Variable - y = Happiness Score
#     
# Independent Variable 
#                        
#                        x1 = GDP per capita
#     
#                        x2 = Social support
#     
#                        x3 = Healthy life expectancy
#     
#                        x4 = Freedom to make life choices
#     
#                        x5 = Generosity
#     
#                        x6 = Perceptions of corruption
#     
# 
# </span>

# [Source Data Set] :  (https://www.kaggle.com/datasets/priyanka841/2019-world-happiness-report-csv-file)

# # Importing the Libraries

# In[2]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.compat import lzip
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
sns.set()


# # Loading the data

# In[4]:


df = pd.read_csv(r'C:\Users\Subham PC\OneDrive\Desktop\Operation Research\SEM 2\Econometrics Project\2019 world happiness report.csv')
df.head(10) #displaying first 10 rows of our data


# # Displaying descriptive statistics of the variables

# In[123]:


df.describe(include='all')


# # Removing the irrelevent column 'country' and 'overall rank' 

# In[5]:


df = df.drop(['Country or region', 'Overall rank'],axis=1)


# # Checking for missing values in our data

# In[127]:


df.isnull().sum()


# <span style='color:Blue; font-family:Helvetica; font-size:1em'>  There is no missing values in our data </span>

# # Checking the OLS Assumptions

# # 1. Linearity

# <span style='color:Black; background :LightGreen; font-family:Helvetica; font-size:em'> Linearity means that the mean of the response variable is a linear combination of the parameters
# (regression coefficients) and the predictor variables or in other words, the relationship between
# x(independent features) and the mean of y(dependent feature) is linear. We check for this assumption using
# a scatter plot. </span>

# In[243]:


#Plotting scatter plot to check for the relationship between different factors and Happiness Score
f, (ax1, ax2, ax3) = plt.subplots(3, 2, sharey=True, figsize =(20,15))
ax1[0].scatter(df['GDP per capita'],df['Score'], color = 'hotpink')
ax1[0].set_title('GDP per capita and Happiness Score', fontsize=20)
ax1[1].scatter(df['Social support'],df['Score'], color = 'Blue')
ax1[1].set_title('Social support and Happiness Score', fontsize=20)
ax2[0].scatter(df['Healthy life expectancy'],df['Score'], color = 'Red')
ax2[0].set_title('Healthy life expectancy and Happiness Score', fontsize=20)
ax2[1].scatter(df['Freedom to make life choices'],df['Score'], color = 'Violet')
ax2[1].set_title('Freedom to make life choices and Happiness Score', fontsize=20)
ax3[0].scatter(df['Generosity'],df['Score'], color = 'Orange')
ax3[0].set_title('Generosity', fontsize=20)
ax3[1].scatter(df['Perceptions of corruption'], df['Score'], color = 'Black')
ax3[1].set_title('Perceptions of corruption', fontsize=20)
plt.show()


# # <span style='color:Black; background :Silver; font-family:Helvetica; font-size:0.5em'> We can clearly see that all the factors have almost a linear relationship with our dependent variable </span>

# # 2. Multicollinearity

# <span style='color:Black; background :LightGreen; font-family:Helvetica; font-size:em'> Multicollinearity is a condition in which the independent variables are highly correlated with each other. In
# regression analysis, we assume that there is no multicollinearity between the features. To check for this
# condition we use, the variance inflation factor. If VIF > 10, we conclude that there is multicollinearity. </span>

# In[125]:


df.columns #displaying columns in our dataset


# In[128]:


X=df.drop('Score', axis=1) #defining our independent variables


# In[129]:


Y=df['Score'] #defining our dependent variable


# In[130]:


#Applying Box-Cox transformation to normalise our data
pt=PowerTransformer(method='box-cox')
X_tf=pt.fit_transform(X+0.0000001)
X_tf=pd.DataFrame(X_tf,columns=X.columns)
X_tf.head()


# In[131]:


con_data=pd.concat([X_tf, Y], axis=1)


# In[244]:


#Plotting the heatmap to check for the correlation between our independent variables
plt.figure(figsize=(15,15))
sns.heatmap(con_data.corr(), annot=True, cmap='RdYlGn')
plt.show()


# # Checking for multicollinearity using variance inflation factor

# In[133]:


vif_data=pd.DataFrame()
vif_data["feature"]=X_tf.columns

vif_data["VIF"]=[variance_inflation_factor(X_tf.values, i)
                 for i in range(len(X_tf.columns))]

vif_data


# # <span style='color:Black; background :Silver; font-family:Helvetica;font-size:0.5em'> We can clearly see all the VIF values in the table above is less than 5. Hence, there is no multicollinearity in our data </span>

# # 3. Auto-correlation

# <span style='color:Black; background :LightGreen; font-family:Helvetica; font-size:em'> Autocorrelation occurs when the residuals(actual-predicted) are not independent from each other. To test
# for auto-correlation, we take the help of the Durbin-Watson test. A 'statsmodel' summary table will give us
# a comprehensive summary analysis and the required Durbin-Watson test value.
#  </span>

# In[134]:


import statsmodels.api as sm


# In[136]:


X_endog1=sm.add_constant(X_tf)


# In[137]:


model=sm.OLS(Y,X_endog1)
model.fit().summary()


# # <span style='color:Black; background :Silver; font-family:Helvetica; font-size:0.5em'> Here, from the above table, we can clearly see that our Durbin-Watson value is 1.660 which is close to 2 and indicates that there is no autocorrelation present in our dataset
#  </span>

# # 4. Heteroscedasticity

# <span style='color:Black; background :LightGreen; font-family:Helvetica; font-size:em'> 
#     Heteroscedasticity or non-constant variance occurs if if the variability of the random disturbance is unequal
# across elements of the vector. Here, we will be using Goldfeld Quandt test to look for heteroscedasticity.
#     * Null hypothesis
#  H0 : Homoscedasticity or Absence of heteroscedasticity ; 
#    * Alternate hypothesis 
#  H1 :Presence of heteroscedasticity </span>

# In[138]:


mod=model.fit()


# In[139]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(mod.resid,X_endog1)
lzip(name, test)


# # <span style='color:Black; background :Silver; font-family:Helvetica; font-size:0.5em'> For any hypothesis test, the decision rule is: * If p-value < level of significance (alpha); then null hypothesis is rejected. * If p-value > level of significance (alpha); then we fail to reject the null hypothesis. </span>

# # <span style='color:Black; background :Silver; font-family:Helvetica; font-size:0.5em'> Here, our p value > 0.05, level of significance. Therefore, we fail to reject the null hypothesis. Hence, we can conclude the absence of heteroscedasticity in our dataset.</span>

# # Building a Linear Regression Model

# # Train Test Split

# In[210]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_endog1,Y, test_size=0.2,random_state=100)


# In[222]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)


# In[223]:


y_pred = reg.predict(x_train)


# In[224]:


#Scatter Plot between the actual and the predicted values
plt.scatter(y_train, y_pred)
plt.xlabel('Targets/Actual',size=18)
plt.ylabel('Predictions',size=18)
plt.show()


# In[225]:


#probability distribution of the residual
sns.distplot(y_train - y_pred)
plt.title("Residuals PDF", size=18)


# In[226]:


#Finding the R^2 Value
print("The R squared value is",reg.score(x_train,y_train))


# In[227]:


#Finding the intercept of our fitted line
print("The intercept of our regression model is",reg.intercept_)


# In[228]:


#Finding the coefficients of our regression  model
print("The bias/coefficients of our regression model is",reg.coef_)


# # TESTING THE MODEL

# In[229]:


y_test_pred = reg.predict(x_test)


# In[230]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# ### Finding the R squared value and MSE of the model

# In[231]:


print('The model R squared value is:',r2_score(y_true=y_test,y_pred=y_test_pred))


# In[232]:


print('The mean squared error is:',mean_squared_error(y_test,y_test_pred))


# # Actual Vs Predicted Values

# In[233]:


#Plotting a scatter plot to see the relation between actual and predicted values
plt.scatter(y_test, y_test_pred)
plt.xlabel('Targets',size=15)
plt.ylabel('Predictions',size=15)
plt.title('Actual vs predicted in the test set',size=18)
plt.show()


# In[234]:


#creating a new dataframe with the actual target values and the model predicted values
df_pf = pd.DataFrame((y_test_pred), columns=['Prediction'])
df_pf['Target'] = (y_test)


# In[235]:


y_test = y_test.reset_index(drop=True)
y_test.head()


# In[236]:


df_pf['Target'] = (y_test)
df_pf.head(7)


# In[237]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)


# #### Displaying the targets and predictions from the most accurate ones to the least accurate value.
# 

# In[238]:


pd.options.display.max_rows = 200
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])


# # <center style='color:Brown ; font-family:Helvetica; font-size:1em'>Conclusion</center>

# <span style='color: black ; font-family:Helvetica; font-size:1em'> In this project, we have deployed a multiple regression model on our dataset of World Happiness Report(2021). We predcticted our dependent variable (Happiness Score) based on the six different factors that contributes towards it. 
#     
# Our initial aim was to check and handle the four basic assumptions of 
# regression: Linearity, No Multicollinearity, No Autocorrelation, Homoscedasticity. We 
# successfully established these assumptions using various techniques in python.'
#     
# Furthermore, we built a linear regression model using sklearn and found out the 
# optimal coefficients for each feature as well as the intercept value. Our multiple regression model is:
#        
#         Happiness Score = 5.39001663344672 + 0.20925027(X1) + 0.41120071(X2) + 0.32350341(X3) + 0.1545508(X4) +  0.06703365(X5) + 0.10206294(X6)
#    
#     Where X1, X2, X3, X4, X5, X6 are Score, GDP per capita, Social support, Healthy life expectancy,   
#     Freedom to make life choices, Generosity and Perceptions of corruption.
# 
# This model was further used to predict values in the test dataset and check the accuracy by calculating 
# the residuals and the percentage difference for each of the predicted values.
#     
# 
# </span>

# In[ ]:




