#importing libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#reading data
df_ = pd.read_csv("Telco-Customer-Churn.csv")
df = df_.copy()
df.head()

#general information about the dataset
df.shape
df.info()

#Sum of missing observations in the data set
df.isnull().sum()

#Descriptive statistics of the dataset
df.describe().T

#type change
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')


#Giving new values to the "Contract" variable

df["Contract"].replace("Month-to-month",1,inplace=True)
df["Contract"].replace("One year",12,inplace=True)
df["Contract"].replace("Two year",24,inplace=True)
df["Contract"] = df["Contract"].astype(int)


# I thought that when we multiply the monthly amount collected from the customer and the contract period of the customer, we can find the total amount to be paid.
# I created a new variable called “new_totalcharges”.When I compared this new variable with the “TotalCharges” variable, I saw that the two were different.

df["new_totalcharges"]=df["MonthlyCharges"]*df["Contract"]
df[["new_totalcharges","TotalCharges"]].head()


#Number of transactions
df["Transection"]=df["TotalCharges"]/df["MonthlyCharges"]
df['Transection']=round( df['Transection'],0)



#Customer Lifetime Value calculation

def cltv_c(dataframe, profit=0.10):

    cltv_c = dataframe.groupby('customerID').agg({'Contract': lambda x: x.sum(),
                                                   'Transection': lambda x: x.sum(),
                                                   'TotalCharges': lambda x: x.sum()})

    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

    # Avg_order_value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']

    # Purchase_Frequency
    cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]

    # Repeat rate & Churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # Profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit

    # Customer Value
    cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])

    # Customer Lifetime Value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(cltv_c[["cltv"]])
    cltv_c["scaled_cltv"] = scaler.transform(cltv_c[["cltv"]])

    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c

final_df = cltv_c(df)

final_df.reset_index().head()


#averages of scaled_cltv in segment breakdown
final_df.groupby("segment").agg({"scaled_cltv": "mean"})

#Number of customers in group A segment
final_df[final_df["segment"]=="A"].shape

#Number of customers in group D segment
final_df[final_df["segment"]=="D"].shape

#number of customers with churn in the dataset
df[df["Churn"]=="Yes"].shape



