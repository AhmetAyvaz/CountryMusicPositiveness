import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
import statsmodels.api as sm
rawData = pd.read_csv("C:/Users/USER/Desktop/Projects/Project/csv/Top-50-musicality-global.csv")

rawData.drop("Unnamed: 0",axis=1,inplace=True)

temp = pd.DataFrame()
tempData = rawData.copy()

Q1 = tempData["Energy"].quantile(0.25)
Q3 = tempData["Energy"].quantile(0.75)
IQR = Q3-Q1
highestLimit = Q3 + 1.5*IQR
lowestLimit = Q1 - 1.5*IQR
outlierhighest = tempData[tempData["Energy"] >= highestLimit]
outlierlowest = tempData[tempData["Energy"] <= lowestLimit]

index1 = sorted([])
for i in outlierlowest.index.tolist():
    index1.append(i)
for i in tempData.index.tolist():
    for j in index1:
        if i == j:
            tempData["Energy"].iloc[i] = tempData["Energy"].mean()


Q1 = tempData["Danceability"].quantile(0.25)
Q3 = tempData["Danceability"].quantile(0.75)
IQR = Q3-Q1
highestLimit = Q3 + 1.5*IQR
lowestLimit = Q1 - 1.5*IQR
outlierhighest = tempData[tempData["Danceability"] >= highestLimit]
outlierlowest = tempData[tempData["Danceability"] <= lowestLimit]


Q1 = tempData["Positiveness"].quantile(0.25)
Q3 = tempData["Positiveness"].quantile(0.75)
IQR = Q3-Q1
highestLimit = Q3 + 1.5*IQR
lowestLimit = Q1 - 1.5*IQR
outlierhighest = tempData[tempData["Positiveness"] >= highestLimit]
outlierlowest = tempData[tempData["Positiveness"] <= lowestLimit]

index2 = sorted([])
for i in outlierlowest.index.tolist():
    index2.append(i)
for i in outlierhighest.index.tolist():
    index2.append(i)
for i in tempData.index.tolist():
    for j in index2:
        if i == j:
            tempData["Positiveness"].iloc[i] = tempData["Positiveness"].mean()


temp = tempData.groupby("Country").agg({"Positiveness":"mean","Energy":"mean","Danceability":"mean"})
country = temp.reset_index()["Country"]
list1 = []
list2 = []
list3 = []
list4 = []
for i in country:
    list1.append(i)
for i in temp["Positiveness"]:
    list2.append(i)
for i in temp["Energy"]:
    list3.append(i)
for i in temp["Danceability"]:
    list4.append(i)
data = pd.DataFrame({"Country":pd.Series(list1),"Positiveness":pd.Series(list2),"Energy":pd.Series(list3),"Danceability":pd.Series(list4)})

shapiroPositiveness = stats.shapiro(data["Positiveness"])
shapiroEnergy = stats.shapiro(data["Energy"])
shapiroDanceability = stats.shapiro(data["Danceability"])

corr = pg.pairwise_corr(data[["Positiveness","Energy","Danceability"]],method="spearman")

partialCorr = pg.partial_corr(data[["Positiveness","Energy","Danceability"]],x="Positiveness",y="Energy",covar="Danceability",method="spearman") 

partialCorr = pg.partial_corr(data[["Positiveness","Energy","Danceability"]],x="Positiveness",y="Danceability",covar="Energy",method="spearman")

partialCorr = pg.partial_corr(data[["Positiveness","Energy","Danceability"]],x="Energy",y="Danceability",covar="Positiveness",method="spearman")


x = data[["Energy","Danceability"]]
y = data["Positiveness"]

constant = sm.add_constant(x)
model = sm.OLS(y,constant).fit()
summary = model.summary()

high = data[data["Positiveness"] > data["Positiveness"].mean()][["Country","Positiveness"]].sort_values(ascending=False,by="Positiveness")
low = data[data["Positiveness"] <= data["Positiveness"].mean()][["Country","Positiveness"]].sort_values(ascending=False,by="Positiveness")
