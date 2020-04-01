import streamlit as st
import pandas as pd
import numpy as np
st.title('Snack Fair Recommendation System')

df = pd.read_csv("Trained_output.csv")
df1 = df
st.dataframe(df)
df_coordinate = df.iloc[:,3:5]
st.map(df_coordinate)
del df1['lon']
del df1['lat']
df_sum = df1.groupby(['Snack Subscription ID']).sum()
#df_coordinate = pd.read_csv('Loudoun_Parcel_XY.csv')
#df_coordinate = df_coordinate.iloc[:102960,0:]
#df_coordinate = df_coordinate.rename(columns={"POINT_X":"lon","POINT_Y":"lat"}, errors="raise")
#coordinate_list = list(zip(df_coordinate.POINT_X.values,coordinate.POINT_Y.values))
# df_coordinate['lat'] = df_coordinate['lat']/100
# df_coordinate['lon'] = df_coordinate['lon']/100

# df_coordinate = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [77.76, -125.4],
#     columns=['lat', 'lon'])
st.dataframe(df_sum)
st.line_chart(df_sum)
st.bar_chart(df_sum)
#st.pie_chart(df_sum)

