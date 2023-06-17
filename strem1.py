import streamlit as st
import pandas as pd
import numpy as np
import warnings 
import pickle
import pylab 
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings('ignore')

dataset=pd.read_csv('dataset.csv',compression='gzip')

st.subheader('BMI Calculator')

weight_in_kgs =st.number_input("Weight_in_kgs",min_value=1)

height_in_mtr =st.number_input("Height_in_mtr",min_value=1.0,max_value=3.0,step=0.1)

age=st.number_input("Age (yr)",min_value=1)

#sex=male (1) or female(2)
st.text('Enter 1 for male or 2 for female')
sex=st.number_input("male (1) or female(2)",min_value=1)

BMI = (weight_in_kgs)/((height_in_mtr)*(height_in_mtr))
st.text(BMI)

st.divider()
st.subheader('BMI status:-')
if BMI <16:
    st.text('under weight')
elif BMI <17 and BMI >=16:
    st.text('under weight')
elif BMI <18.5 and BMI >=17:
    st.text('under weight')
elif BMI <25 and BMI >=18.5:
    st.text('normal')
elif BMI <30 and BMI >=25:
    st.text('over weight')
else :
    st.text('obesity')


### BMR (kcal/day)

if sex == 1:
    BMR = 10 * weight_in_kgs + 6.25 *height_in_mtr*1000 - 5 *age  + 5
    st.text('kcal req per day {}'.format(BMR))
else :
    BMR = 10 * weight_in_kgs + 6.25 *height_in_mtr*1000 - 5 *age  -161
    st.text('kcal req per day {}'.format(BMR))

st.divider()

number_of_meals=st.slider('Meals per day',min_value=3,max_value=5,step=1,value=3)
if number_of_meals==3:
    meals_calories_perc={'breakfast':0.35,'lunch':0.40,'dinner':0.25}
    st.text('breakfast:{},lunch:{},dinner:{}'.format(0.35*BMR,0.4*BMR,0.25*BMR))
    #for lunch only
    Calories=0.4*BMR
elif number_of_meals==4:
    meals_calories_perc={'breakfast':0.30,'morning snack':0.05,'lunch':0.40,'dinner':0.25}
    st.text('breakfast {}, morning snack {},lunch {},dinner {}'.format(0.30*BMR,0.05*BMR,0.40*BMR,0.25*BMR))
    #for lunch only
    Calories=0.40*BMR
else:
    meals_calories_perc={'breakfast':0.30,'morning snack':0.05,'lunch':0.40,'afternoon snack':0.05,'dinner':0.20}
    st.text('breakfast {}, morning snack {},lunch {},afternoon snack {},dinner {}'.format(0.30*BMR,0.05*BMR,0.40*BMR,0.05*BMR,0.20*BMR))
    #for lunch only
    Calories=0.40*BMR




FatContent = st.number_input("fat content",min_value=0.00,max_value=100.00)
SaturatedFatContent = st.number_input("SaturatedFatContent",min_value=0.00,max_value=13.00)
CholesterolContent = st.number_input("CholesterolContent",min_value=0.00,max_value=300.00) 
SodiumContent= st.number_input("SodiumConten",min_value=0.00,max_value=2300.00) 
CarbohydrateContent= st.number_input("CarbohydrateContent",min_value=0.00,max_value=325.00) 
FiberContent= st.number_input("FiberContent",min_value=0.00,max_value=40.00) 
SugarContent= st.number_input("SugarContent",min_value=0.00,max_value=40.00) 
ProteinContent= st.number_input("ProteinContent",min_value=0.00,max_value=200.00) 

df=['Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']

df=np.array(df).reshape(-1, 1)
df=df.reshape(1,9)
print(df)


neigh = pickle.load(open('neigh.pkl', 'rb'))

#'''
def recommand(dataframe,_input,max_nutritional_values,ingredient_filter=None,params={'return_distance':False}):
    extracted_data=extract_data(dataframe,ingredient_filter,max_nutritional_values)
    prep_data,scaler=scaling(extracted_data)
    neigh=nn_predictor(prep_data)
    pipeline=build_pipeline(neigh,scaler,params)
    return apply_pipeline(pipeline,_input,extracted_data)


def extract_data(dataframe,ingredient_filter,max_nutritional_values):
    extracted_data=dataframe.copy()
    for column,maximum in zip(extracted_data.columns[6:15],max_nutritional_values):
        extracted_data=extracted_data[extracted_data[column]<maximum]
    if ingredient_filter!=None:
        for ingredient in ingredient_filter:
            extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(ingredient,regex=False)] 
    return extracted_data

def scaling(dataframe):
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(dataframe.iloc[:,6:15].to_numpy())
    return prep_data,scaler

def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)
    return neigh


def build_pipeline(neigh,scaler,params):
    transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    return pipeline

def apply_pipeline(pipeline,_input,extracted_data):
    return extracted_data.iloc[pipeline.transform(_input)[0]]







if st.button('Suggest for lunch'):
    recommand(dataset,df,Calories)
   

#'''
