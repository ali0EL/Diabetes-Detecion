import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import Final_Recommender
from Final_Recommender import Recommender

def recomender(user_id):
   model = joblib.load('Recommender_model1.pkl',mmap_mode ='r')
   return model.recommend(user_id)


def prediction_model():
    gender =['Male','Female']
    
    a = st.selectbox('Pick Your Gender',gender)
    if a =='Male':
        X1=0
    else:
        X1=st.number_input('Enter Number of Pregnancies',0,20)
   
    X2=st.number_input('Enter Glucose Concentration Value ')
    X3=st.number_input('Enteer Blood Pressure Value')
    X4=st.number_input('Enter Skin Thickness Value')
    X5=st.number_input('Enter Insulin Concentration Value ')
    M =st.number_input('Enter Your Weight in Kilogrames')
    H =st.number_input('Enter Your Hight in meter',min_value=0.40)
    X6 = M/H**2
    X7=st.number_input('Enter DiabetesPedigreeFunction ')
    X8=st.number_input('Enter Your Age ')
    
    model = joblib.load('small_data')
    prediction = model.predict([[X1,X2,X3,X4,X5,X6,X7,X8]])
    
    
    
    return prediction
    



def main():
    
    menu1 = ['Home','Diabetes App','Recommendation']
    task = st.sidebar.selectbox('Menu',menu1)
    if task =='Home':
        
        
        
        st.title('HomePage')
       
        
        image =Image.open('diabetes-report-card-SM.jpg')
        st.image(image,use_column_width=True)
        st.info("""Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.
                           
                           
Most of the food you eat is broken down into sugar (also called glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body’s cells for use as energy.


If you have diabetes, your body either doesn’t make enough insulin or can’t use the insulin it makes as well as it should. When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease.


There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help. Taking medicine as needed, getting diabetes self-management education and support, and keeping health care appointments can also reduce the impact of diabetes on your life.""")
    elif task =='Diabetes App':
        st.title('Diabete prediction App')
        
        pred= prediction_model()
        if st.button('Predict'):
            if pred ==0:
                st.success('Patient is not diabetic')
            elif pred== 1:
                st.warning('Patient may be diabetic')
       
            
    elif task =='Recommendation':
        st.title('Recommendation App')
        user_id = st.text_input("Enter your Id (must be in this form User_'number <10000') ").lower().split('_')
       
                  
                       
        if st.button('Recommend'):
            if user_id[0] != 'user' or int(user_id[1]) >10000 :
                st.warning("invalid User_id")
            try:          
                x = user_id[0]+'_'+user_id[1] 
                     
                a = recomender(x)
                
                
                df=pd.DataFrame(a)
                st.dataframe(df)
                st.download_button('Download Recommendation', df.to_csv(), file_name="user Recommendation.csv ", mime='text/csv')
            except:
                pass
    
    else:
        st.warning('Incorrect username or password')
    
if __name__=='__main__':
    main()
       
