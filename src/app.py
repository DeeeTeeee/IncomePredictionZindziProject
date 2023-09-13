#Importing the libraries
import gradio as gr
import pickle
import pandas as pd
import numpy as np
import joblib
from PIL import Image

#using joblib to load the model:
encoder = joblib.load('encoder.joblib') # loading the encoder
scaler = joblib.load('scaler.joblib') # loading the scaler
model = joblib.load('ml.joblib') # loading the model


# Create a function that applies the ML pipeline and makes predictions
def predict(age,gender,education,marital_status,race,employment_stat,wage_per_hour,working_week_per_year,industry_code,occupation_code,
    total_employed,vet_benefit,tax_status,gains,losses,stocks_status,citizenship,mig_year,importance_of_record):



    # Create a dataframe with the input data
     input_df = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'education': [education],
        'marital_status': [marital_status],
        'race': [race],
        'employment_stat': [employment_stat],
        'wage_per_hour': [wage_per_hour],
        'working_week_per_year': [working_week_per_year],
        'industry_code': [industry_code],
        'occupation_code': [occupation_code],
        'total_employed': [total_employed],
        'vet_benefit': [vet_benefit],
        'tax_status': [tax_status],
        'gains': [gains],
        'losses': [losses],
        'stocks_status': [stocks_status],
        'citizenship': [citizenship],
        'mig_year': [mig_year],
        'importance_of_record': [importance_of_record]

 })

# Create a list with the categorical and numerical columns
     cat_columns = [col for col in input_df.columns if input_df[col].dtype == 'object']
     num_columns = [col for col in input_df.columns if input_df[col].dtype != 'object']

    # # Impute the missing values
    #  input_df_imputed_cat = cat_imputer.transform(input_df[cat_columns]) 
    #  input_df_imputed_num = num_imputer.transform(input_df[num_columns]) 

    # Encode the categorical columns
     input_encoded_df = pd.DataFrame(encoder.transform(input_df[cat_columns]).toarray(),
                                   columns=encoder.get_feature_names_out(cat_columns))

    # Scale the numerical columns
     input_df_scaled = scaler.transform(input_encoded_df)
     input_scaled_df = pd.DataFrame(input_df_scaled , columns = num_columns)


    #joining the cat encoded and num scaled
     final_df = pd.concat([input_encoded_df, input_scaled_df], axis=1)


    # Make predictions using the model
     predict = model.predict(final_df)


     prediction_label = "INCOME ABOVE LIMIT" if predict.item() == '1' else "INCOME BELOW LIMIT"


     return prediction_label

     #return predictions

#define the input interface

input_interface = []

with gr.Blocks(css=".gradio-container {background-color:silver}") as app:
    title = gr.Label('INCOME PREDICTION APP.')
    img = gr.Image("income_image.png").style(height= 210 , width= 1250)

 
    with gr.Row():
        gr.Markdown("This application provides predictions on whether a person earns above or below the income level. Please enter the person's information below and click PREDICT to view the prediction outcome.")

    with gr.Row():
        with gr.Column(scale=4, min_width=500):
            input_interface = [
                gr.components.Number(label="How Old are you?"),
                gr.components.Radio(['male', 'female'], label='What is your Gender?'),
                gr.components.Dropdown(['High School', 'left', 'Undergrad', 'Grad', 'Associate Degree',
                                         'Doctorate'], label='What is your level of education?'),
                gr.components.Dropdown(['Widowed', 'Single', 'Married', 'Divorced', 'Separated'], label='Marital Status?'),
                gr.components.Dropdown([' White', ' Black', ' Asian or Pacific Islander',
                                        ' Amer Indian Aleut or Eskimo', ' Other'], label='Whats your race?'),
                gr.components.Dropdown([0, 2, 1], label='Whats your emploment status? (0 = Unemployed, 1 = Self-Employed, 2 = Employed)'),
                gr.components.Number(label='How much is your Wage per Hour? (0 - 10000)'),
                gr.components.Number(label='How many weeks have you worked in a year? (1 - 52)'),
                gr.components.Number(label='How many working weeks per year do you work?'),
                gr.components.Number(label='What is your Industry Code? (1 - 51)'),
                gr.components.Number(label='What is your occupation Code? (1 - 46)'),
                gr.components.Number(label='Number of persons working for employer? (1 - 7)'),
                gr.components.Number(label='Benefit? (1 - 3)'),
                gr.components.Dropdown([' Head of household', ' Single', ' Nonfiler', ' Joint both 65+',
                                        ' Joint one 65+ & one under 65', ' Joint one under 65 & one 65+'],label='Whats your tax status?'),
                gr.components.Number(label='What is your Gain'), 
                gr.components.Number(label='What is your Loss'),
                gr.components.Number(label='What is your Stock Status'),
                gr.components.Dropdown(['Native', ' Foreign born- Not a citizen of U S ',
                                         ' Foreign born- U S citizen by naturalization',
                                         ' Native- Born abroad of American Parent(s)',
                                         ' Native- Born in U S',' Native- Born in Puerto Rico or U S Outlying'], label='Whats is your Citizenshiip?'),
                gr.components.Radio([94,95], label='Whats your year of migration?'),
                gr.components.Number(label='Whats your Weight Of Instance?')    
                       
            ]

    with gr.Row():
        predict_btn = gr.Button('Predict') 
        
 

# Define the output interfaces
    output_interface = gr.Label(label="INCOME ABOVE LIMIT")

    predict_btn.click(fn=predict, inputs=input_interface, outputs=output_interface)


    app.launch(share=False)