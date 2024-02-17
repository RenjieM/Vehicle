from posixpath import dirname
import string
import pandas as pd 
import boto3
import numpy as np
from datetime import date, timedelta
import os
import matplotlib.pyplot as plt
import math
import re
from ray import method
import spacy
import torch

# Download files from S3
s3 = boto3.resource("s3")
def download(Key, Filename, Bucket = "twttr12138"):
    s3 = boto3.client("s3")
    s3.download_file(
        Bucket = "twttr12138", Key = Key, Filename = Filename
    )

def daterange(start_date, end_date):
    return int((end_date - start_date).days)

def down_to_local(start_date, end_date):
    filename = []
    if os.path.exists('/Users/Roger/Temp'):
        pass
    else:
        os.makedirs('/Users/Roger/Temp')
        print('"Temp" is created')

    for i in range(daterange(start_date, end_date)+1):
        d = str(start_date + timedelta(i))
        file_name = 'facebook_vehicles_' + d + '_moreinfo.csv'
        try: 
            download(file_name, '/Users/Roger/Temp/'+file_name)
            filename.append(file_name)
            print(f'got facebook_vehicles_{d}_moreinfo.csv')
        except:
            print(f'facebook_vehicles_{d}_moreinfo.csv is not existed')
            pass
    
    return filename

start_date = date(2022, 5, 1)
end_date = date(2022, 7, 13)
filename = down_to_local(start_date, end_date)

# get list of files in S3
def get_list():
    my_bucket = s3.Bucket('twttr12138')
    filename = []
    for items in my_bucket.objects.all():
        filename.append(items.key)
    return filename

filename = get_list()

# Concat all files together
def conc(filename):
    l = list()
    for i in range(len(filename)):
        file = pd.read_csv('/Users/Roger/Temp/' + filename[i])
        file.drop(columns='Cover_picture', inplace=True)
        l.append(file)
    
    df = pd.concat(l, axis=0, ignore_index=True)
    return df

nlp = spacy.load("Transformer/model-best")

def feature_extraction(df):
    # Clean description
    df.loc[:, 'Description'] = df.loc[:, 'Description'].str.replace(r"^About This Vehicle", '', regex=True)
    df.loc[:, 'Description'] = df.loc[:, 'Description'].str.replace(r"\n", ' ', regex=True)
    df.loc[:, 'Description'] = df.loc[:, 'Description'].str.replace(r",", '', regex=True)

    def get_unique(l):
        l_array = np.array(l)
        return list(np.unique(l_array))
    
    description = df.loc[:, 'Description']
    df_new = pd.DataFrame(columns = ['Fuel_type', 'Ex_color', 'In_color', 'City_mpg', 'Highway_mpg', 'Combined_mpg', 'Num_owner', 'Condition', 'Bad_words'])
    i = -1
    for text in description:
        if isinstance(text, str) == False:
            text = ''
        for doc in nlp.pipe([text], disable=["tagger"]):
            Fuel_type = [ent.text for ent in doc.ents if ent.label_ == 'FUEL_TYPE']
            Ex_color = [ent.text for ent in doc.ents if ent.label_ == 'EX_COLOR']
            In_color = [ent.text for ent in doc.ents if ent.label_ == 'IN_COLOR']
            City_mpg = [ent.text for ent in doc.ents if ent.label_ == 'CITY']
            Highway_mpg = [ent.text for ent in doc.ents if ent.label_ == 'HIGHWAY']
            Combined_mpg = [ent.text for ent in doc.ents if ent.label_ == 'COMBINED']
            Num_owner = [ent.text for ent in doc.ents if ent.label_ == 'NUM_ONWER']
            Condition = [ent.text for ent in doc.ents if ent.label_ == 'CONDITION_GOOD']
            Bad_words = [ent.text for ent in doc.ents if ent.label_ == 'BAD_WORDS']
            if Fuel_type == [] or len(get_unique(Fuel_type)) >= 2:
                Fuel_type = np.nan
            else:
                Fuel_type = Fuel_type[0]

            if Ex_color == [] or len(get_unique(Ex_color)) >= 2:
                Ex_color = np.nan
            else:
                Ex_color = Ex_color[0]

            if In_color == [] or len(get_unique(In_color)) >= 2:
                In_color = np.nan
            else:
                In_color = In_color[0]      

            if City_mpg == [] or len(get_unique(City_mpg)) >= 2:
                City_mpg = np.nan
            else:
                City_mpg = City_mpg[0]

            if Highway_mpg == [] or len(get_unique(Highway_mpg)) >= 2:
                Highway_mpg = np.nan
            else:
                Highway_mpg = Highway_mpg[0] 
                    
            if Combined_mpg == [] or len(get_unique(Combined_mpg)) >= 2:
                Combined_mpg = np.nan
            else:
                Combined_mpg = Combined_mpg[0]

            if Num_owner == [] or len(get_unique(Num_owner)) >= 2:
                Num_owner = np.nan
            else:
                Num_owner = Num_owner[0]  

            if Condition == []:
                Condition = 0
            else:
                Condition = 1

            if Bad_words == []:
                Bad_words = 0
            else:
                Bad_words = 1

            df_new = df_new.append({'Fuel_type': Fuel_type, 
                            'Ex_color': Ex_color, 
                            'In_color': In_color, 
                            'City_mpg': City_mpg, 
                            'Highway_mpg': Highway_mpg,
                            'Combined_mpg': Combined_mpg,
                            'Num_owner': Num_owner,
                            'Condition': Condition,
                            'Bad_words': Bad_words},
                            ignore_index=True)
            i += 1
            print(f'Number {i} is done')   

    df = pd.concat([df, df_new], axis = 1)
    
    return df
    
df_extract = feature_extraction(conc(filename))

df_extract.head()
df_extract.info()
df_extract.shape

def dataclean(df):
    # Perpare 
    exclude_words = ['bike', 'bicycle', 'boat', 'electricycle', 'polaris', 'yamaha', 'excavator', 'bulldozer', 'harley', 'ducati', 'ktm', 
            'kawasaki', 'triumph', 'aprilia', 'connondale', 'kona', 'colnago', 'bianchi', 'raleigh', 'cervelo', 'Orbea', 'Peace Sports']
    car_brands = [
    "Abarth", "Acura", "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "Bugatti", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Citroën",
    "Dacia", "Daewoo", "Daihatsu", "Dodge", "Donkervoort", "DS", "Eagle Talon", "Ferrari", "Fiat", "Fisker", "Ford", "GMC", "Honda", "Hummer", "Hyundai", "Infiniti", "Iveco", "Jaguar", "Jeep", "Kia", "KTM",
    "Lada", "Lamborghini", "Lancia", "Land Rover", "Landwind", "Lexus", "Lotus", "Lincoln", "Maserati", "Maybach", "Mazda", "McLaren", "Mercedes-Benz", "Mercury", "MG", "Mini", "Mitsubishi", "Morgan",
    "Nissan", "Opel", "Peugeot", "Porsche", "Pontiac", "Ram", "Renault", "Rolls-Royce", "Rover", "Saab", "Saturn", "Scion", "Seat", "Skoda", "Smart", "SsangYong", "Subaru", "Suzuki",
    "Tesla", "Toyota", "Volkswagen", "Volvo"
    ]

    brands = []
    for b in car_brands:
        brands.append(b.lower())

    def comb_words(words):
        words_comb = ''
        for word in words:
            if len(word.split(' ')) == 2:
                words_comb += word + '|'
                words_comb += word.split(' ')[0][0].upper() + word.split(' ')[0][1:] + ' ' + word.split(' ')[1][0].upper() + word.split(' ')[1][1:] + '|'
                words_comb += word.upper() + '|'   
            elif len(word.split('-')) == 2:
                words_comb += word + '|'
                words_comb += word.split('-')[0][0].upper() + word.split('-')[0][1:] + '-' + word.split('-')[1][0].upper() + word.split('-')[1][1:] + '|'
                words_comb += word.upper() + '|'  
            else:
                words_comb += word + '|'
                words_comb += word[0].upper() + word[1:] + '|'
                words_comb += word.upper() + '|'
        return words_comb[:-1]

    # Remove abmornal prices
    ind_abpri = list(df[(df.loc[:, 'Price'] < 2000) | (df.loc[:, 'Price'] > 100000)].index)
    df.drop(ind_abpri, axis=0, inplace=True)

    # Mileage
    # Remove dealership
    df = df[df.loc[:, 'Mileage'].apply(lambda x: str(x).split(' ')[-1]) != 'Dealership']
    # Set abnormal values to nan
    ind_abmil = list(df[(df.loc[:, 'Mileage'].apply(lambda x: str(x).split(' ')[-1]) != 'miles') | (df.loc[:, 'Mileage'].apply(lambda x: str(x).split(' ')[0]) == '1M')].index)
    df.loc[:, 'Mileage'][ind_abmil] = np.nan
    # Extract miles
    df.loc[:, 'Mileage_extract'] = df.loc[:, 'Mileage'].str.extract(r'(^\d*\.?\d+K)')
    df.loc[:, 'Mileage_extract'][df.loc[:, 'Mileage_extract'].notnull()] = df.loc[:, 'Mileage_extract'][df.loc[:, 'Mileage_extract'].notnull()].apply(lambda x: float(str(x)[:-1])*1000)
    # Miles without K
    ind_hmile = list(df[(df.loc[:, 'Mileage_extract'].isnull()) & (df.loc[:, 'Mileage'].notnull())].index)
    df.loc[:, 'Mileage_extract'][ind_hmile] = df.loc[:, 'Mileage'][ind_hmile].apply(lambda x: float(str(x).split(' ')[0]))
    # Log miles
    df.loc[:, 'Mileage_extract_log'] = df.loc[:, 'Mileage_extract'].apply(lambda x: np.log(x))

    # Location seperation
    df[['City', 'State']] = df.loc[:, 'Location'].str.split(",", expand=True)
    df.loc[:, 'City'] = df.loc[:, 'City'].apply(lambda x: x if x in df['City'].value_counts().index.tolist()[:50] else 'Other')

    # Title
    # Remove non car data
    index_noncar = list(
        df[(df.loc[:, 'Title'].str.contains(comb_words(exclude_words))) | (df.loc[:, 'Bad_words'] == 1)].index
    )
    df.drop(index_noncar, axis=0, inplace=True)
    # Extraction - years
    df.loc[:, 'Title'] = df.loc[:, 'Title'].str.replace(r"\'\"\(\)", '', regex=True)
    df.loc[:, 'Year_of_vehicle'] = df.loc[:, 'Title'].apply(lambda x: str(x).split(' ')[0])
    df.loc[:, 'Year_of_vehicle'] = pd.to_numeric(df.loc[:, 'Year_of_vehicle'], errors='coerce')
    df.loc[:, 'Title'] = df.loc[:, 'Title'].apply(lambda x: ' '.join(str(x).split(' ')[1:]).strip())
    # Extraction - brands
    df.loc[:, 'Brand_of_vehicle'] = df.loc[:, 'Title'].str.extract('('+comb_words(brands)+')')
    df.loc[:, 'Brand_of_vehicle'][df.loc[:, 'Brand_of_vehicle'].notnull()] = df.loc[:, 'Brand_of_vehicle'][df.loc[:, 'Brand_of_vehicle'].notnull()].apply(lambda x: str(x).upper())
    # Transformer features
    df.loc[:, 'City_mpg'] = pd.to_numeric(df.loc[:, 'City_mpg'], errors='coerce')
    df.loc[:, 'Highway_mpg'] = pd.to_numeric(df.loc[:, 'Highway_mpg'], errors='coerce')
    df.loc[:, 'Combined_mpg'] = pd.to_numeric(df.loc[:, 'Combined_mpg'], errors='coerce')
    df.loc[:, 'Condition'] = pd.to_numeric(df.loc[:, 'Condition'], errors='coerce')

    df.loc[df[df['Ex_color'] == 'Red ·'].index, 'Ex_color'] = 'Red'
    df.loc[df[(df['Num_owner'] == '2nd') | ((df['Num_owner'] == '2 owners'))].index, 'Num_owner'] = '2'
    df.loc[:, 'Ex_color'] = df.loc[:, 'Ex_color'].apply(lambda x: x if x in df['Ex_color'].value_counts().index.tolist()[:10] else 'Other')
    df.loc[:, 'In_color'] = df.loc[:, 'In_color'].apply(lambda x: x if x in df['In_color'].value_counts().index.tolist()[:10] else 'Other')
    # Get dummy
    dummy = pd.get_dummies(df[['City', 'State', 'Num_owner', 'In_color', 'Ex_color', 'Fuel_type', 'Brand_of_vehicle']])
    df = df.join(dummy)
    # Convert date to dense rank
    df['Date_rank'] = df.sort_values(['Collection_date']).rank(method='dense')['Collection_date']
    # Drop and reset index
    df.drop(columns=['Collection_date', 'Mileage_extract', 'Mileage', 'Location', 'Path', 'Bad_words', 'Description', 
        'Title', 'City', 'State', 'Num_owner', 'In_color', 'Ex_color', 'Fuel_type', 'Brand_of_vehicle'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
    
df = dataclean(df_extract)

df.head()
df.info()
df.shape

# Get train and val set
df_train, df_test = df.iloc[:math.floor(df.shape[0]*0.7), :], df.iloc[math.floor(df.shape[0]*0.7)+1:, :]
df_train.to_csv('facebook_train.csv', header=True, index=False)
df_test.to_csv('facebook_test.csv', header=True, index=False)
