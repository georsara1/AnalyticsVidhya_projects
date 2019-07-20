import pandas as pd
from tqdm import tqdm

spacy_data = pd.read_csv('submission_spacy1.csv')
nltk_data = pd.read_csv('submission_nltk1.csv')

spacy_data2= spacy_data.copy()

f = []
for i in tqdm(range(nltk_data.shape[0])):
    if (nltk_data['tag'].loc[i] =='B-indications') and (spacy_data['tag'].loc[i] =='B-indications'):
        f.append('B-indications')
    else:
        f.append('O')

spacy_data['tag'] = f

spacy_data.to_csv('submission_ensemble1.csv', index = False)

nltk_data2 = nltk_data.copy()

for i in range(1, nltk_data2.shape[0]):
    if (nltk_data2['tag'].iloc[i]=='B-indications') and (nltk_data2['tag'].iloc[i-1]=='B-indications'):
        nltk_data2['tag'].iloc[i-1] = 'I-indications'

nltk_data2.to_csv('nltk_2.csv', index = False)