import pandas as pd
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from tqdm import tqdm

df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sample_submission.csv')

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

pattern = 'NP: {<DT>?<JJ>*<NN>}'
df_test['tag'] = 'O'

all_tags = []

for i in tqdm(range(df_test.shape[0])):
    #print('We are at row {} out of {}'.format(i, df_test.shape[0]))

    ex = str(df_test['Word'].iloc[i])

    sent = preprocess(ex)
    #print(sent)

    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    #print(cs)


    #cs = df_train['Word'].values
    iob_tagged = tree2conlltags(cs)
    #pprint(iob_tagged)

    all_tags.append(iob_tagged[0][2])


df_submission['tag'] = all_tags

df_submission['tag'][df_submission['tag']=='B-NP'] = 'B-indications'

df_submission.to_csv('submission_nltk1.csv', index = False)

nltk_data = pd.read_csv('submission_nltk1.csv')
nltk_data2 = nltk_data.copy()

for i in range(1, nltk_data2.shape[0]):
    if (nltk_data2['tag'].iloc[i]=='B-indications') and (nltk_data2['tag'].iloc[i-1]=='B-indications'):
        nltk_data2['tag'].iloc[i-1] = 'I-indications'

nltk_data2.to_csv('nltk_2.csv', index = False)