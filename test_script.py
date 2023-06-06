import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy

import pickle
import itertools
import collections
import matplotlib.pyplot as plt
import scipy
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from sklearn.cluster import DBSCAN

from gensim.models import Word2Vec

import re
import string

import warnings

# Filter out warning messages
warnings.filterwarnings('ignore')

from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 

# stop_words=set(stopwords.words('english'))

ps = PorterStemmer()


necessary_words = ['eftn', 'ft', 'bkash', 'nogod', 'rtgs', 'pos',
                   'cib', 'paywell', 'challan', 'npsb', 'dps', 'atm', 'trf', 'sonod',
                   'smart','app','smart app']

unnecessary_words=['month', 'bari', 'china', 'german', 'zero', 'first',
                   'today', 'daily', 'mim', 'nova', 'bas', 'year', 'week', 
                   'martin', 'two', 'june', 'monthly', 'khan', 'twelve', 'eighteen',
                   'quarterly', 'bakula', 'weekly', 'hour', 'august', 'annual', 'patwari',
                   'gore', 'fourteen', 'al', 'second', 'yesterday', 'shanghai', 'kokan', 
                   'noon', 'fifteen', 'japan', 'sec', 'abu','para', 'weekly', 'ba', 'saddik', 
                   'oct', 'eighteen', 'ka', 'c', 'dor', 'na', 'gate', 'point', 'dal', 'feb', 
                   'german', 'december', 'id', 'begum', 'ink', 'zero', 'mim', 'bas', 'tapu', 
                   'orient', 'mo', 'abu', 'brother', 'tala', 'daud', 'new', 'type', 'title', 
                   'outlet', 'jan', 'name', 'october', 'th', 'kokan', 'aug', 'currier', 
                   'doll', 'u', 'august', 'service', 'tara', 'nov', 'tony', 'mar', 'bin',
                   'february', 'k', 'china', 'martin', 'st', 'jowel', 'x', 'sha', 'dada',
                   'today', 'noon', 'ad', 'yesterday', 'ae', 'amir', 'sweety', 'mother', 'mu',
                   'hanif', 'mullah', 'july', 'first', 'nova', 'japan', 'ge', 'rocky', 
                   'rana', 'pu', 'annual', 'second', 'omer', 'bibi', 'fakir', 'southeast',
                   'da', 'cotton', 'apr', 'coxs', 'al', 'jun', 'sima', 'e', 'bakula', 'dola',
                   'pur', 'quarterly', 'amenia', 'shanghai', 'shahin', 'babu', 'ar', 'bu',
                   'tania', 'p', 'm', 'june', 'patwari', 'barman', 'dey', 'sir', 'daily',
                   'i', 'khan', 'raj', 'rani', 'week', 'boro', 'momo', 'sep', 'b', 'pally', 
                   'sultana', 'fourteen', 'link', 'palli', 'ghat', 'chad', 'l', 'das', 'dec', 
                   'mir', 'march', 'hour', 'sri', 'kaka', 'september', 'r', 'auto', 'nandi',
                   'month', 'amt', 'kazi', 'year', 'puja', 'hasan', 'november', 'amin', 'may', 
                   'date', 'monthly', 'razor', 'sheik', 'road', 'gore', 'january', 'bari', 
                   'nid', 'say', 'april', 'total', 'twelve', 'shah', 'sec', 'fifteen', 'doc',
                   'son', 'maria', 'jul', 'two','ac','mm','agri','inter','polli',
                   'bidyut','islam','islami','hossain','akter','md','ltd','salari','salrari','salrary','mia',
                   'miah','ali','agrani','rahman','pubali','shonali','sonali','bangladesh','saiful','arif',
                   'thousand','taka','lakh','uddin','udc','ab','dol','doll','somiti','purush',
                   'asol','uttor','rd','rlp','polli bidyut','janata','rupali ','m','store','sandwip',
                   'enterprise','sadar','polly','sme','senderamlabo','cashseddate','teota'
                   'raigor','mohila','suraighat','bazar','shohor','uposhohor']
pre_pos= ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid',
          'among', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath',
          'beside', 'between', 'beyond', 'but', 'by', 'concerning', 'considering', 
          'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 
          'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past',
          'regarding', 'round', 'since', 'through', 'throughout', 'to', 'toward', 'under',
          'underneath', 'until', 'unto', 'up', 'upon', 'with', 'within', 'without']

def drop_duplicates(dataset):
    cols=['NARATION']
    sub_data = dataset.drop_duplicates(subset=cols)
    return sub_data
# sub_data.isnull().sum()

def drop_null(dataset):
    dataset.dropna(subset=['NARATION'],inplace=True)
    return dataset
# sub_data.isnull().sum()

def drop_empty_tokens(dataset):
    # Find rows with empty token lists
    empty_token_rows = dataset['tokens'].apply(lambda tokens: len(tokens) == 0)

    # Drop rows with empty token lists
    dataset = dataset[~empty_token_rows]

    return dataset

stop_words = set(stopwords.words("english")+unnecessary_words+pre_pos)

def name_removal(text):
  doc=nlp(text)
  sz=len(doc.ents)
#   print(doc.ents)
  modified_string = ""
  if(sz>0):
    for entity in doc.ents:
        ent=entity.text
#         print(ent)
        if(ent.lower() in necessary_words):
#             print(ent)
            continue
        elif ((entity.label_ == "PERSON" or entity.label_ == "NORP" or entity.label_ == "LOC" or entity.label_ == "GPE")):
            modified_string = text.replace(ent, "")
        else:
            modified_string=text
#         modified_string = text.replace(ent, "")
  else:
    modified_string=text
  return modified_string


def preprocess_text(text):
    # Remove punctuation
    text=str(text)
    text=name_removal(text)
    text=text.lower()
#     text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^A-Za-z\s]", "", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
#     stop_words = set(stopwords.words("english"))
#     lematizer.lemmatize(w) for w in word_tokenize(text) if not w in stop_words
    tokens = [lematizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]
#     tokens=[ps.stem(w) for w in tokens]
    return tokens

def tokenize(dataset):
    tokens_list=list()
    for index, row in dataset.iterrows():
#         p=model.predict(row['avg_vec'].reshape(1, -1))
        t=preprocess_text(row['NARATION'])
        tokens_list.append(t)
        
#         print(type(t))
        
#         p=preprocess_text(row['NARATION'])
        
#     print(len(tokens_list))
#     print(dataset.shape) 
    dataset['tokens']=tokens_list
   
    return dataset
    

def most_common_tokens_bar(dataset):
    lemmatized_tokens = list(dataset['tokens'])
    token_list = list(itertools.chain(*lemmatized_tokens))
    counts_no = collections.Counter(token_list)
    clean_tweets = pd.DataFrame(counts_no.most_common (30), columns=['words', 'count'])
    fig, ax = plt.subplots(figsize=(8, 8))
    clean_tweets.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color='blue')
    ax.set_title("Most Frequently used words in Narrations")

    plt.show()

def vectorizing_model(token_list):
    model=Word2Vec(token_list,min_count=1)
    return model



def avg_vec(vectors):
    s_vec=np.zeros(100)
    cnt=0
    for v in vectors:
        cnt+=1
        s_vec+=v
    s_res=s_vec/cnt
    return s_res


def avg_vectorized(dataset):
    # tokenlist
    token_list=dataset['tokens'].tolist()
#     make model
    model=vectorizing_model(token_list)
#     vectorized field
    dataset['vectorized'] = dataset['tokens'].apply(lambda x: [model.wv.get_vector(word ,norm=True) for word in x])
    
    dataset['avg_vec']= dataset['vectorized'].apply(avg_vec)
    
    dataset.drop('vectorized',axis=1)

    # Drop rows with empty token lists
#     dataset = dataset[~empty_token_rows]

    return dataset

def predict(dataset,model):
    for index, row in dataset.iterrows():
        p=model.predict(row['avg_vec'].reshape(1, -1))
        dataset.at[index, 'pred'] = p[0]
   
    return dataset


def text_predict(text,model):
    d=list()
    d.append(text)
    df_dummy=pd.DataFrame({'NARATION':d})
    df_dummy=tokenize(df_dummy)
    df_dummy=avg_vectorized(df_dummy)
    df_dummy=predict(df_dummy,model)
    return df_dummy


def plot(dataset,column):
    

    minibatch_groups = dataset.groupby(column)
    num_clusters = len(minibatch_groups)
    num_cols = 3  # Number of columns to display

    num_rows = num_clusters // num_cols
    if num_clusters % num_cols != 0:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12))

    for i, (cluster, group) in enumerate(minibatch_groups):
        tokens = []
        for token_list in group['tokens']:
            tokens.extend(token_list)
        common_tokens = pd.Series(tokens).value_counts().head(10)
        common_tokens = common_tokens.iloc[::-1]
        # Plot the most common words in a subplot
        ax = axes[i // num_cols, i % num_cols]
        common_tokens.plot(kind='barh', ax=ax)
        ax.set_title(f"Cluster {cluster}: Most Common Words")
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def show_dendogram(dataset):
    column=dataset['avg_vec']
    X = np.stack(column.values)  # Convert column to a 2D np array

    # Define a distance function for comparing np arrays
    def array_distance(a, b):
        return np.linalg.norm(a - b)  # Euclidean distance

    # Compute the distance matrix using the defined distance function
    distance_matrix = np.zeros((X.shape[0], X.shape[0]))  # Initialize the distance matrix
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            distance_matrix[i, j] = array_distance(X[i], X[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    # Compute linkage matrix using the distance matrix
    Z = linkage(distance_matrix, method='average')

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
    #     show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        ) 
    plt.show()

def dbscan_prediction(dataset):
    test_avg_vec_list=dataset['avg_vec'].tolist()
    dbscan = DBSCAN(eps=.55, min_samples=1).fit_predict(test_avg_vec_list)
    dataset['d_c']=dbscan
    print(len(np.unique(dbscan)))
    return dataset

def hierarichal_cluster(dataset,mx_d): 
    max_d = mx_d
    clusters = fcluster(Z, max_d, criterion='distance')
    dataset['h_c']=clusters
    return dataset



def get_model(file_path):
    with open(file_path, "rb") as file:
    model_tst = pickle.load(file)
    return model_tst


# main function
if __name__ == "__main__":
    # read data
    dataset='dataset path'
    model=get_model('model path')
    dataset=tokenize(dataset)
    dataset=avg_vectorized(dataset)
    dataset=predict(dataset,model) #gets a 'pred' column with prediction
    dataset=dbscan_prediction(dataset) #gets a 'd_c' column with cluster number
    show_dendogram(dataset)
    dataset=hierarichal_cluster(dataset,1.5) # 1.5 is the max distance can be tuned for other dataset based on dendogram
                                            # gets a 'h_c' column with cluster number

    dataset.to_csv('result.csv')




