
#######################################################################
# text Input data - modified 18 March 2023 
# Input data sample removed @
# training purpose 
# By : Mehdi (Zadeh1980mehdi@gmail.com)
#####################################################################


from pathlib import Path
import numpy as np 
import pandas as pd
import math
import os 
import time
import sys
import polars as pl 


import re
import string
from unicodedata import normalize


from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein,JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np 

from datetime import datetime
from unicodedata import normalize
#from google.colab import drive

import sparse_dot_topn.sparse_dot_topn as ct

import polars as pl
import pandas as pd
import os
import feather
import warnings
warnings.filterwarnings('ignore')
import string
import re
import statistics
import editdistance
import time


def currentDateString():
    current_date_formatted = datetime.today().strftime ('%d%m%Y')
    str_current_date = str(current_date_formatted)
    return str_current_date


def readBodyTxtFileReturnList(my_text_file="../data/input.txt"):
    with open(my_text_file, 'r') as f:
        text = f.read()

    long_text = re.findall(r'(?<=")([^"]+)(?=")|(?<=\\")(.*?)(?=\\")|(?<=\\\"\\")(.*?)(?=\\\"\\")', text, re.DOTALL)
    
    long_text = " ".join(str(x) for x in long_text)
    long_text = [t.replace('\n', ' ') for t in long_text]
    
    return long_text



def readIDTxtFileReturnList(my_text_file="../data/input.txt"):
    print("Read a text file to return list")
    # opening the file in read mode
    my_file = open(my_text_file, "r")

    # reading the file
    data = my_file.read()

    # replacing end splitting the text 
    # when newline ('\n') is seen.
    data_into_list = data.split(",")
    data_into_list = list(map(lambda s: s.strip(), data_into_list))
    #print(data_into_list)
    my_file.close()
    return data_into_list

def readTxtFile(my_text_file="../data/input.txt"):
    # Open a file: file
    file = open(my_text_file,mode='r')

    # read all lines at once
    all_of_it = file.read()

    # close the file
    file.close()
    return all_of_it

####################################################
def countChatWords(my_chat):
    my_text=my_chat
    test_string = my_text
    
    count_chat_words = len(re.findall(r'\w+', test_string))
    # total no of words
    print ("The number of words in string are : " + str(count_chat_words))
    return count_chat_words


def testAccess():
    print("connected to src.input_data")

# original - correct one 
def readcsvInputData(my_data_path='../input_data/', my_csv_file='96_sample.tsv', 
                     my_cols=['row','id','title','body'],
                     my_col_1='title', my_col_2='body', 
                     my_sep='\t', my_lineterminator='\n'):
    df = pd.read_csv(my_data_path + my_csv_file, sep=my_sep, lineterminator=my_lineterminator)
    df.columns =my_cols
    df[my_cols[2]]= df[my_cols[2]].str.strip()
    df[my_cols[3]]= df[my_cols[3]].str.strip()
    return df

# single Entry 
def createEmailDFFromText(my_string):
    # Dict object
    sampleEmail = {'row':[1],"id":["Unknown_ID"],
                   'title':["N/A"], "body":[my_string.strip()]}

    # Create DataFrame from dict
    df = pd.DataFrame.from_dict(sampleEmail,orient='columns')
    return df


#readtsvStreamlitInputData
def readtsvStreamlitInputData(my_tsv_file, 
                     my_cols=['row','id','title','body'],
                     my_col_1='title', my_col_2='body', 
                     my_sep='\t', my_lineterminator='\n'):
    df = pd.read_csv(my_tsv_file, sep=my_sep, lineterminator=my_lineterminator)
    df.columns =my_cols
    df[my_cols[2]]= df[my_cols[2]].str.strip()
    df[my_cols[3]]= df[my_cols[3]].str.strip()
    return df



#Append suffix/prefix to strings in list

def appendSuffixPrefixList(lst, len_str=20,  prefix="This is a sample email that I would like to share here. ", suffix=". this is a sample text added at the end of email paragraph. "):
    suf_res = list(map(lambda x: x+ suffix if len(re.findall(r'\w+', x)) <len_str else x , lst))
   
 
    return suf_res
 
def transactionIDEmailClassification(my_input_number=20230318):
    my_dict_input_data={
        #ID in system
        "trans_id_20230318":[123,456 ],
        
        
        "trans_id_20230303":[345,5677]
    }
    
    my_email_ID=my_dict_input_data.get(f'trans_id_{my_input_number}')
    
    return my_email_ID
    
def convertListTostring(my_numeric_list):
    list_string = list(map(str, my_numeric_list))
    return list_string

def inputDataEmailClassification(my_input_number=20230318):
    my_dict_input_data={
    
        "my_input_20230318":[""" this is first  """, """this is second """],
        
        "my_input_20230303":["this is a sample", "this is another sample" ]     
     }

    
    my_email_body=my_dict_input_data.get(f'my_input_{my_input_number}')
    
    return my_email_body


def inputDataChat(my_chat_number=6):
    
    my_dict_input_data={
     "my_chat_27":"""
     This are bad. I hates tis manner and this prodct is disgutin . i shuld tak with your manger. we deserve the better services. thee smelt of fliwers bring back mmemories.
     """,
        "my_chat_26":"""
     i lik the flowr and song ob birde . i realy luv sae side and sun over sand whil i wold like to enjoyt foggy clmiate. 
     """,
     
     
        
                
        "my_chat_17":"Bonnie: I just to introduce ourselves to like I said, get, you know, get aligned on kinda like what, you know, what, who we are and it's pretty much about.",

        "my_chat_16":"I hop that you're happy now. i alwys try to usee contractioning ov these vord to be clairfied",
        "my_chat_15":"The smelt of fliwers bring back memories.",
        
        
        "my_chat_14":"you're happy now",
        "my_chat_13":"u r hateful person" ,
        "my_chat_12":"I hate you" ,
    
        
        "my_chat_11":"I do not know" ,
    
        "my_chat_10":"The second amendment that sowed the seeds of discord affirmed that French is the common language of Quebec, which the Liberals could not accept. Ashton and Ontario Conservative MP Marilyn Gladu both voted against it and did not explain their decisions. I don't see why this word is necessary. It's the official language, yes, French. But to add the word 'common' could imply obligations according to someone's interpretation in the future, pleaded Marc Garneau, the Liberal MP for Notre-Dame-de-Grâce-Westmount. His colleague, MP Patricia Lattanzio of Saint-Léonard-Saint-Michel, then stated in English that this new notion is not defined anywhere. However, an expert from the Department of Canadian Heritage told her that this term is well and truly defined in the Charter of the French Language. The common language means the language of convergence, the language that brings everyone together, said Beaulieu, the Bloc MP who tabled the amendment requested by the Quebec government, to reporters. If we want to have a coherent society, (we) have to be able to talk to each other at some point. Beaulieu said the Charter of the French Language does not take anything away from anglophones and said he has the impression that those who object to it do not accept (...) that Quebec society integrates immigrants. The Standing Committee on Official Languages has scheduled up to six more meetings for clause-by-clause consideration of the bill. One of the highlights of the bill's study is expected to come in the next few weeks when MPs vote on an amendment to make private companies under federal jurisdiction subject to the Charter of the French Language, as called for by the Quebec government. The parties have already weighed in: the Liberals are against it, but the Conservatives, the Bloc and the NDP are in favour, which should allow the amendment to pass, barring a blowout. Bill C-13 establishes a new right to work and be served in French in Quebec and in regions with a strong francophone presence in other provinces in private companies under federal jurisdiction, such as banks, airlines or railways.", 
    
    "my_chat_9":"The new design is awful! . I really do not like your help in this matter. You are nut. You are completely wasting my money and time.",
    
    "my_chat_8":"the company stops all operations and goes completely out of business. \
    Company should file bankruptcy as they're unable to pay their debts. Simply they should liquidate the company's assets. ",
  
    "my_chat_7":"The new design is perfect! . I really like your help in this matter. You are courageous.\
    you are empathetic ,  intuitive ,  creative ,  passionate ,  a life-long learner .",
    
    "my_chat_6":"super excited ! and awsome . you are the best. Your product is the best to invest money ",
    
    "my_chat_5":"Wisdom of Great Investors ",
    
  
    "my_chat_4":"the company has profitable businesses in different sectors like as energy and steel industry. Stocks rallied and the currency gained again. ",


    "my_chat_3":"Stocks rallied and the British pound gained.",

    "my_chat_2":"Bids or offers include at least 1,000 shares and the value of the shares must correspond to at least EUR 4,000.",
    
  }
    
    my_chat=my_dict_input_data.get(f'my_chat_{my_chat_number}')
    
    return my_chat





###########################################################


# Read and write Files with Pandas and Polar in CSV and Feather format  - 18 March 2023
#########################################################################
        
#polar Settings ! 

def lazyPreprocessingWithPolars(my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['row_ID','text_ID','col1_name','col2_name']):    
   
    df_read_scan_csv=pl.scan_csv(my_csv_data.format(my_file_format), ignore_errors = True, dtypes={'row_ID': pl.UInt32,'text_ID': pl.Utf8 , 'col1_name':pl.Utf8, 'col2_name':pl.Utf8 })      #fetch(1042131)
    df_read_scan_csv= df_read_scan_csv.select([pl.col(my_cols_name[0]), pl.col(my_cols_name[1]), pl.col(my_cols_name[2]),pl.col(my_cols_name[3])])
    df_read_scan_csv=preProcessingFunction(df_read_scan_csv)
    df_read_scan_csv=df_read_scan_csv.collect()
    return df_read_scan_csv


def lazyPreprocessingWithPolars02(my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['text_ID','col1_name','col2_name']):    
   
    df_read_scan_csv=pl.scan_csv(my_csv_data.format(my_file_format), ignore_errors = True, dtypes={'text_ID': pl.Utf8 , 'col1_name':pl.Utf8, 'col2_name':pl.Utf8 })      #fetch(1042131)
    df_read_scan_csv= df_read_scan_csv.select([pl.col(my_cols_name[0]), pl.col(my_cols_name[1]), pl.col(my_cols_name[2])])
    #df_read_scan_csv=dataframe_NaNRemoval_polar(df_read_scan_csv)
    df_read_scan_csv=preProcessingFunction(df_read_scan_csv)
    df_read_scan_csv=df_read_scan_csv.sort([pl.col("sorted_clean_col2_name")],reverse=True, nulls_last = True)
    
    df_read_scan_csv=df_read_scan_csv.collect()
    return df_read_scan_csv



def readLazyScanRawCSVWithPolars(my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['text_ID','col1_name','col2_name']):    
    
    df_read_scan_csv=pl.scan_csv(my_csv_data.format(my_file_format), ignore_errors = True, dtypes={'text_ID': pl.Utf8 , 'col1_name':pl.Utf8, 'col2_name':pl.Utf8 })      #fetch(1042131)
    df_read_scan_csv= df_read_scan_csv.select([pl.col(my_cols_name[0]), pl.col(my_cols_name[1]), pl.col(my_cols_name[2])])
    df_read_scan_csv=df_read_scan_csv.collect()
    return df_read_scan_csv
    


def readLazyScanRawCSVWithPolars(my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['text_ID','my_col1_name','sorted_clean_col2_name']):    
    df_read_scan_csv=(pl.scan_csv(my_csv_data.format(my_file_format), with_column_names=my_cols_name,dtypes={'text_ID': pl.UInt32 , 'col1_name':pl.Utf8, 'col2_name':pl.Utf8 }).collect())      #fetch(1042131)
    return df_read_scan_csv
    
def readLazyScanRawFeatherWithPolars(my_feather_col2_name='../data/data_all/*.{}',my_file_format='ftr',my_cols_name=['text_ID','my_col1_name','sorted_clean_col2_name']):
    df_read_scan_ftr=pl.scan_ipc(my_feather_col2_name.format(my_file_format)).collect()      #fetch(1042131)
    return df_read_scan_ftr


def readCSVWithPolars(my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['text_ID','col1_name','col2_name']):
    df=(pl.read_csv(my_csv_data.format(my_file_format),columns=my_cols_name,dtypes={'text_ID': pl.UInt32 , 'col1_name':pl.Utf8, 'col2_name':pl.Utf8 }))
    df_read=df.select([pl.col('text_ID'),pl.col('col1_name').apply(textNormalize),pl.col('col2_name').apply(textNormalize)])
 
    return df_read



def readCSVWithPolarsWithoutPreprocessing(str_current_date,my_csv_data='../data/data_all_clean/{}_tempcleanData.csv',my_cols_name=['text_ID','my_col1_name','sorted_clean_col2_name']):
    df=(pl.read_csv(my_csv_data.format(str_current_date),columns=my_cols_name,dtypes={'text_ID': pl.UInt32 , 'col1_name':pl.Utf8, 'col2_name':pl.Utf8 }))
    df_read=df.select([pl.all()])

    return df_read

   
    
def writeCSVWithPolars(my_dataframe,str_current_date, my_file_path="../data/data_all_clean/{}_tempcleanData.csv"):
    my_pingInfoFilePath =my_file_path.format(str_current_date)
    my_dataframe.write_csv(my_pingInfoFilePath, sep=",")
    return my_pingInfoFilePath
    
  
 
# https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.DataFrame.write_ipc.html#polars.DataFrame.write_ipc    
def writeFeatherWithPolars(my_dataframe,str_current_date, my_file_path="../data/data_all_clean/{}_tempcleanData.ftr"):
    
    my_pingInfoFilePath =my_file_path.format(str_current_date)
    my_dataframe.write_ipc(my_pingInfoFilePath)
       
    return my_pingInfoFilePath
    


def readFeatherWithPolars(str_current_date, my_file_path="../data/data_all_clean/{}_tempcleanData.ftr"):
    my_pingInfoFilePath =my_file_path.format(str_current_date)
    df_read=pl.read_ipc(my_pingInfoFilePath)
    return df_read
        
    
    
def writeFeatherData(my_dataframe,str_current_date, my_file_path="../data/data_all_clean/{}_tempcleanData.ftr"):
    my_pingInfoFilePath =my_file_path.format(str_current_date)
    #df_read=my_dataframe.sort_values(by = ['my_col1_name'], ascending = [True], na_position = 'first')
    df_read.to_feather(my_pingInfoFilePath)
    return my_pingInfoFilePath

def readFeatherData(my_pingInfoFilePath, my_columns=None, my_use_threads=True):
    ############################################################################
    # Reading Feather file ! Check archive to be sure right and correct format ( Date )

    df_concat_feather = pd.read_feather(my_pingInfoFilePath, columns=my_columns, use_threads=my_use_threads)
    #print(type(df_concat_feather))

    df_concat_feather=df_concat_feather.astype(str)
    #print(df_read.dtypes)

    df_read_feather=df_concat_feather
    
    return df_read_feather
    
    
# with Pandas 
# concat different CSV files into one : 

# https://medium.com/@stella96joshua/how-to-combine-multiple-csv-files-using-python-for-your-analysis-a88017c6ff9e
#list all csv files only

def readCSVWithPandas(my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['text_ID','col1_name','col2_name']): 
    csv_files = glob.glob(my_csv_data.format(my_file_format))
    df_concat = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)
    df_concat=df_concat[my_cols_name]
    df_read=df_concat
    df_read=df_read.astype(str)
    ####################################################################################
    df_read['col2_name'] = df_read['col2_name'].apply(lambda x: x.lower())
    df_read.col1_name=df_read.col1_name.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_read.col2_name=df_read.col2_name.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    #######################################################
    return df_read


def readCSVWithPandasWithoutPreprocessing(my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['text_ID','col1_name','col2_name']): 
    csv_files = glob.glob(my_csv_data.format(my_file_format))
    df_concat = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)
    df_concat=df_concat[my_cols_name]
    df_read=df_concat
    return df_read









###################################################################################





def readFolderCSVWithPolars(my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['Customer_Id','my_col1_name','my_alt_Id']):
    df=(pl.read_csv(my_csv_data.format(my_file_format),columns=my_cols_name,dtypes={my_cols_name[0]: pl.UInt32 , my_cols_name[1]:pl.Utf8, my_cols_name[2]:pl.Utf8 }))
    df_read=df.select([pl.col(my_cols_name[0]),pl.col(my_cols_name[1]).apply(textNormalize),pl.col(my_cols_name[2]).apply(textNormalize)])
 
    return df_read


def lazycol1_nameSimilarityWithPolars2(my_row_num=9000,my_csv_data='../data/data_all/*.{}',my_file_format='csv',
                                      my_cols_name=['row_ID','text_ID','col1_name','my_alt_Id','col2_name']):    
   
    df_read_scan_csv=pl.scan_csv(my_csv_data.format(my_file_format), ignore_errors = True, 
                                 dtypes={my_cols_name[0]: pl.UInt32 , my_cols_name[1]:pl.Utf8, my_cols_name[2]:pl.Utf8, my_cols_name[3]:pl.Utf8,my_cols_name[4]:pl.Utf8 })      #fetch(1042131)
    df_read_scan_csv= df_read_scan_csv.select([pl.col(my_cols_name[0]), pl.col(my_cols_name[1]), pl.col(my_cols_name[2]), pl.col(my_cols_name[3]), pl.col(my_cols_name[4])])
    df_read_scan_csv=df_read_scan_csv.sort([pl.col("my_col1_name")],reverse=True, nulls_last = True)
    
    df_read_scan_csv=df_read_scan_csv.with_row_count(name="idx", offset=0)
    ##### limitted rows ! 
    df_read_scan_csv = df_read_scan_csv.filter(pl.col("idx") <= my_row_num)
    
    df_read_scan_csv= preProcessingcol1_nameFunction(df_read_scan_csv)
    df_read_scan_csv= combinationProcessingFunction(df_read_scan_csv)
    df_read_scan_csv=probabilisticSimilarityProcessingFunction(df_read_scan_csv)
    
    df_read_scan_csv= df_read_scan_csv.collect()
    return df_read_scan_csv

#row_ID	text_ID	col1_name	my_alt_Id	col2_name	col2_alt_name
def lazycol1_nameJarowinklerSimilarityWithPolars_barCode(my_row_num=9000,my_csv_data='../data/data_all/*.{}',my_file_format='csv',
                                                my_cols_name=['row_ID', 'text_ID', 'col1_name', 'my_alt_Id', 'col2_name', 'col2_alt_name']):    
   
    df_read_scan_csv_2=pl.scan_csv(my_csv_data.format(my_file_format), ignore_errors = True,
                                   dtypes={my_cols_name[0]: pl.UInt32 , my_cols_name[1]:pl.Utf8, my_cols_name[2]:pl.Utf8 ,
                                           my_cols_name[3]:pl.Utf8,my_cols_name[4]:pl.Utf8,
                                   my_cols_name[5]:pl.Utf8})      
    df_read_scan_csv_2= df_read_scan_csv_2.select([pl.col(my_cols_name[0]), pl.col(my_cols_name[1]), 
                                                   pl.col(my_cols_name[2]),pl.col(my_cols_name[3]),pl.col(my_cols_name[4])])
    df_read_scan_csv_2=df_read_scan_csv_2.sort([pl.col("col1_name")],reverse=True, nulls_last = True)
    
    
    
    df_read_scan_csv_2=df_read_scan_csv_2.with_row_count(name="idx", offset=0)
    
    ##### limitted rows ! 
    df_read_scan_csv_2 = df_read_scan_csv_2.filter(pl.col("idx") <= my_row_num)
    
    df_read_scan_csv_2= preProcessingcol1_nameSameBarCodeFunction(df_read_scan_csv_2)
    #df_read_scan_csv_2=df_read_scan_csv_2.unique(subset=["my_col1_name_clean"])
    df_read_scan_csv_2= combinationProcessingFunction(df_read_scan_csv_2)
    df_read_scan_csv_2=probabilisticJarowinklerSimilarityProcessingFunction(df_read_scan_csv_2)
    
    df_read_scan_csv_2= df_read_scan_csv_2.collect()
    return df_read_scan_csv_2

def lazycol1_nameJarowinklerSimilarityWithPolars(my_row_num=9000,my_csv_data='../data/data_all/*.{}',my_file_format='csv',
                                                my_cols_name=['row_ID', 'client_name', 'Customer_Id', 'data_code', 'my_col1_name', 'col2_alt_name']):    
   
    df_read_scan_csv_2=pl.scan_csv(my_csv_data.format(my_file_format), ignore_errors = True,dtypes={my_cols_name[0]: pl.UInt32 , my_cols_name[1]:pl.Utf8, my_cols_name[2]:pl.Utf8 ,my_cols_name[3]:pl.Utf8,my_cols_name[4]:pl.Utf8,
                                   my_cols_name[5]:pl.Utf8})      #fetch(1042131)
    df_read_scan_csv_2= df_read_scan_csv_2.select([pl.col(my_cols_name[0]), pl.col(my_cols_name[1]), pl.col(my_cols_name[2]),pl.col(my_cols_name[3]),pl.col(my_cols_name[4])])
    df_read_scan_csv_2=df_read_scan_csv_2.sort([pl.col("my_col1_name")],reverse=True, nulls_last = True)
    
    
    
    df_read_scan_csv_2=df_read_scan_csv_2.with_row_count(name="idx", offset=0)
    
    ##### limitted rows ! 
    df_read_scan_csv_2 = df_read_scan_csv_2.filter(pl.col("idx") <= my_row_num)
    
    df_read_scan_csv_2= preProcessingcol1_nameFunction(df_read_scan_csv_2)
    
    df_read_scan_csv_2= combinationProcessingFunction(df_read_scan_csv_2)
    df_read_scan_csv_2=probabilisticJarowinklerSimilarityProcessingFunction(df_read_scan_csv_2)
    
    df_read_scan_csv_2= df_read_scan_csv_2.collect()
    return df_read_scan_csv_2


def lazycol1_nameSimilarityWithPolars(my_row_num=9000,my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['text_ID','col1_name','my_alt_Id','col2_name']):    
   
    df_read_scan_csv_2=pl.scan_csv(my_csv_data.format(my_file_format), ignore_errors = True,dtypes={my_cols_name[0]: pl.UInt32 , my_cols_name[1]:pl.Utf8, my_cols_name[2]:pl.Utf8 ,my_cols_name[3]:pl.Utf8})      #fetch(1042131)
    df_read_scan_csv_2= df_read_scan_csv_2.select([pl.col(my_cols_name[0]), pl.col(my_cols_name[1]), pl.col(my_cols_name[2]),pl.col(my_cols_name[3])])
    df_read_scan_csv_2=df_read_scan_csv_2.sort([pl.col("col1_name")],reverse=True, nulls_last = True)
    
    
    
    df_read_scan_csv_2=df_read_scan_csv_2.with_row_count(name="idx", offset=0)
    
    ##### limitted rows ! 
    df_read_scan_csv_2 = df_read_scan_csv_2.filter(pl.col("idx") <= my_row_num)
    
    df_read_scan_csv_2= preProcessingcol1_nameFunction(df_read_scan_csv_2)
    #df_read_scan_csv_2=df_read_scan_csv_2.unique(subset=["my_col1_name_clean"])
    df_read_scan_csv_2= combinationProcessingFunction(df_read_scan_csv_2)
    df_read_scan_csv_2=probabilisticSimilarityProcessingFunction(df_read_scan_csv_2)
    
    df_read_scan_csv_2= df_read_scan_csv_2.collect()
    return df_read_scan_csv_2



def lazycol1_nameJarowinklerSimilarityWithPolars2(my_row_num=9000,my_csv_data='../data/data_all/*.{}',my_file_format='csv',my_cols_name=['Customer_Id','my_col1_name','my_alt_Id']):    
   
    df_read_scan_csv_2=pl.scan_csv(my_csv_data.format(my_file_format), ignore_errors = True, dtypes={my_cols_name[0]: pl.UInt32 , my_cols_name[1]:pl.Utf8, my_cols_name[2]:pl.Utf8 })      #fetch(1042131)
    df_read_scan_csv_2= df_read_scan_csv_2.select([pl.col(my_cols_name[0]), pl.col(my_cols_name[1]), pl.col(my_cols_name[2])])
    df_read_scan_csv_2=df_read_scan_csv_2.sort([pl.col("my_col1_name")],reverse=True, nulls_last = True)
    
    
    df_read_scan_csv_2=df_read_scan_csv_2.with_row_count(name="idx", offset=0)
    
    ##### limitted rows ! 
    df_read_scan_csv_2 = df_read_scan_csv_2.filter(pl.col("idx") <= my_row_num)
    
    df_read_scan_csv_2= preProcessingcol1_nameFunction(df_read_scan_csv_2)
    df_read_scan_csv_2= combinationProcessingFunction(df_read_scan_csv_2)
    df_read_scan_csv_2=probabilisticJarowinklerSimilarityProcessingFunction(df_read_scan_csv_2)
    
    df_read_scan_csv_2= df_read_scan_csv_2.collect()
    return df_read_scan_csv_2
