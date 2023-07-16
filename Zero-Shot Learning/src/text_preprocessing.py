# text preprocessing - modified 18 March 2023 -Version 02
#training Purpose ! 
# By : Mehdi (zadeh1980mehdi@gmail.com)

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

# Grammer Checking ! 
#from gingerit.gingerit import GingerIt

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
import contractions
import textwrap
import cloudscraper

import multiprocessing
from multiprocessing import Pool


def StringLabelToNumbericScoreLabel(my_list):
    res=my_list
    transformed_lables = list(map(lambda x: -1.0  if x =='Very Negative' else -0.5 if x =='Negative' else 0.0 if x =='Mixed' else 0.5 if x =='Positive' else 1.0 , res))
    final_sentiment_score='%.3f'%(sum(transformed_lables)/len(res))
    return final_sentiment_score

def StringLabelToNumbericScoreLabel02(my_list):
    new_list = []
    
    for x in my_list:
        if x=='Very Negative':
            new_list.append(-1.0)
        elif x =='Negative':
            new_list.append(-0.5)
        elif x =='Mixed':
            new_list.append(0.0)
        elif x =='Positive':
            new_list.append(0.5)    
        else:
            new_list.append(1)
    
    return new_list    


def currentDateString():
    current_date_formatted = datetime.today().strftime ('%d%m%Y')
    # convert datetime obj to string
    str_current_date = str(current_date_formatted)
    return str_current_date




def countCPU():
    num_partitions = multiprocessing.cpu_count()
    num_cores = multiprocessing.cpu_count()
    return num_cores 

def polarsSetStringLen(my_len=100):
    pl.Config.set_fmt_str_lengths(my_len)

# Ginger accept 600 words at once !

URL = "https://services.gingersoftware.com/Ginger/correct/jsonSecured/GingerTheTextFull"  # noqa
API_KEY = "6ae0c3a0-afdc-4532-a810-82ded0054236"



#pl.Config.set_fmt_str_lengths=1000


def inputDataConcatenate(my_text_list):
    my_text_list_merged = " ".join([str(item) for item in my_text_list])
    return my_text_list_merged.strip()
    

def happySadRegularExpression(my_text):
    my_text = re.sub(r"\:\)|\:\(", " ", my_text)                               # change multiple :) or :( by space
    return my_text.strip()
    
    
    
def inputDataRemoveMultipleSpace(my_text):
    my_text = re.sub(r"\s+", " ", my_text)                               # change multiple blank spaces by just one
    return my_text.strip()

    


# Wrap and truncate a string with textwrap in Python
def inputDataSplit(my_text, text_len=600):
    string_wrap_list = textwrap.wrap(my_text, text_len)
    return string_wrap_list
    



class GingerIt(object):
    def __init__(self):
        self.url = URL
        self.api_key = API_KEY
        self.api_version = "2.0"
        self.lang = "US"

    def parse(self, text, verify=True):
        session = cloudscraper.create_scraper()
        request = session.get(
            self.url,
            params={
                "lang": self.lang,
                "apiKey": self.api_key,
                "clientVersion": self.api_version,
                "text": text,
            },
            verify=verify,
        )
        data = request.json()
        return self._process_data(text, data)

    @staticmethod
    def _change_char(original_text, from_position, to_position, change_with):
        return "{}{}{}".format(
            original_text[:from_position], change_with, original_text[to_position + 1 :]
        )

    def _process_data(self, text, data):
        result = text
        corrections = []

        for suggestion in reversed(data["Corrections"]):
            start = suggestion["From"]
            end = suggestion["To"]

            if suggestion["Suggestions"]:
                suggest = suggestion["Suggestions"][0]
                result = self._change_char(result, start, end, suggest["Text"])

                corrections.append(
                    {
                        "start": start,
                        "text": text[start : end + 1],
                        "correct": suggest.get("Text", None),
                        "definition": suggest.get("Definition", None),
                    }
                )

        return {"text": text, "result": result, "corrections": corrections}




    
parser = GingerIt()

def inputDataSpellGrammerCorrection(my_text):
    
    my_corrected_text=parser.parse(my_text)
    my_clean_text=my_corrected_text['result']
    return my_clean_text



def inputDataContractions(my_text, my_slang=False):
    
    my_text_contraction=contractions.fix(my_text, slang=my_slang)
    return my_text_contraction

def addContractions(initial_text,clean_text):
    contractions.add(initial_text, clean_text)


###################################


def similarityEditDistance(phrase1,phrase2):
    try:
        if phrase1 in ['null','NULL'] or phrase2 in ['null','NULL']:
            return 1000
        else : 
            return editdistance.eval(phrase1,phrase2)
        
    except:
        return 1000        

def similarityTFIDFDistance(phrase1,phrase2):
    
    try:
        if phrase1 in ['null','NULL'] or phrase2 in ['null','NULL']:
            return 1000
        else:
            
            myTuple=(phrase1,phrase2)
            corpus=myTuple # tuple 

            # Initialize an instance of tf-idf Vectorizer
            tfidf_vectorizer = TfidfVectorizer()


            # Generate the tf-idf vectors for the corpus
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            myTFIDFSimilarity=round(cosine_sim.mean(),2)

            return myTFIDFSimilarity
    except:
        return 1000

def similarityColumnTFIDFDistance(my_dataframe,my_dataframe_column_name,my_ngram_range=(1,2),my_max_df=0.9,my_min_df=5,my_token_pattern='(\S+)'):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=my_ngram_range, max_df=my_max_df, min_df=my_min_df, token_pattern=my_token_pattern)
    my_tf_idf_matrix = tfidf_vectorizer.fit_transform(my_dataframe[my_dataframe_column_name])
    return my_tf_idf_matrix
    
    
    
def similarityJaccardDistance(phrase1,phrase2):
    try:
        if phrase1 in ['null','NULL'] or phrase2 in ['null','NULL']:
            return 1000
        else:
            myTuple=(phrase1,phrase2)
            myCombination=myTuple
            mySimilarityArray=[]
            mySetA=set(myCombination[0])
            mySetB=set(myCombination[1])


            #Find intersection of two sets
            nominator = mySetA.intersection(mySetB)

            #Find union of two sets
            denominator = mySetA.union(mySetB)
            if len(denominator)==0:
                mySimilarityArray=0


            mySimilarityArray.append(len(nominator)/len(denominator))        

            return round(statistics.mean(mySimilarityArray),2)
    except:
        return 1000 

    
# Not used !
def similarityHammingDistance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

def similarityHammingDistance2(chaine1, chaine2):
    return len(list(filter(lambda x : ord(x[0])^ord(x[1]), zip(chaine1, chaine2))))


def avgJaccardTFIDFDistance(phrase1,phrase2):
    name_similarity_score=(similarityJaccardDistance(phrase1,phrase2)+similarityTFIDFDistance(phrase1,phrase2))/2
    return name_similarity_score

##########################################







####################################################
def countChatWords(my_chat):
    my_text=my_chat
    test_string = my_text
    
    count_chat_words = len(re.findall(r'\w+', test_string))
    # total no of words
    print ("The number of words in string are : " + str(count_chat_words))
    return count_chat_words




def countWords(my_text):
    test_string = my_text
    res = len(re.findall(r'\w+', test_string))
    # total no of words
    print ("The number of words in string are : " + str(res))
    return res


def textNormalize(my_text: string) -> string:
    
    my_text=my_text.lower()
    my_text_normalized=normalize('NFKD', my_text).encode('ascii', errors='ignore').decode('utf-8')
    return my_text_normalized

def regularExpressionTextCleaning(my_text: string) -> string:
    
    text=my_text
    
    replacement = "please eat all food that you have it"
    replacement2=". please replace my soup with a pasta "
    replacement3="please replace my food dish with another recipe and meal that would have more nutrition. "
    
   
    text=text.replace('follow us on ', '')
    text=text.replace('good morning','')
    text=text.replace('good afternoon','')
    text=text.replace('good evening','')
    text=text.replace('good day','')
    
        
    text=text.replace('re:','')
    text=text.replace('fwd:','')
    text=text.replace('ext:','')
    text=text.replace('attn','')
    text=text.replace('disclaimer','')
    text=text.replace ('***** please do not respond to this email *****', '')
    text=text.replace('-----original message-----','')
    
    text=text.replace('alert: this email originated from outside. do not click links or open attachments unless you know the sender and trust the content is safe.','')
    
    text=text.replace('this message contains confidential information intended only for the person named above. if you have received this message in error, please notify the sender immediately by replying to this e-mail. if you are not the intended recipient you must not use, disclose, distribute, copy, or print this e-mail. thank you.','')
    
    text=text.replace('the information contained in this communication from the sender is confidential. it is intended solely for use by the recipient and others authorized to receive it. if you are not the intended recipient, you are hereby notified that any disclosure, copying, use, or distribution of the information included in this email is prohibited and may be unlawful.','')
    
    
    text=text.replace("confidentiality notice: the information contained with this transmission are the private, confidential property of the sender, and the material is privileged communication intended solely for the individual(s) indicated. if you are not the intended recipient, you are hereby notified that any review, disclosure, copying, distribution or the taking of any other action relevant to the contents of this transmission are strictly prohibited. If you have received this transmission in error, please contact the sender by reply email and destroy all copies of the original message.",'')
    
    text=text.replace("the information in this e-mail is confidential. it is intended for the exclusive use of the individual or entity to whom it is addressed. this message may contain information that is confidential. if the reader of this message is not the intended recipient, be aware that any disclosure, dissemination, distribution or copying of this communication, or the use of its contents, is prohibited. if you have received this email in error, please notify the sender and delete this email.",'') 
    
    text=text.replace("this message and any attached documents are only for the use of the intended recipient(s), are confidential and may contain privileged information. any unauthorized review, use, retransmission, or other disclosure is strictly prohibited. if you have received this message in error, notify the sender immediately, and delete the original message.",'')
    
    text=text.replace('all rights reserved','')
    
    text=text.replace('the entered text is','')
    
    text=text.replace ('please do not reply directly to this email. this email address is not monitored.','')
    
    text=text.replace('if you have received this message in error, or if there is a problem with the communication, please notify the sender immediately and destroy all copies of this e-mail and any attachments. the unauthorized use, disclosure, reproduction, forwarding, copying or alteration of this message is strictly prohibited and may be unlawful.','')
        
    text=text.replace('if you wish to no longer receive electronic messages from this sender, please respond and advise accordingly in your return email.','')
    
    text=text.replace('this message was system generated.','')
    text=text.replace ('if you have any questions, you can contact our team at','')
    text=text.replace ('if this is your first time logging into the payer direct hub you should have received a temporary password via email so you can complete the free registration process. ','')
    
    text=text.replace ('note: some browser security settings may prevent you from accessing the URL directly if you click on it so you may need to copy the URL text and paste it into your browsers col2_name field.', '')
        
    
    text=text.replace(""" please do not reply to this automated message """,'')
    
    text=text.replace("""let us know if you have questions or concerns""",'')
    
    text=text.replace("""please consider the environment before printing this email.""",'')
    
    text=text.replace("""e-mail messages may contain viruses, worms, or other malicious code. by reading the message and opening any attachments, the recipient accepts full responsibility for taking protective action against such code. Henry Schein is not liable for any loss or damage arising from this message.""",'')
    
    text=text.replace('the information in this email is confidential and may be legally privileged. it is intended solely for the col2_nameee(s). access to this e-mail by anyone else is unauthorized.','')
        
    text=text.replace('[external]','')
    text=text.replace('report suspicious','')
    text=text.replace('caution: this email originated outside of the company. do not click on links or open attachments unless you have authenticated the sender.','')
    text=text.replace('caution: this email originated outside of the company.\
                      do not click on links or open attachments unless you have authenticated the sender.','')
   
    text=text.replace('for more information about the handling of your personal data, please click on the following link:','') 
    text=text.replace('if you do not have a password or need assistance, our support team is here to help.','')
    text=text.replace('note: if you no longer wish to receive this notification, please contact your administrator.','')
    
    text =text.replace('--confidentiality notice: this e-mail message, including any attachments, is for the sole use of the intended recipient(s) and may contain confidential, proprietary, and/or privileged information protected by law. if you are not the intended recipient, you may not read, use, copy, or distribute this e-mail message or its attachments. if you believe you have received this e-mail message in error, please contact the sender by reply e-mail or telephone immediately and destroy all copies of the original message.','') 
    
    text =text.replace('confidentiality notice - this e-mail transmission, and any documents, files or previous e-mail messages attached to it, may contain information that is confidential and/or proprietary or legally privileged. If you are not the intended recipient, or a person responsible for delivering it to the intended recipient, you are hereby notified that you must not read or play this transmission and that any disclosure, copying, printing, distribution or use of any of the information contained in or attached to this transmission is strictly prohibited. if you have received this transmission in error, please immediately notify the sender by telephone or return e-mail and delete the original transmission and its attachments without reading or saving in any manner. thank you.','')
     
    
    text=text.replace('if you need adobe reader to view the pdf document, you can download the latest version from http://adobe.com free of charge.','')

            
    text=text.replace ('this message is from an external sender. be cautious, especially with links and attachments. ','')
    
    text=text.replace('this message is intended for the exclusive use of the intended addressed. if you have received this message in error or are not the intended col2_nameee or his or her authorized agent, please notify me immediately by e-mail, discard any paper copies and delete all electronic files of this message.','') 
    
    text=text.replace('if you have concerns about the validity of this message, please contact the sender directly','')
    text=text.replace('if you require assistance opening this message, please click here.','')

   
    text=text.replace('process food','cooking meal and food ')
    
    text=text.replace('health advice from','nutrition advice from')
    ######
    text = re.sub("nutrition information", 
                  "here we have nutrition information that helps us to prepare healthy food and meal ", text)
    
    ### 
    text=text.replace('sandwich',' a dish of fast food with delicious healthy meals')
    text=text.replace('deposited','transferred')
    
   
    text=re.sub(r"stir.*.bowl.*", replacement3, text)
    text=re.sub(r"food.*.invoice.*", "remittance invoice is attached to be procceed money transfer for food , meal and recipe .", text)
    

    text=re.sub(r"settlement.*.resturant.*", replacement3, text)
    text=re.sub(r"online.*.foodie.*",  replacement3, text)
    text=re.sub(r"menu.*.meal.*",replacement3, text)  
    text=re.sub(r"culinary arts of food.*",replacement3,text)
    
    
    text=re.sub(r"gourmet.*.culinary.*.fine.*", replacement2, text)  
   
    
    
    #processing your sentences 
    
    text=re.sub(r"running card.*", replacement3, text)
    text=text.replace("collect"+" "+r"\w*"+"food" ,'eat yummy tasty home made recipe and food that it healthy nutritious foods.  ')
    text=text.replace("process"+" "+r"\w*"+"food" ,'eat yummy tasty home made recipe and food that it healthy nutritious foods. ')
    text=text.replace("charge"+" "+r"\w*"+"food" ,'eat yummy tasty home made recipe and food that it healthy nutritious foods. ')
    text=text.replace("find"+" "+r"\w*"+"food" ,'eat yummy tasty home made recipe and food that it healthy nutritious foods. ')
    text=text.replace("authorize"+" "+r"\w*"+"food" ,'eat yummy tasty home made recipe and food that it healthy nutritious foods. ')
    text=text.replace("accept"+" "+r"\w*"+"food" ,'eat yummy tasty home made recipe and food that it healthy nutritious foods. ')
    text = re.sub("move forward"+" "+r"\w*"+"food"+r"\w*", "eat yummy tasty home made recipe and food that it healthy nutritious foods. ", text)
  

    #Food abbreviations used in the database
    # For example : https://help.pearsoncmg.com/mda/en-us/student/content/food-abbreviations.htm
    text=text.replace('add wtr','added water')
    text=text.replace('bbq ','barbeque ')
    text=text.replace('fcc ','food chemical codex ')
    text=text.replace('cl ','credit limit ')
    text=text.replace('rtb ','lready to bake ')
    text=text.replace('rte ','ready to eat ')
    
    text=text.replace('rth ','ready to heat ')
    text=text.replace('process food payment','recompense meal, recipe, well-cooked and food payment')
    #text=text.replace('food','meal, recipe, well-cooked and food ')
    
    
    if not text:
        return ''

    ############### Removing URL and Email #########################################################
    patternURLCommonRegularExpression = re.compile(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''')
    text = patternURLCommonRegularExpression.sub("", text)  # Remove all forms of URLs from a given string 
    
    patternEmailCommonRegularExpression = re.compile(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+")
    text = patternEmailCommonRegularExpression.sub("", text)  # Remove all forms of Email from a given string 
    
    
    # Emoji patterns
    emoji_pattern = re.compile("["
             u"\U0001F600-\U0001F64F"  # emoticons
             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
             u"\U0001F680-\U0001F6FF"  # transport & map symbols
             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
             u"\U00002702-\U000027B0"
             u"\U000024C2-\U0001F251"
             "]+", flags=re.UNICODE)

    #remove emojis from string
    text = emoji_pattern.sub(r'', text)

     

    text = re.sub(r"\:\)|\:\(", " ", text)                         # remove :) or :( from text      
    text = re.sub(r"([a-zA-Z]+)\?s ", r"\1 's ", text)             # special case toto?s => toto 's
    text = re.sub(r"('s|'d)", r' \1', text)                        # this one makes the other redundant
    text = re.sub(r'([$><+@{}!?()/;,%\^\[\]-])', r' \1 ', text)    # Separete this characters from text? -> text ?
    text = re.sub(r"([\-:$><+@{}!?()/;,%\^])",r" \1 ", text)       # these two do basically the same
    text = re.sub(r'([^&])([&])([^&])', r'\1 \2 \3', text)         #P&G -> P & G
    text = re.sub(r'&quot;|&lt;|&gt;|&lsquo;|&rsquo;|&ldquo;|&rdquo;|&nbsp;|&amp;|&apos;|&cent;|&pound;|&yen;|&euro;|&copy;|&reg;', ' ', text)
   
    #text = re.sub("\(.*?\)","()",text)                             # How to remove text inside brackets in Python

    text = re.sub(r"[\(|\{|\[]\s*?[\)|\}|\]]", ' ', text)          # (   ) -> '' removes empty brackets
    text = re.sub(r"'(?!(s|d))|^'", ' ', text)                     #'f -> ' ' removes apostrophe+letter that are not 's or 'd
    text = re.sub('\n', ' ', text)                                 # Remove a Newline Character From the String
   
    
    #after preprocessing the colon symbol left remain after #removing mentions
    text = re.sub(r":", " ", text)
    text = re.sub(r"‚Ä¶", " ", text)

    #replace consecutive non-ASCII characters with a space
    ###### Depends on case ########
    #text = re.sub(r'[^\x00-\x017F]+',' ', text)


    #remove symbols from text
    #atternCommonRegularExpression = re.compile(r"\{|\}|\:|\\|/|\[|\]|\+|\<|\>|\_\•|\®|\*|\"|\“|\”|\!|\^|\↑|\❏|\$|\--|\|") 
    

    patternHappyCommonRegularExpression = re.compile(r" \＼\(\^o\^\)\／|\:\-\)|\:\)|\;\)|\:o\)| \:\]| \:3| \:c\)| \:\>|\=\]|8\)| \=\)| \:}|\:\^\)| \:\-D| \:D|8\-D|8D|x\-D|xD|X\-D|XD| \=\-D| \=D|\=\-3| \=3| \:\-\)\)| \:\'\-\)| \:\'\)| \:\*| \:\^\*| \>\:P| \:\-P| \:P|X\-P|x\-p| xp| XP|\:\-p|\:p|\=p|\:\-b|\:b| \>\:\)| \>\;\)| \>\:\-\)|\<3 ")
    text = patternHappyCommonRegularExpression.sub("", text)  #

    patternSadCommonRegularExpression = re.compile(r"\=/\/\|;\(|>\:?\\*|\:\{|\:c|\:\-c|\:'\-\(|>.*?<|:\(|>\:\(|=\/|\:L|\:-/|\>:/|\:S|\:\[|\:\-\|\|\:\-\)|\:\-\|\||\=L|\:<|\:\-\[|\:\-<|=/\/\|=\/|>\:\(|\:\(|\:'\-\(|\:'\(|\:?\\*|\=?\\?")
    text = patternSadCommonRegularExpression.sub("", text)  #
   
    # with comma removal 
    patternCommonRegularExpression = re.compile(r"\{|\}|\:|\\|/|\[|\]|\+|\<|\>|\_\•|\®|\*|\"|\“|\”|\!|\^|\↑|\®|\❏|\→|\$|\--|\||,")    
 

    text = patternCommonRegularExpression.sub("", text)  #
    
    text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text) # removing/replacing the bullet points
    
    # Modified 13 March 2023 

    text = text.translate({ ord(c): None for c in "~--	._!§*		" }) # with Translate removing different symbols
    
  
    
    
    text = re.sub(r"\s+", " ", text)                               # change multiple blank spaces by just one

    
    
    
    return text.strip()
  
    

# https://stackoverflow.com/questions/713798/regex-to-find-a-number-in-a-string    



#work with python version 3.11  : remove str2 from str1  such as removing a resturant name from a meal recipe 
def removeCompanyName(str1: string,str2: string) -> string:  
    str2=str2.lower()
    return str2.removeprefix(str1.lower())


# work with old versions
def removeCompanyName_old(str1: string,str2: string) -> string: 
    if str1 in str2:
        return str2.replace(str1,'')            



def currentDateString():
    current_date_formatted = datetime.today().strftime ('%d%m%Y')
    # convert datetime obj to string
    str_current_date = str(current_date_formatted)
    return str_current_date


def textNormalize(my_text: string) -> string:
    
    my_text=my_text.lower()
    my_text_normalized=normalize('NFKD', my_text).encode('ascii', errors='ignore').decode('utf-8')
    return my_text_normalized


    
# USA - col2_name standardization (or col2_name normalization) : Correcting zip code records to a standard format (Potential 9-digits) 

def cleanUSZipCode(text: string) -> string:
    text=str(text)
    x = re.search("\d{5}", text)
    if x:
        x2=re.search("\d{5}\-\d{4}", text)
        if x2:
            text =re.sub(r"\d{5}",x2.group(), text, 1)
            

            text_split = re.search("\d{5}\-\d{4}", text)
            text_splitted = text[:text_split.span()[1]]
            clean_col2_name=text_splitted
        else:
            clean_col2_name=text
    else:
        clean_col2_name=text
    return clean_col2_name


##################################################################################################
#https://pola-rs.github.io/polars-book/user-guide/howcani/missing_data.html
#https://stackoverflow.com/questions/67834912/apply-function-to-all-columns-of-a-polars-dataframe


def dataframe_NaNRemoval_polar_cols(df_read,col1=-2,col2=-1):
    my_col_list=df_read.columns
    
    
    
    df_read_NaNRemoved = df_read.with_column(pl.col(my_col_list[col1]).fill_null(pl.lit('NULL'),), )
    df_read_NaNRemoved = df_read.with_column(pl.col(my_col_list[col2]).fill_null(pl.lit('NULL'),), )
    

def dataframe_NaNRemoval_polar(df_read):
    my_col_list=df_read.columns
    
    
    
    df_read_NaNRemoved = df_read.with_column(pl.col(my_col_list[-1]).fill_null(pl.lit('NULL'),), )
    
    
    df_read_NaNRemoved.apply(lambda x: str(x))
    df_read_NaNRemoved.apply(lambda x: x if x is not None else 'NULL')
    
    return df_read_NaNRemoved





def dataframe_NaNRemoval(df_read):
    
    df_read_NaNRemoved=df_read.fillna('NULL')
    #print('step1 -Fill NULL')
    df_read_NaNRemoved.apply(lambda x: str(x))
    #print('step2 - String Casting')
    df_read_NaNRemoved.fillna('NULL', inplace=True)
    #print('step3 - Refill NULL')
    df_read_NaNRemoved.apply(lambda x: x if x is not None else 'NULL')
    #print('step4 - Dataframe after NULL, None and NaN Removal \n')
    
    return df_read_NaNRemoved


def dataframe_lower_removeSpaces_from_string(my_string: string) -> string:
    
    return my_string.lower().translate({ord(c): None for c in string.whitespace})



def dataframe_lower_string(my_string: string) -> string:
    
    return my_string.lower().strip()


#'row_ID','text_ID','col1_name','col2_name'
# Customer_Id','my_col1_name','my_alt_Id'

def preProcessingcol1_nameFunction(data_split):
    out = data_split.select(
    [
        pl.col('idx'),
        pl.col('row_ID'),
        pl.col('my_col1_name'),
        pl.col('Customer_Id'),
        pl.col('text_ID'),
        pl.col("col1_name").apply(dataframe_RegularExpression_patternRemoval).apply(dataframe_lower_string).alias("col1_name_clean")
        # pl.col("col2_name"),
        # pl.col('my_alt_Id')     
    ]
    )
    return out



#row_ID	dim_AccountId	col1_name	my_alt_Id	col2_name	Bill_Addr

def preProcessingcol1_nameSameBarCodeFunction(data_split):
    out = data_split.select(
    [
        pl.col('idx'),
        pl.col('row_ID'),
        #pl.col('my_col1_name'),
        pl.col('dim_AccountId'),
        pl.col('my_alt_Id'),
        pl.col("col1_name").apply(dataframe_RegularExpression_patternRemoval).apply(dataframe_lower_string).alias("col1_name_clean"),
        pl.col("col2_name")
        # pl.col('my_alt_Id')     
    ]
    )
    return out






def preProcessingcol1_nameFunction2(data_split):
    out = data_split.select(
    [
        
        pl.col('idx'),
        pl.col('row_ID'),
        pl.col('Customer_Id'),
        pl.col("col1_name").apply(dataframe_RegularExpression_patternRemoval).apply(dataframe_lower_string).alias("col1_name_clean"),
        #pl.col("col1_name"),
        pl.col('my_alt_Id')     
    ]
    )
    return out



def preProcessingFunction0(data_split):
    out = data_split.select(
    [
        pl.col('dim_AccountId'),
        pl.col("col1_name").apply(dataframe_RegularExpression_patternRemoval).apply(dataframe_lower_string).alias("col1_name"),
        pl.struct(["col1_name", "col2_name"]).apply(lambda cols: removeCompanyName(cols["col1_name"], cols["col2_name"])).
        apply(lambda myString: cleanUSZipCode(myString)).
        apply(lambda myString: dataframe_RegularExpression_patternRemoval_col2_name(myString)).alias("sorted_clean_col2_name")
        #         pl.col("Sales").apply(add_counter).alias("a2"),
#         (pl.col("Sales") + pl.arange(1, pl.count() + 1)).alias("a3"),
#         (pl.col("Sales") + 10).apply(add_counter).apply(add_counter).alias("a4"),
#         (pl.col("Sales") + 1000).alias("a5"),
#       pl.struct(["Company", "Country"]).apply(lambda x: len(x["Company"]) + len(x["Country"])).alias("a6"),
#       pl.struct(["Company", "Country"]).apply(lambda x: x["Company"] +" "+ x["Country"]).alias("a7"),

        
     
      #pl.all(),
     
     
     
    ]
    )
    return out



# it was : def preProcessingFunction(data_split)
def preProcessingcol1_namecol2_nameFunction(data_split):
    out = data_split.select(
    [
        pl.col('row_ID'),
        pl.col('dim_AccountId'),
        pl.col("col1_name"),
        pl.col("col2_name"),
        pl.col("col1_name").apply(dataframe_RegularExpression_patternRemoval).apply(dataframe_lower_string).alias("col1_name"),
        pl.struct(["col1_name", "col2_name"]).apply(lambda cols: removeCompanyName(cols["col1_name"], cols["col2_name"])).
        apply(lambda myString: cleanUSZipCode(myString)).
        apply(lambda myString: dataframe_RegularExpression_patternRemoval_col2_name(myString)).alias("sorted_clean_col2_name")
        #         pl.col("Sales").apply(add_counter).alias("a2"),
#         (pl.col("Sales") + pl.arange(1, pl.count() + 1)).alias("a3"),
#         (pl.col("Sales") + 10).apply(add_counter).apply(add_counter).alias("a4"),
#         (pl.col("Sales") + 1000).alias("a5"),
#       pl.struct(["Company", "Country"]).apply(lambda x: len(x["Company"]) + len(x["Country"])).alias("a6"),
#       pl.struct(["Company", "Country"]).apply(lambda x: x["Company"] +" "+ x["Country"]).alias("a7"),

        
     
      #pl.all(),
     
     
     
    ]
    )
    return out




def combinationProcessingFunction(df_pl):
    df_combinations = df_pl.join(df_pl,how="cross", on="idx",  suffix="_2", ).filter(pl.col("idx") != pl.col("idx_2"))
    return df_combinations


def probabilisticSimilarityProcessingFunction(df_combinations):
    out=df_combinations.with_columns(
         [
          # Combine "idx" and "idx_2" columns to one struct column.
           pl.struct(pl.col(["idx", "idx_2"])).alias("idx_comb"),
           # Combine "full_name" and "full_name_2" columns to one struct column.
           pl.struct(pl.col(["col1_name_clean", "col1_name_clean_2"])).alias("full_name_comb"),
       ]
   ).with_columns(
       [
         # Run custom functions on struct column.
           pl.col("full_name_comb").apply(lambda t: Levenshtein.normalized_similarity(t["col1_name_clean"], t["col1_name_clean_2"])).alias("levenshtein"),
           pl.col("full_name_comb").apply(lambda t: JaroWinkler.similarity(t["col1_name_clean"], t["col1_name_clean_2"])).alias("jarowinkler"),
           pl.col("full_name_comb").apply(lambda t: similarityTFIDFDistance(t["col1_name_clean"], t["col1_name_clean_2"])).alias("TFIDFDistance"),
           pl.col("full_name_comb").apply(lambda t: similarityJaccardDistance(t["col1_name_clean"], t["col1_name_clean_2"])).alias("similarityJaccardDistance"),
           pl.col("full_name_comb").apply(lambda t: avgJaccardTFIDFDistance(t["col1_name_clean"], t["col1_name_clean_2"])).alias("avgJaccardTFIDFDistance"),
          
        
       ]
   )
    return out


def probabilisticJarowinklerSimilarityProcessingFunction(df_combinations):
    out=df_combinations.with_columns(
         [
          # Combine "idx" and "idx_2" columns to one struct column.
           pl.struct(pl.col(["idx", "idx_2"])).alias("idx_comb"),
           # Combine "full_name" and "full_name_2" columns to one struct column.
           pl.struct(pl.col(["col1_name_clean", "col1_name_clean_2"])).alias("full_name_comb"),
       ]
   ).with_columns(
       [
         # Run custom functions on struct column.
           #pl.col("full_name_comb").apply(lambda t: Levenshtein.normalized_similarity(t["col1_name_clean"], t["col1_name_clean_2"])).alias("levenshtein"),
           pl.col("full_name_comb").apply(lambda t: JaroWinkler.similarity(t["col1_name_clean"], t["col1_name_clean_2"])).alias("jarowinkler"),
           #pl.col("full_name_comb").apply(lambda t: similarityTFIDFDistance(t["col1_name_clean"], t["col1_name_clean_2"])).alias("TFIDFDistance"),
           #pl.col("full_name_comb").apply(lambda t: similarityJaccardDistance(t["col1_name_clean"], t["col1_name_clean_2"])).alias("similarityJaccardDistance"),
           #pl.col("full_name_comb").apply(lambda t: avgJaccardTFIDFDistance(t["col1_name_clean"], t["col1_name_clean_2"])).alias("avgJaccardTFIDFDistance"),
          
        
       ]
   )
    return out

