########################################################################################################
# Streamlit - Zero shot Classification - single raw text as input 
# modified 18 March 2023
# by Mehdi (zadeh1980mehdi@gmail.com)


#############################
from pathlib import Path
import numpy as np 
import pandas as pd
import math
import os 
import time
import sys
import polars as pl 
import streamlit as st 
pl.Config.set_fmt_str_lengths=1000

########################################

service_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(service_root)

######################################### 


from src.text_preprocessing import countWords,textNormalize,regularExpressionTextCleaning,inputDataContractions
from src.text_input_data  import readcsvInputData,transactionIDEmailClassification,convertListTostring
# from src.text_input_data import testAccess
from src.text_input_data import readtsvStreamlitInputData,createEmailDFFromText

from src.text_input_data  import inputDataEmailClassification,appendSuffixPrefixList,currentDateString
from src.text_labels import emailClassificationLabels,emailClassificationThresholdLabels

import eds_entry


def dict_to_polarData(my_dict):
    data=my_dict
    
    dataframe_polars = pl.DataFrame(data)
    return dataframe_polars

def polarData_to_tsv(my_dataframe_polars,my_sep_):
    dataframe_polars=my_dataframe_polars
    return dataframe_polars.write_csv(sep=my_sep_).encode('utf-8')
    
def show_upload():
    st.session_state.downloaded = True

def calculation(test_data_dataframe,my_input_limit=1):
    test_data_clean=[]
    
    # =============== Slice of Data ---------------------
    test_data_dataframe=test_data_dataframe.iloc[:my_input_limit]

    #################################################################################

    df_email_row = test_data_dataframe['row']
    df_email_row_list = df_email_row.tolist()

    # when we have manual labels : ####################################################
    # df_email_label = test_data_dataframe['label']
    # df_email_label_list = df_email_label.tolist()

    df_email_id = test_data_dataframe['id'].str.strip()
    df_email_id_list = df_email_id.tolist()


    df_email_title = test_data_dataframe['title'].str.strip()
    df_email_title_list = df_email_title.tolist()


    df_email_body = test_data_dataframe['body'].str.strip()
    df_email_body_list = df_email_body.tolist()


    # Either Email body or Email Title :? 
    test_data=df_email_body_list
    #test_data=df_email_title_list

  

    ############ Batch of data ##################
    
    df_email_id_list=convertListTostring(df_email_id_list)




    suf_res=appendSuffixPrefixList(test_data, len_str=20,  prefix="This is a sample email that I would like to share here. ", suffix=". this is a sample text added at the end of email paragraph. ")


    test_data=suf_res



    #################################################################################################


    print('Number of text:', len(test_data))


    for email_body in test_data:
        email_body_clean=textNormalize(email_body)
        email_body_clean=regularExpressionTextCleaning(email_body_clean)
        test_data_clean.append(email_body_clean)



    print('Number of Text - cleaned:', len(test_data_clean))




    my_class_labels=emailClassificationLabels()

    my_threshold_value=emailClassificationThresholdLabels()
    my_current_date= currentDateString()

    st=time.time()
    print("Starting ....................") 

    print("------------------------------------")

    returned_values = eds_entry.main(sentences = test_data_clean, labels=my_class_labels, multi_label=True, verbose=True, threshold_value=my_threshold_value)

    print("----------------------------")
    et=time.time()
    elapsed_time=et-st
    print("Duration: Text classification with Zero Shot Learning " , elapsed_time , 'seconds')

    

    
    data = {"Machine_Learning_Label": returned_values,"Email_Body": df_email_body_list ,"Email_Row":df_email_row_list,"Email_ ID":df_email_id_list, "Email_Title":df_email_title_list}
    return data



###############


def main():
    

    st.info(" Zero-shot Classification (Food - Related vs Non-Food Related) - Data Science Dept -  McGill  - March 2023 - ")
    st.warning("Takes few seconds/mins to present result.")
    
    #adding a text area input widget

    my_email_body = st.text_area('Text body to analyze: ',max_chars=2000000)

    #displaying the text entered by the user

    st.write('The entered text is: ',my_email_body)
 
   
    if 'downloaded' not in st.session_state:
        st.session_state.downloaded = True
        
    if my_email_body:
      
        # Desired Columns in raw dataframe
        my_cols_=['row','id','title', 'body']
        my_col_1_='title' 
        my_col_2_='body' 
        my_sep_='\t'
        my_lineterminator_='\n'

        test_data_clean=[]

############################################################################

        
        
        test_data_dataframe=createEmailDFFromText(my_email_body)
        

        
        st.dataframe(test_data_dataframe.head(1))
        
        #st.markdown(""" div.stButton > button:first-child {background-color: #00cc00;color:white;font-size:20px;height:3em;width:30em;border-radius:10px 10px 10px 10px; } """, unsafe_allow_html=True)
        
        
       
        start_calculation = st.button('Zero-shot Classification')

        if start_calculation:
            
            my_result_dict = calculation(test_data_dataframe)
            my_polarData_temp=dict_to_polarData(my_result_dict)
            my_result_label=my_result_dict["Machine_Learning_Label"]

           
            
            st.write('<p style="font-size:30px; color:blue;">{}</p>'.format(my_result_label),
unsafe_allow_html=True)
                      

            
            
            #upload_placeholder.empty()
            download_placeholder = st.empty()
            tsv_data =polarData_to_tsv(my_polarData_temp,my_sep_)
            
              
            
            #if st.session_state.downloaded and st.session_state.uploaded:
            if st.session_state.downloaded:    
                my_current_date= currentDateString()
                is_downloaded = download_placeholder.download_button(label='Download TSV file',
                                                                     data=tsv_data,
                                                                     file_name="{}_streamlit_singleEntry_text_classification_labels.tsv".format(my_current_date),
                                                                     mime="text/csv",
                                                                     on_click=show_upload)
                st.session_state.uploaded = True

                if is_downloaded:
                    st.write(
                        f":copyright: {datetime.datetime.now().year}."
                        f"McGil. All rights reserved."
                        f"Successfully done!"
                    )
                    download_placeholder.empty()







###############################

if __name__ == '__main__':

    main()

# streamlit run myApp.py
