# Text Zero-shot learning behind Text with Different Length 

    __status__          = "Dev"
    __version__         = "1.0"
    __clientname__      =  "{School of Continuing Studies - McGill University, Data Science Dept. }"
    __copyright__       = "Copyright 2023, {McGill University, Montréal, Québec, Canada }" 

Contributors
--------------

    __project_manager __   = "sampleProjectManager@mcgill.ca"
    __data_engineer   __   = "sampleDataEngineer@mcgill.ca"
    __data_scientist  __   = "sampleDataScientist@mcgill.ca"

  
Releases
--------------

| status | ver | date | categories |
| :--------| :--------| :-------- | :-------- |
| PoC | 1.0 | 30 March 2023 | Text Classification |


## Summary : 

The purpose of the project is to evaluate the capability of zero-shot learning on unlabeled dataset with various text length and formality.
It returns :

Labels : 'Food', 'Non-Food', 'Potential'
Score : [0 , +1]

## Project Layout :
Project root in ... instance (): 

*/home/users/zadeh/zeroshotclassification*

## Getting started
--------------
Guide potential devolers / users through getting code up-and-running on their own system. In the following sub-sections, you 
could walk though sub-sections:

1.  Introduction
3.  How to config
2.  How to call API 
4.  Tools/Dependencies
5.  Test Discovery
6.  Future work
7.  References

Introduction
--------------


How to config
--------------
Create an environment from an [environment-zeroshotclassification.yaml](https://github.com/MNLepage08/YCNG-229/blob/main/Zero-Shot%20Learning/readme/environment-zeroshotclassification.yaml) file
  ```
  $ conda env create -f readme/environment-zeroshotclassification.yaml
  ```
  Activate the new environment:
  ```
  $ conda activate zeroshot 
  ```
  Verify that the new environment was installed correctly:
  ```
  $ conda env list
  ```
  Deploy the model locally
  ```
  $ python models/model_download_and_cache.py  
  ```

How to call API / End point 
--------------


Tools/Dependencies
--------------



Test Discovery
--------------



Future work
--------------


References
----------


Git commands 
----------


(c) 2023, [McGill University, Montréal, Québec, Canada ](https://continuingstudies.mcgill.ca/public/category/courseCategoryCertificateProfile.do?method=load&certificateId=569881) License.
