# Modified 17 March 2023
# By Mehdi 
#["food","cooking",'recipe',"nutrition","politics","waive and suspension","confirmation"]

import argparse
from pprint import pprint
from typing import List,Dict,Union,Optional
from dataclasses import dataclass


from src.auto_zero_shot_classifier import zero_shot_classifier

def _preprocess(sequences:List[str]):
    """fix None and empty string for zero-shot inputs"""
    if isinstance(sequences,list):
        return [' ' if s is None or str(s).strip() == '' else s for s in sequences ]
    return sequences
    
def _postprocess(r:Dict):
    if r.get('sequence') == ' ':
        r['scores'] = [0]*len(r['scores'])
        # r['labels'] = [None] *len(r['labels'])
        r['sequence'] = None
    else:
        r['scores'] = [round(scr,3) if label is None or str(label).strip() != '' else 0 for scr,label in zip(r['scores'] , r['labels'])]
    return r


@dataclass  
class ArgSanitizer:
    sentences:str
    labels:Union[List[str],str]
    multi_label: Optional[bool] = False
    hypothesis_template: Optional[str] = None
    verbose: Optional[bool] = False
    
    def __post_init__(self):
        #inputs str or list[str]
        sentences_ = self.sentences
        sentences_ = [sentences_] if isinstance(sentences_,str) else sentences_
        self.sentences = sentences_
        if not self.is_list_of_string(sentences_):
            raise TypeError("sentences type must be a string or a list of string.")
            
        self.labels  = self.convert_label_str2list(self.labels)
        labels_ = self.labels
        if len(labels_) == 0 :
            raise ValueError("You must include at least one label value.")
        if not self.is_list_of_non_empty_string(labels_):
            raise TypeError("labels type must be a list of non empty string or a string of comma-separated labels.")
        
        hypothesis_ = self.hypothesis_template
        if hypothesis_ is not None:
            if not isinstance(hypothesis_,str):
                raise TypeError("hypothesis_template type must be a string and contains the `{}` label placeholder")
            elif hypothesis_.format('_UNK_') == hypothesis_:
                raise ValueError("The hypothesis_template must include a `{}` for label placeholder")
        
        multi_label_ = self.multi_label     
        if not isinstance(multi_label_,bool):
            raise TypeError("multi_label type must be a boolean `True or False`")
        
        verbose_ = self.verbose
        if not isinstance(verbose_,bool):
            raise TypeError("verbose type must be a boolean `True or False`")
 
    def convert_label_str2list(self, labels):
        """convert comma separated labels to list of str: 'label1,label2' -> ['label1','label2']"""
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels
        
    def is_list_of_string(self,sequences:List[str]) -> bool:
        if isinstance(sequences,list):
            for s in sequences:
                if not isinstance(s,str):
                    return False
            return True
        return False
    
    def is_list_of_non_empty_string(self,sequences:List[str]) -> bool:
        if isinstance(sequences,list):
            for s in sequences:
                if self.is_empty_string(s):
                    return False
                if not isinstance(s,str):
                    return False
            return True
        return False
    
    def is_empty_string(self,s):
        if isinstance(s,str):
            s = s.strip()
            return len(s) == 0
        return False
       
       
       
def main(sentences='', labels='', multi_label = False, hypothesis_template = None ,verbose=False, threshold_value=0.85):
    """

    Entry point to service logics

    NOTE:

    Deployment/release pipeline may call this script directly from python as

    sentences = [
    "MSc / Ain Shams University, Cairo, Egypt / 2002",
    "Combined Residency in Internal Medicine and Psychiatry, Medicine And Psychiatry, Duke University School of Medicine, 2012 - 2017",
    "DUMC 3670, 40 Duke Medicine Circle, Durham, NC 27710",
    "Palmetto Health Emergency Medicine"
    ]
    
    labels = ["degree & year","address street","medical practice","Residency program"]
    
    res = eds_entry.main(inputs = sentences, labels = labels,multi_label=True,verbose=True)

    """
    my_potential_Food_label=["food","cooking",'recipe',"nutrition"]
    my_final_seq_list=[]
    my_final_class_list=[]
    
    sentences = _preprocess(sentences)
    argfn = ArgSanitizer(
            sentences = sentences,
            labels = labels,
            multi_label = multi_label,
            hypothesis_template = hypothesis_template,
            verbose = verbose
        )
    
    
    hypothesis_template = argfn.hypothesis_template
    
    results = zero_shot_classifier(
                argfn.sentences,
                candidate_labels= argfn.labels,
                hypothesis_template = "this text is about {}" if hypothesis_template is None else hypothesis_template,
                multi_label = argfn.multi_label
            )
    
    
    for i,r in enumerate(results):
        my_potential_score=[]
        my_potential_score_non=[]
        
#         my_potential_score_non.append(r_scores[my_index])
        
        r = _postprocess(r)
        
        r_sequence=r['sequence']
        r_scores=r['scores']
        r_labels=r['labels']
        
        #results_non[i] = r_labels[:len(r_scores)]
        
        for my_label_non in my_potential_Food_label:
            
            my_index_non=r_labels.index(my_label_non)
            my_potential_score_non.append(r_scores[my_index_non])
        
        
        avg_Food_non=float(sum(my_potential_score_non)/len(my_potential_score_non))
            
        if avg_Food_non <= 0.25:
            my_final_class="Non-Food (almost perfect) -"+ str("%") + str(1-avg_Food_non)
        
        if 0.25 <avg_Food_non <= 0.35:
            my_final_class="Non-Food (substantial)-"+ str("%") + str(1-avg_Food_non)
        
        if 0.35 <avg_Food_non <= 0.88:
            my_final_class="Food (Potential)-"+ str("%") + str(avg_Food_non)
        #my_final_class="Non-Food -"
        
        
        
        r_scores_sorted = sorted(my_score for my_score in r_scores if my_score >= threshold_value)
        # print("----- r_scores_sorted ------")
        # print(r)
        # print(type(r))
        # print(type(r_scores_sorted))
        # print(len(r_scores_sorted))
        
        results[i] = r_labels[:len(r_scores_sorted)]
        
        for my_label in my_potential_Food_label:
            try:
                my_index=results[i].index(my_label)
                if my_index in [0,1,2,3]:
                    my_potential_score.append(r_scores_sorted[my_index])
                    if len(my_potential_score)>=3:
                        avg_Food=float(sum(my_potential_score)/len(my_potential_score))
                        
                        if 0.3 <= avg_Food <= 0.89:
                            my_final_class="Food (Substantial)-"+str("%")+str(avg_Food)
                            #my_final_class="Potential Food - Review"
                        
                        if 0.1 <= avg_Food <= 0.299:
                            my_final_class="Non-Food (Potential)-"+str("%")+str(1-avg_Food)
                            #my_final_class="Potential Food - Review"
                        
                        
                        if avg_Food > 0.89:
                            my_final_class="Food (Almost Perfect)- "+str("%")+str(avg_Food)
                            #my_final_class="Food"
#                 else:
                    
#                     results[i] = r_labels[:len(r_scores)]
#                     my_index=results[i].index(my_label)
#                     my_potential_score.append(r_scores[my_index])
#                     if len(my_potential_score)==4:
#                         avg_Food=float(sum(my_potential_score)/len(my_potential_score))
#                         my_final_class="Non-Food - "+str("%")+str(1-avg_Food)      
                    
                    
                            
                            
            except:
                pass
            # if my_index in [0,1,2]:
            #     my_potential_score.append(r_scores_sorted[my_index])
            #     if len(my_potential_score)==3:
            #         avg_Food=float(sum(my_potential_score)/3)
            #         if avg_Food > 0.89:
            #             my_final_class="Food"
                        
         
        
        
        my_final_seq_list.append(r_sequence)
        my_final_class_list.append(my_final_class)
        if verbose: 
            print("\n\n")
            pprint(r,indent=3,width=150,sort_dicts=False)
    
            
    #return results
    return my_final_class_list
        

if __name__ == '__main__':
    """
    How to access this entry point ?

    $ python eds_entry.py --sentences="Palmetto Health Emergency Medicine" --labels="degree & year,address street,medical practice" --multi_label --verbose
    """
    parser = argparse.ArgumentParser(description='Simple Interface of Zero Shot deep-learning leveraging NLI models')
    parser.add_argument('-s', '--sentences',
                        type=str,
                        required=True,
                        help="sentences (`str` or `List[str]`): The sequence(s) to classify, will be truncated if the model input is too large.")
                        
    parser.add_argument('-l', '--labels',
                        action='store',
                        type=str,
                        required=True,
                        help='The set of possible class labels to classify each sequence into. Can be a single label, a string of comma-separated labels')
                
    parser.add_argument("-t", "--hypothesis_template", 
                        action="store",
                        help="""(`str`, *optional*, defaults to `"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. For example, the default
                template is `"This example is {}."` With the candidate label `"sports"`, this would be fed into the
                model like `"<cls> sequence to classify <sep> This example is sports . <sep>"`. The default template
                works well in many cases, but it may be worthwhile to experiment with different templates depending on
                the task setting..""")
                
    parser.add_argument("-m", "--multi_label", 
                        action="store_true",
                        help="""(`bool`, *optional*, defaults to `False`):
                Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that
                the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.""")
                
    parser.add_argument("-v", "--verbose", 
                        action="store_true",
                        help="print the output")

    args = parser.parse_args()
    #TODO allow multiple inputs or read inputs sequences from a file
    main(sentences=args.sentences, labels=args.labels ,multi_label = args.multi_label, hypothesis_template = args.hypothesis_template ,verbose=args.verbose,threshold_value=0.85)
   
   
