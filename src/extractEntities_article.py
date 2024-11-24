from flair.data import Sentence
from flair.nn import Classifier

import os
import pandas as pd
import nltk
from tqdm import tqdm

import artcileDetail_xml as ad

path_to_included_articles="/home/aberhe/Projects/SANTAL/ASLR/Results/Included_articles"
path_to_results="/home/aberhe/Projects/SANTAL/ASLR/Results/BioNER/"

tagger = Classifier.load("hunflair")
def extractSave(file_xml,bacteria_name="Veillonella"):
    #file_xml='/data/projects/santal/data/PMC_Articles/oa_comm/xml/all/PMC10000167.xml'#PMC10000014.xml'
    folder_path=os.path.join(path_to_results,bacteria_name)
    if os.path.exists(file_xml):
        with open(file_xml, 'r') as f:
            data = f.read()
        #try:
        articleDetails=ad.ArticleDetails(file_xml)
        #articleDetails.getDetails()
        secs=articleDetails.getSections()
        #except:
        #   print(file_xml,"xml file has key problem!")
        sent_id_d=file_xml.split("/")[-1].replace(".xml","")
        result_file=folder_path+"/"+sent_id_d+".csv"
        if not os.path.exists(result_file) and secs!=[]:
            df=pd.DataFrame(columns=("sent_id","sentence","entity","start_chunck","end_chunk","label","score"))
            i=0
            for key in secs:#articleDetails.sections:
                sentences = nltk.sent_tokenize(secs[key])#articleDetails.sections[key])
                for sentence in sentences:
                    sentence=Sentence(sentence)
                    tagger.predict(sentence)
                    sent_id=sent_id_d+"_"+str(i)
                    for labels in sentence.get_labels():
                        #print(la.value,la.data_point.text,la.data_point.start_position,la.data_point.end_position)
                        taggedEntities=[sent_id,sentence.tokenized,labels.data_point.text,labels.data_point.start_position,labels.data_point.end_position,labels.value,round(labels.score,3)]
                        df.loc[len(df)]=taggedEntities
                    i=i+1
            if not os.path.exists(folder_path):
                #folder_path=os.path.join(path_to_results,bacteria_name)
                os.mkdir(folder_path)

            df.to_csv(folder_path+"/"+sent_id_d+".csv")
        
        
    else:
        print(file_xml,"Does not exist")

def extarctEntities(file_xml,bacteria_name):
    folder_path=os.path.join(path_to_results,bacteria_name)
    if os.path.exists(file_xml):
        with open(file_xml, 'r') as f:
            data = f.read()
        articleDetails=ad.ArticleDetails(file_xml)
        articleDetails.getDetails()
        
        sent_id_d=file_xml.split("/")[-1].replace(".xml","")
        result_file=folder_path+"/"+sent_id_d+".csv"
        if not os.path.exists(result_file):
            df=pd.DataFrame(columns=("sent_id","sentence","entity","start_chunck","end_chunk","label","score"))
            i=0
            for key in articleDetails.sections:
                sentences = nltk.sent_tokenize(articleDetails.sections[key])
                for sentence in sentences:
                    sentence=Sentence(sentence)
                    tagger.predict(sentence)
                    sent_id=sent_id_d+"_"+str(i)
                    for labels in sentence.get_labels():
                        #print(la.value,la.data_point.text,la.data_point.start_position,la.data_point.end_position)
                        taggedEntities=[sent_id,sentence.tokenized,labels.data_point.text,labels.data_point.start_position,labels.data_point.end_position,labels.value,round(labels.score,3)]
                        df.loc[len(df)]=taggedEntities
                    i=i+1
        return (df,folder_path+"/"+sent_id_d+".csv")
        

def parallel_ner_extraction(file_list,bacteria_name):
        num_cores = mp.cpu_count()
        print(num_cores)
        pool = mp.Pool(processes=num_cores-10)

        progress_bar = tqdm(total=len(self.files),desc='searching in '+str(len(self.files))+" files")
        items=[(self.search_pattern, file) for file in self.files]
        
        pattern=self.search_pattern
        # Apply the function to each file path in parallel
        results = []
        for result in pool.imap_unordered(extractSave, (file_list,bacteria_name)):
            progress_bar.update(1)  # Update progress bar
            if result is not None:
                results.append(result)
        #results=pool.map(search_file,files)

        pool.close()
        progress_bar.close()
        return results


def main():
    bacteria_to_search_df=pd.read_csv("/home/aberhe/Projects/SANTAL/ASLR/data/Bacteria_to_search_with_higher_taxonomy.csv")

    for bacteria in bacteria_to_search_df["Species"].tolist():
        print("Bacteria: {}".format(bacteria))
        print()
        species_splt=bacteria.split("_")
        if len(species_splt)>2:
            bacteria_name= species_splt[-1]
        else:
            bacteria_name= species_splt[0]
        
        with open(os.path.join(path_to_included_articles,(bacteria_name+"_SPECIES.txt")), "r") as f:
            files_list=f.readlines()
        
        for file in tqdm(files_list):
            file_xml=file.replace("txt","xml").replace("\n","")
            extractSave(file_xml=file_xml,bacteria_name=bacteria_name)
        

if __name__=="__main__":
    main()