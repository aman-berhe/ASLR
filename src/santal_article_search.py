import pandas as pd 
import json
import os
import re
import sys
from tqdm import tqdm

#Multiprocessing
import multiprocessing as mp

#Loading xml file content
import artcileDetail_xml as ad

def search_string(args):
        """
        if  self.file_type.endswith(".txt"):
            with open(file,"r",encoding='utf-8') as f:
                content=f.readlines()
                if self.search_pattern in content:
                    return file
        elif self.file_type.endswith(".xml"):
            artCont=ad.ArticleDetails(file)
            artCont.getArticleContent()
            if self.search_pattern in artCont.articleContent:
                return file
        else:
            return None
        """
        pattern=args[0]
        file_path=args[1]
        try:
            with open(file_path, 'r',encoding='utf-8') as file:
                content = file.read()
                search_res=re.search(pattern.lower(),content.lower())
                #if (pattern in content or patter.lower() in content.lower()) and ("cirrhosis" in content or "Cirrhosis" or "CIRRHOSIS" in content):
                if search_res!=None and "cirrhosis" in content.lower():
                    #print("found ",file_path.split(".")[-1])
                    return file_path
                else:
                    return None
        except:
           #print("BAD FILE FORMAT!")
           return None

class Santal():
    
    def __init__(self,file_type="txt",where='all',search_pattern="Veillonella"):
        self.search_pattern=search_pattern
        self.file_type=file_type
        self.ROOT_PATH="/data/projects/santal/data/PMC_Articles"
        self.OA_COMM_PATH="/data/projects/santal/data/PMC_Articles/oa_comm/"+ self.file_type+"/all"
        self.OA_NONCOMM_PATH="/data/projects/santal/data/PMC_Articles/oa_noncomm/"+ self.file_type+"/all"
        self.AUTHOR_MANUSCRIPT="/data/projects/santal/data/PMC_Articles/author_manuscript/"+ self.file_type+"/all"
        self.PHE_TIMEBOUND="/data/projects/santal/data/PMC_Articles/phe_timebound/"+ self.file_type+"/all"

        self.files_metadata_oa_comm=pd.read_csv("/data/projects/santal/data/PMC_Articles/oa_comm/"+ "txt"+"/metadata/csv/oa_comm.filelist.csv")
        self.files_metadata_oa_noncomm=pd.read_csv("/data/projects/santal/data/PMC_Articles/oa_noncomm/"+"txt"+"/metadata/csv/oa_noncomm.filelist.csv")
        self.files_metadata_author_manuscript=pd.read_csv("/data/projects/santal/data/PMC_Articles/author_manuscript/"+"txt"+"/metadata/csv/author_manuscript.filelist.csv")
        self.files_metadata_phe_timebound=pd.read_csv("/data/projects/santal/data/PMC_Articles/phe_timebound/"+ "txt"+"/metadata/csv/phe_timebound.filelist.csv")

        self.oa_comm_files=[os.path.join(self.ROOT_PATH, self.files_metadata_oa_comm["Key"][i]) for i in range(len(self.files_metadata_oa_comm))]
        self.oa_noncomm_files=[os.path.join(self.ROOT_PATH, self.files_metadata_oa_noncomm["Key"][i]) for i in range(len(self.files_metadata_oa_noncomm))]
        self.author_manuscript_files=[os.path.join(self.ROOT_PATH, self.files_metadata_author_manuscript["Key"][i]) for i in range(len(self.files_metadata_author_manuscript))]
        self.phe_timebound_files=[os.path.join(self.ROOT_PATH, self.files_metadata_phe_timebound["Key"][i]) for i in range(len(self.files_metadata_phe_timebound))]
        
        if where=='all':
            self.files=self.oa_comm_files + self.oa_noncomm_files + self.author_manuscript_files + self.phe_timebound_files
        elif where.lower()=="commercial":
            self.files=self.oa_comm_files
        elif where.lower()=="non-commercial":
            self.files=self.oa_noncomm_files
        elif where.lower()=="author manscripts" or where.lower()=="author-manscripts" or "author" in where.lower():
            self.files=self.author_manuscript_files
        else:
            self.files=self.phe_timebound_files
    
    def __len__(self):
        return len(self.files)
    
            
    def parallel_term_search(self):
        num_cores = mp.cpu_count()
        print(num_cores)
        pool = mp.Pool(processes=num_cores-10)

        progress_bar = tqdm(total=len(self.files),desc='searching in '+str(len(self.files))+" files")
        items=[(self.search_pattern, file) for file in self.files]
        
        pattern=self.search_pattern
        # Apply the function to each file path in parallel
        results = []
        for result in pool.imap_unordered(search_string, items):
            progress_bar.update(1)  # Update progress bar
            if result is not None:
                results.append(result)
        #results=pool.map(search_file,files)

        pool.close()
        #progress_bar.close()
        return results

def main():
    santal=Santal()
    #bacteria_to_search_df=pd.read_csv("/home/aberhe/Projects/SANTAL/ASLR/data/Bacteria_to_search_with_higher_taxonomy.csv")
    bacteria_to_search_df=pd.read_csv(os.path.join("/home/aberhe/Projects/SANTAL/ASLR/data/from_predomics_models/Species_Patterns_to_search.csv"))

    list_bacteria=['Streptococcus_downei','Clostridium_perfringens','Veillonella_ratti','Fusobacterium_gonidiaformans','Gordonibacter_pamelaeae','Haemophilus_sputorum','Clostridium_hiranonis','Rothia_aeria','Clostridium_spiroforme','Bacillus_megaterium']
    for i,bacteria in enumerate(list_bacteria):#bacteria_to_search_df["pattern"].tolist()):
        #bacteria=bacteria.replace("_",".")
        if not os.path.exists("/home/aberhe/Projects/SANTAL/ASLR/Results/Included_articles/Species/"+bacteria+".txt"):
            with open("/home/aberhe/Projects/SANTAL/ASLR/Results/Included_articles/Species/"+bacteria+".txt", "w") as fw:
                fw.write("")
            fw.close()
            pattern=bacteria.replace("_",".")
            print("Bacteria: {}".format(bacteria.replace(".","_")))
            print()
            """
            species_splt=bacteria.split("_")
            if len(species_splt)>2:
                pattern= species_splt[-1]
            else:
                pattern= species_splt[0]
            """
            santal.search_pattern=pattern
            res=santal.parallel_term_search()

            print("FILES FOUND:",len(res))
            res = [f + '\n' for f in res]
            #if file_type=='txt':
            with open("/home/aberhe/Projects/SANTAL/ASLR/Results/Included_articles/Species/"+santal.search_pattern+".txt", "w") as fw:
                fw.writelines(res)
        else:
            print(f'{i}: {bacteria} already processed')
            continue
         
        
if __name__=="__main__":
    main()