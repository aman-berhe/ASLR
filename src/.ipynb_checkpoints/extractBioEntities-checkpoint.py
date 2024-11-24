import json
import pandas as pd
import os
from tqdm import tqdm
#from bioFlairBasedNER import FlairBasedNER
from hunflairNER import FlairBasedNER
import artcileDetail_xml as ad
import param

def main():
    #path_to_included_articles="Datasets/Articles/Predomics/Included Articles"
    included_articles="/home/aberhe/Projects/SANTAL/ASLR/data/Veillonella_articles_found_test.txt"
    path_to_results="/home/aberhe/Projects/SANTAL/ASLR/Results/BioNER/"
    with open("/home/aberhe/Projects/SANTAL/ASLR/data/Veillonella_articles_found_exits.txt", "r") as f:
        files_veillonella=f.readlines()
    for article in tqdm(files_veillonella):#(path_to_included_articles+'/'+signature)):
        #articlePath=path_to_included_articles+'/'+signature+'/'+ article
        articlePath="/data/projects/santal/data/PMC_Articles/oa_comm/xml/all/"+article.split("/")[-1].replace(".txt",".xml")
        print(article.split("/")[-1].replace(".txt",".xml"))
        print(articlePath,os.path.exists(articlePath))
        #outputPath=path_to_results+'/'+signature+'/'+article.split(".")[0] 
        outputPath=path_to_results+article.split("/")[-1].split(".txt")[0]
        if not os.path.exists(outputPath+"bioEntity.csv"):
            #try:
                #file=open(articlePath)
                #jsonData=json.load(file)
                #print("{}: LOADED".format(articlePath))


                #Get details of the article; i.e; abstarct, fullarticle, tittle, etc.
            articleDetails=ad.ArticleDetails(articlePath)
            articleDetails.getDetails()

            #Extract entities inside the abstract of the articles
            #fbNER=FlairBasedNER()
            #fbNER.sentence=articleDetails
            fbNER.tagDocument(articleDetails.abstract)
            fbNER.saveExtractNER(outputFile=outputPath+"abstract",outputFormat="csv",saveResult=1)

            #Extract entities from the full article
            #fbNER2=FlairBasedNER()
            fbNER.tagDocument(articleDetails.sections)
            fbNER.saveExtractNER(outputFile=outputPath+"fullArticle",outputFormat="csv",saveResult=1)
            #except:
            #continue

if __name__=="__main__":
	main()
