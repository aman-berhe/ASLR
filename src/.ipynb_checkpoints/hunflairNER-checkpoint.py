from flair.data import Sentence
from flair.nn import Classifier

import json
import pandas as pd
import os
import artcileDetail_xml as ad
#from articleDetails import ArticleDetails
#import param
import nltk 

class FlairBasedNER():
	def __init__(self,modelName="hunflair"):
		self.sentence=None
		if modelName=="hunflair":
			self.tagger=Classifier.load("hunflair")
		else:
			self.tagger=SequenceTagger.load(modelName)

	def tagSentence(self, sentence):
		self.sentence=sentence
		self.sentence=Sentence(self.sentence)
		self.tagger.predict(self.sentence)

		#return self.sentence

	def tagDocument(self,document):
		#sentenceSpliter=SciSpacySentenceSplitter()
		self.sentence=nltk.sent_tokenize(document)
        self.sentence=Sentence(self.sentence)
		self.tagger.predict(self.sentence)

		#return self.sentence

	def saveExtractNER(self,outputFile="", outputFormat="csv",saveResult=1):
		"""docstring for ExtractNER"""
		if outputFormat=="csv":
			df=pd.DataFrame(columns=("sentence","entity","start_chunck","end_chunk","label","score"))

			for labels in sentence.get_labels():
                print(la.value,la.data_point.text,la.data_point.start_position,la.data_point.end_position)
                taggedEntities=[sentence.tokenized,labels.data_point.text,labels.data_point.start_position,labels.data_point.end_position,labels.value,round(labels.score,3)]
                df.loc[len(df)]=taggedEntities
			if saveResult:
				df.to_csv(outputFile+".csv")
			else:
				return df
		
		elif outputFormat=="json":
			jsonDataL=[]
			for sentTag in self.sentence:
			    s=sentTag.to_original_text()
			    jsonData={"setence":s,"entities":[]}
			    entityDicL=[]
			    for spanTag in sentTag.get_spans():
			        #taggedEntities=[s,spanTag.text,spanTag.start_pos,spanTag.end_pos,spanTag.tag,round(spanTag.score,3)]
			        entityDic={"text":spanTag.text,"start_chunk":spanTag.start_pos, "end_chunk":spanTag.end_pos,"label":spanTag.tag,"score":spanTag.score}
			        entityDicL.append(entityDic)
			        jsonData['entities'].append(entityDic)
			    jsonDataL.append(jsonData)
			if saveResult:
				with open(outputFile+'.json', 'w') as fout:
				    json.dump(jsonDataL, fout)
			else:
				return jsonDataL