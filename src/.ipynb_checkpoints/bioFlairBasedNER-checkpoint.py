from flair.data import Sentence
from flair.models import SequenceTagger,MultiTagger
from flair.tokenization import SciSpacySentenceSplitter,SciSpacyTokenizer

import json
import pandas as pd
import os
from articleDetails import ArticleDetails
#import param
import nltk 

class FlairBasedNER():
	def __init__(self,modelName="hunflair"):
		self.sentence=None
		if modelName=="hunflair":
			self.tagger=MultiTagger.load(modelName)
		else:
			self.tagger=SequenceTagger.load(modelName)

	def tagSentence(self, sentence):
		self.sentence=sentence
		self.sentence=Sentence(self.sentence,SciSpacyTokenizer)
		self.tagger.predict(self.sentence)

		#return self.sentence

	def tagDocument(self,document):
		sentenceSpliter=SciSpacySentenceSplitter()
		self.sentence=sentenceSpliter.split(document)
		self.tagger.predict(self.sentence)

		#return self.sentence

	def saveExtractNER(self,outputFile="", outputFormat="csv",saveResult=0):
		"""docstring for ExtractNER"""
		if outputFormat=="csv":
			df=pd.DataFrame(columns=("sentence","entity","start_chunck","end_chunk","label","score"))

			for sentTag in self.sentence:
				s=sentTag.to_original_text()
				for spanTag in sentTag.get_spans():
					taggedEntities=[s,spanTag.text,spanTag.start_pos,spanTag.end_pos,spanTag.tag,round(spanTag.score,3)]
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
				