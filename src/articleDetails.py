
import json
import os
import sys
import shutil
from datetime import date

class ArticleDetails:
	"""
	self.jsonData=None
	self.date=0000
	self.title=""
	self.abstract=""
	self.introduction=""
	self.result=""
	self.method=""
	self.fullArticle=""
	"""

	def __init__(self, jsonData):
		self.jsonData=jsonData
		self.date=0000
		self.title=""
		self.abstract=""
		self.keywords=""
		self.introduction=""
		self.result=""
		self.method=""
		self.fullArticle=""

	def getTitle(self):
		if self.jsonData['documents'][0]['passages'][0]['infons']['section_type']== 'TITLE':
			return self.jsonData['documents'][0]['passages'][0]['text']
		else:
		    return "No Title"

	def getAuthors(self):
	    authorList=[]
	    for k in self.jsonData['documents'][0]['passages'][0]['infons'].keys():
	        if 'name' in k:
	            namesData=self.jsonData['documents'][0]['passages'][0]['infons'][k].split(';')
	            fullname=""
	            for name in namesData:
	                fullname=fullname+' '+ name.split(':')[1]
	            authorList.append(fullname)
	        else:
	            continue
	    return authorList

	def getKeywords(self):
		keywords=self.jsonData['documents'][0]["passages"][0]["infons"]['kwd'].split()
		return keywords

	def getDate(self):
	    year=int(self.jsonData['date'][0:4])
	    month=int(self.jsonData['date'][4:6])
	    day=int(self.jsonData['date'][6:8])
	    publishedDate=date(year=year,month=month,day=day)
	    return publishedDate

	def getAbstract(self):
	    abstract=""
	    for j in range(len(self.jsonData['documents'][0]['passages'])):
	        if self.jsonData['documents'][0]['passages'][j]['infons']['section_type']=="ABSTRACT" and self.jsonData['documents'][0]['passages'][j]['infons']['type']=='abstract':
	            abstract=abstract+" "+self.jsonData['documents'][0]['passages'][j]['text']
	        else:
	            continue
	    return abstract

	def getParagrpahs(self):
	    contents=""
	    for j in range(len(self.jsonData['documents'][0]['passages'])):
	        if self.jsonData['documents'][0]['passages'][j]['infons']['type']=="paragraph":
	            contents=contents+"\n"+self.jsonData['documents'][0]['passages'][j]['text']
	    
	    return contents.strip()

	def getIntroduction(self):
	    intro=""
	    for j in range(len(self.jsonData['documents'][0]['passages'])):
	        if self.jsonData['documents'][0]['passages'][j]['infons']['section_type']=="INTRO" and self.jsonData['documents'][0]['passages'][j]['infons']['type']=='paragraph':
	            intro=intro+self.jsonData['documents'][0]['passages'][j]['text']+"\n"
	    
	    return intro

	def getResults(self):
	    results=""
	    for j in range(len(self.jsonData['documents'][0]['passages'])):
	        if self.jsonData['documents'][0]['passages'][j]['infons']['section_type']=="RESULTS" and self.jsonData['documents'][0]['passages'][j]['infons']['type']=='paragraph':
	            results=results+self.jsonData['documents'][0]['passages'][j]['text']+"\n"
	            
	    return results

	def getMethods(self):
	    methods=""
	    for j in range(len(self.jsonData['documents'][0]['passages'])):
	        if self.jsonData['documents'][0]['passages'][j]['infons']['section_type']=="METHODS" and self.jsonData['documents'][0]['passages'][j]['infons']['type']=='paragraph':
	            methods=methods+self.jsonData['documents'][0]['passages'][j]['text']+"\n"
	            
	    return methods

	def getDetails(self):
		self.title=self.getTitle()
		self.abstract=self.getAbstract()
		self.date=self.getDate()
		self.introduction=self.getIntroduction()
		self.result=self.getResults()
		self.method=self.getMethods()
		self.fullArticle=self.getParagrpahs()
		#.keywords=self.getKeywords()