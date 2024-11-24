from bs4 import BeautifulSoup
import os
import re
import json
import sys
import lxml

class ArticleDetails:
    def __init__(self, filepath):
        self.metadata=""
        self.title=""
        self.abstract=""
        self.keywords=""
        self.sections=""
        self.references=""
        if filepath.endswith(".xml"):
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = f.read()
                #print("FILE READ!!")
                self.soup=BeautifulSoup(data, 'lxml')
            else:
                print(filepath)
                print("ERROR: FILE DOES NOT EXIST. MAKE SURE YOU PROVIDED THE CORRECT FILE!!")
        else:
            print ("Please use the xml format of the file")
        
    def getTitle(self):
        """
        Uses th soup of the class and returns the title of the article
        It returns list of titles if the title exists in mutiple languages
        """
        title_group=self.soup.find("title-group")
        #print(tg)
        title_tags=title_group.find_all()
        titles=[]
        if title_tags!=[]:
            for tgg in title_tags:
                if tgg.name=="article-title":
                    titles.append(tgg.text)
                else:
                    if tgg.get('xml:lang')!=None:
                        titles.append(tgg.text+ " ["+tgg.get('xml:lang')+"]")
        return titles

    def getMetaData(self):
        """
        Takes the soup object and gets the details of authors of the article

        Retruns a dictionary of authors and their roles, affiliations, authord ids and year of publication
        """
        authorDetails={"Authors":[],"Affiliations":[],"Author-ID":[],"Role":[]}
        for cont in self.soup.find_all("contrib-group"):
            if cont !=[]:
                ref_authors=[]
                for names in cont.find_all("name"):
                    #print(names.find("surname").getText(),names.find("given-names").getText(),end=", ")
                    ref_authors.append(names.find("surname").getText()+" "+names.find("given-names").getText())
                    authorDetails["Authors"].append(names.find("surname").getText()+" "+names.find("given-names").getText())
                if cont.find("role"):
                    authorDetails["Role"].append(cont.find("role").getText())

            #authorDetails["Authors"].append(ref_authors)
        authorDetails["Year"]=self.soup.find_all("article-meta")[0].find("year").getText()
        for afff in self.soup.find_all("article-meta"):
            if afff!=[]:
                for aff in afff.find_all("aff"):
                    authorDetails["Affiliations"].append(aff.getText())
                for conid in afff.find_all("contrib-id"):
                    authorDetails["Author-ID"].append(conid.getText())

        return authorDetails

    def getAbstTags(self):
        """
        Takes the soup (BeuatifulSoup object) and gets the set of tags inside an abstarct. This is designed to get all sections in an article. 
        However there all articles might not have the same sections.

        Returns sa set of tags inside an abstarct which are the different sections of the abstarcu. like objective, methods, results, etc.
        """
        tags=[]
        tags_lits=self.soup.find_all("abstract")[0].find_all()
        if tags_lits !=[]:
            for tag in tags_lits:
                tags.append(tag.name)

        return list(set(tags))

    def getAbstract(self):
        """
        Takes the soup (BeuatifulSoup object) and gets the abstract of an article
        It returns a dictionary of abstract with its different sections as keys and the paragraph inside them as their values.
        """
        abstract={}
        tags=self.getAbstTags()
        if 'sec' in tags or 'section' in tags:
            for absSec in self.soup.find("abstract").find_all('sec'):
                if absSec!=[]:
                    secText=""
                    for parg in absSec.find_all('p'):
                        secText=secText+parg.getText()+"\n"
                    abstract[absSec.find('title').getText().replace(":","")]=secText

            return abstract
        else:
            return self.soup.find("abstract").getText()

    def getKeywords(self):
        keywords=[]
        try:
            for kwd in self.soup.find_all("kwd-group")[0].find_all("kwd"):
                for k in kwd:
                    keywords.append(k.getText())
        except:
            keywords.append("")

        return keywords

    def getSections(self):
        """
        Takes the soup (BeuatifulSoup object) and gets the body of an article
        It returns a dictionary of abstract with its different sections as keys and the paragraph inside them as their values.
        """
        sections={}
        if self.soup.find_all('body')!=[]:
            for sec in self.soup.find_all('body')[0].find_all("sec"):
                secText=""
                if sec!=[]:
                    for parg in sec.find_all("p"):
                        secText=secText+parg.getText()+"\n"
                    if sec.find("title")!=None:
                        sections[sec.find("title").getText()]=secText
        return sections

    def getReference(self):
        referenceDetail={"Title":[],"Authors":[],"Year":[],"Publication-Type":[],"Pub-ID":[]}
        for front in self.soup.find_all("ref"):
            try:
                ref_authors=[]
                for names in front.find_all("name"):
                    #print(names.find("surname").getText(),names.find("given-names").getText(),end=", ")
                    ref_authors.append(names.find("surname").getText()+" "+names.find("given-names").getText())
                referenceDetail["Authors"].append(ref_authors)
                referenceDetail["Title"].append(front.find("article-title").getText())
                referenceDetail["Year"].append(front.find("year").getText())
                referenceDetail["Publication-Type"].append(front.find("element-citation").get_attribute_list("publication-type")[0])
                referenceDetail["Pub-ID"].append(front.find("pub-id").getText())
            except:
                continue
        return referenceDetail
    
    def getDetails(self):
        self.title=self.getTitle()
        self.abstract=self.getAbstract()
        self.keywords=self.getKeywords()
        self.sections=self.getSections()
        self.metadata=self.getMetaData()
        self.references=self.getReference()
        
    def getArticleContent(self):
        self.articleContent=self.getTitle()[0]+'\n'
        
        abst=self.getAbstract()
        cont_joined=""
        for key in abst.keys():
            cont_joined=cont_joined+self.abstract[key]+'\n'
        
        self.articleContent=self.articleContent+cont_joined
        
        sections=self.getSections()
        cont_joined=""
        for key in sections.keys():
            cont_joined=cont_joined+self.sections[key]+'\n'
            
        self.articleContent=self.articleContent+cont_joined
        
        
        
            
        