import numpy as np
import pandas as pd
import csv
import nltk
from nltk.tokenize import word_tokenize
import string

print("reading data")
#reading data from file
temp=pd.read_csv("train2.csv",header=None,low_memory=False)


data=[]


print("filling array")

#comments id
data.append([])
for i in range(1,len(list(temp.iterrows()))):
    data[0].append(temp.ix[i][0])


print("filling array 1")
#comment text
data.append([])
for i in range(1,len(list(temp.iterrows()))):
    data[1].append(temp.ix[i][1])

print("filling array 2")
#toxic
data.append([])
for i in range(1,len(list(temp.iterrows()))):
    data[2].append(temp.ix[i][2])

print("filling array 3")
#severe toxic
data.append([])
for i in range(1,len(list(temp.iterrows()))):
    data[3].append(temp.ix[i][3])


print("filling array 4")
#obscene
data.append([])
for i in range(1,len(list(temp.iterrows()))):
    data[4].append(temp.ix[i][4])


print("filling array 5")
#threat
data.append([])
for i in range(1,len(list(temp.iterrows()))):
    data[5].append(temp.ix[i][5])


print("filling array 6")
#insult
data.append([])
for i in range(1,len(list(temp.iterrows()))):
    data[6].append(temp.ix[i][6])


print("filling array 7")
#identity hate
data.append([])
for i in range(1,len(list(temp.iterrows()))):
    data[7].append(temp.ix[i][7])



# print("printing data")
# print(data[5])

toxic_prob=0
for i in range(len(data[2])):
    toxic_prob+=int(data[2][i])
# print("Probability of toxicity: ")
toxic_prob=float(toxic_prob)/100.0
toxic_prob=toxic_prob-0.1

print("toxic prob",toxic_prob)


severe_toxic_prob=0
for i in range(len(data[3])):
    severe_toxic_prob+=int(data[3][i])
# print("Probability of severe toxicity: ")
severe_toxic_prob=float(severe_toxic_prob)/100.0
print("severe toxic prob",severe_toxic_prob)

obscene_prob=0
for i in range(len(data[4])):
    obscene_prob+=int(data[4][i])
# print("Probability of obscene: ")
obscene_prob=float(obscene_prob)/100.0
print("obscene prob",obscene_prob)
obscene_prob=obscene_prob-0.02
threat_prob=0
for i in range(len(data[5])):
    threat_prob+=int(data[5][i])
# print("Probability of threat: ")
threat_prob=float(threat_prob)/100.0
print("threat prob",threat_prob)

insult_prob=0
for i in range(len(data[6])):
    insult_prob+=int(data[6][i])
# print("Probability of insult: ")
insult_prob=float(insult_prob)/100.0
print("insult prob",insult_prob)

identify_hate_prob=0
for i in range(len(data[4])):
    identify_hate_prob+=int(data[4][i])

# print("Probability of hate: ")
identify_hate_prob=float(identify_hate_prob)/100.0
print("identify prob",identify_hate_prob)

##============================PRE PROCESSING FOR GENERATING VOCAB=============================

vocalbulary=set()

com=""
for i in range(len(data[1])):
    com+=data[1][i]


temp=[]

temp=word_tokenize(com)

# for i in temp:
#   print(i)

print(len(temp))

##============================STOP WORDS=============================

from nltk.corpus import stopwords


stop_words=set(stopwords.words("english"))
stop_words.add("the")
stop_words.add(",")
stop_words.add(".")
stop_words.add("(")
stop_words.add(")")
stop_words.add("?")
stop_words.add("!")
stop_words.add("'")
stop_words.add("-")
stop_words.add("/")
stop_words.add(";")
stop_words.add(":")
stop_words.add(">")
stop_words.add("#")
stop_words.add("@")
stop_words.add("|")
stop_words.add("[")
stop_words.add("]")
stop_words.add("''")
stop_words.add("I")
stop_words.add("``")
stop_words.add("If")
stop_words.add("The")
#stop_words.add("n't")






filtered_words=[item for item in temp if not item in stop_words]

print(len(filtered_words))


##============================STEMMING==============================

# from nltk.stem.snowball import SnowballStemmer

# stemmer = SnowballStemmer("english")

# new_words=[]
# for w in filtered_words:
#   new_words.append(stemmer.stem(w))


##============================LEMMATIZER==============================

from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

new_words=[]
for w in filtered_words:
    new_words.append(lemmatizer.lemmatize(w))



for i in new_words:
    vocalbulary.add(i)

print(len(vocalbulary))

vocab_size=len(vocalbulary)

##============================FREQUENCY DISTRIBUTION==============================

all_words=nltk.FreqDist(new_words)


##===========================================================================




num_words_toxic=0
for i in range(len(data[0])):
    if int(data[2][i])==1:
        num_words_toxic+=len(word_tokenize(data[1][i]))

num_words_severe_toxic=0
for i in range(len(data[0])):
    if(int(data[3][i])==1):
        num_words_severe_toxic+=len(word_tokenize(data[1][i]))
    
num_words_obscene=0
for i in range(len(data[0])):
    if(int(data[4][i])==1):
        num_words_obscene+=len(word_tokenize(data[1][i]))

num_words_threat=0
for i in range(len(data[0])):
    if(int(data[5][i])==1):
        num_words_threat+=len(word_tokenize(data[1][i]))

num_words_insult=0
for i in range(len(data[0])):
    if(int(data[6][i])==1):
        num_words_insult+=len(word_tokenize(data[1][i]))

num_words_identify_hate=0
for i in range(len(data[0])):
    if(int(data[7][i])==1):
        num_words_identify_hate+=len(word_tokenize(data[1][i]))


temp2=pd.read_csv("test2.csv",header=None,low_memory=False)




testData=[]


print("filling test2 array")
i=0

#comments id
testData.append([])
for i in range(1,len(list(temp2.iterrows()))):
    testData[0].append(temp2.ix[i][0])

#comments text
testData.append([])
for i in range(1,len(list(temp2.iterrows()))):
    testData[1].append(temp2.ix[i][1])


print("done")

arr=[]
j=0
for i in range(len(testData[0])):
    arr.append([testData[0][i]])
    


csvfile = "outputFile.csv"

c=0
print("done2")




comm_toxic_val=toxic_prob
comm_severe_toxic_val=severe_toxic_prob
comm_obscene_val=obscene_prob
comm_threat_val=threat_prob
comm_insult_val=insult_prob
comm_identify_hate_val=identify_hate_prob
word=""
this_word_in_toxic=0
this_word_in_severe_toxic=0
this_word_in_obscene=0
this_word_in_threat=0
this_word_in_insult=0
this_word_in_identify_hate=0
second_parameter=0

arr2=[]
for a in range (0, len(testData[0])):
    
    print("new comment")
    temp_comment=testData[1][a]
    tok_temp_comment=word_tokenize(temp_comment)
    for i in range (0, len(tok_temp_comment)):
        word=tok_temp_comment[i]
        for j in range (0, len(data[0])):
            if(int(data[2][j])==1):
                tok_this_comment=word_tokenize((data[1][j]))
                for k in range (0, len(tok_this_comment)):
                    if(tok_this_comment[k]==word):
                        this_word_in_toxic=this_word_in_toxic+1

            num=(this_word_in_toxic+1)/((vocab_size)+(num_words_toxic))
        comm_toxic_val=(comm_toxic_val)*(num)
        
    




    i, j, k, num=0, 0, 0, 0

    for i in range (0, len(tok_temp_comment)):
        word=tok_temp_comment[i]
        for j in range (0, len(data[0])):
            if(int(data[3][j])==1):
                tok_this_comment=word_tokenize((data[1][j]))
                for k in range (1, len(tok_this_comment)):
                    if(tok_this_comment[k]==word):
                        this_word_in_severe_toxic=this_word_in_severe_toxic+1

            num=(this_word_in_severe_toxic+1)/((vocab_size)+(num_words_severe_toxic))
        comm_severe_toxic_val=(comm_severe_toxic_val)*(num)
        

    




    i, j, k, num=0, 0, 0, 0

    for i in range (0, len(tok_temp_comment)):
        word=tok_temp_comment[i]
        for j in range (0, len(data[0])):
            if(int(data[4][j])==1):
                tok_this_comment=word_tokenize((data[1][j]))
                for k in range (1, len(tok_this_comment)):
                    if(tok_this_comment[k]==word):
                        this_word_in_obscene=this_word_in_obscene+1

            num=(this_word_in_obscene+1)/((vocab_size)+(num_words_obscene))
        comm_obscene_val=(comm_obscene_val)*(num)
        



    i, j, k, num=0, 0, 0, 0


    for i in range (0, len(tok_temp_comment)):
        word=tok_temp_comment[i]
        for j in range (1, len(data[0])):
            if(int(data[5][j])==1):
                tok_this_comment=word_tokenize((data[1][j]))
                for k in range (1, len(tok_this_comment)):
                    if(tok_this_comment[k]==word):
                        this_word_in_threat=this_word_in_threat+1

            num=(this_word_in_threat+1)/((vocab_size)+(num_words_threat))
        comm_threat_val=(comm_threat_val)*(num)
        




    i, j, k, num=0, 0, 0, 0



    for i in range (0, len(tok_temp_comment)):
        word=tok_temp_comment[i]
        for j in range (1, len(data[0])):
            if(int(data[6][j])==1):
                tok_this_comment=word_tokenize((data[1][j]))
                for k in range (1, len(tok_this_comment)):
                    if(tok_this_comment[k]==word):
                        this_word_in_insult=this_word_in_insult+1

            num=(this_word_in_insult+1)/((vocab_size)+(num_words_insult))
        comm_insult_val=(comm_insult_val)*(num)
        






    i, j, k, num=0, 0, 0, 0


    for i in range (0, len(tok_temp_comment)):
        word=tok_temp_comment[i]
        for j in range (1, len(data[0])):
            if(int(data[7][j])==1):
                tok_this_comment=word_tokenize((data[1][j]))
                for k in range (1, len(tok_this_comment)):
                    if(tok_this_comment[k]==word):
                        this_word_in_identify_hate=this_word_in_identify_hate+1

            num=(this_word_in_identify_hate+1)/((vocab_size)+(num_words_identify_hate))
        comm_identify_hate_val=(comm_identify_hate_val)*(num)
    
    values=[]
    values.append(comm_toxic_val)
    values.append(comm_severe_toxic_val)
    values.append(comm_obscene_val)
    values.append(comm_threat_val)
    values.append(comm_insult_val)
    values.append(comm_identify_hate_val)

    b=0
    maximum=1
    while b<6:
        if values[b]>=values[maximum]:
            maximum=b
        b+=1

    print("new comment processed")
    
    if (maximum)==0:
        arr1=[testData[0][a],1,0,0,0,0,0]
        arr2.append(arr1)
        
    elif (maximum==1):
        arr1=[testData[0][a],0,1,0,0,0,0]
        arr2.append(arr1)
        
    elif (maximum==2):
        arr1=[testData[0][a],0,0,1,0,0,0]
        arr2.append(arr1)
        
    elif (maximum==3):
        arr1=[testData[0][a],0,0,0,1,0,0]
        arr2.append(arr1)
        
    elif (maximum==4):
        arr1=[[testData[0][a],0,0,0,0,1,0]]
        arr2.append(arr1)
        
    elif (maximum==5):
        arr1=[[testData[0][a],0,0,0,0,0,1]]
        arr2.append(arr1)
        

    print(a)

with open(csvfile, "w") as output:
    writer = csv.writer(output)
    if(c==0):
        writer.writerow(["ID", "Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"])
    c+=1
    writer.writerows(arr2)

