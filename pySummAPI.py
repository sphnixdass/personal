import time
from os import listdir
from os.path import isfile, join
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import threading
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
from heapq import nlargest
import ChequeReadMainV2
import keras
from keras import layers

model_name = "D:\\Application\\SCEPOC\\Model\\T5\\"
path = "D:\\Application\\SCEPOC\\PythonAPI\\Data"
model_namesq = "D:\\Application\\SCEPOC\\Model\\Squad2Large"


model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
device = torch.device('cpu')

modelCheque = keras.models.load_model("D:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodel")
modelCheque2 = keras.models.load_model("D:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodelfont2")

text ="""
The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
"We'll be the comeback kids, all of us," he said. "We want to get our country back."
The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
"""


print("Library loaded")


# a) Get predictions
# nlp = pipeline('question-answering', model=model_namesq, tokenizer=model_namesq)



def SummAbstrative (inputtext, outputfilename):
    text = listToString(inputtext)
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    # print ("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50000,
                                        early_stopping=False)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print ("\n\nSummarized text: \n",summary_ids)
    with open(outputfilename, 'w') as f:
        f.write(output)


def listToString(s):
 
    # initialize an empty string
    str1 = " "
 
    # return string
    return (str1.join(s))


def ThreadCalling(inputfile, outputfile, src_path):
	with open(inputfile, 'r', errors='ignore') as f:
		inputtext = f.readlines()
		print("Calling Summ fucntion")
		ExtractiveMethod(inputtext, src_path)
		SummAbstrative (inputtext, outputfile)
		print("Summ function executed")
		with open(src_path.replace("_Action_PySummAbstractive.txt", "_PySummAbstractiveOutput.txt"), 'w') as f:
			f.write("Dass")

def ThreadCallingOnly(inputfile, outputfile, src_path):
    inputtext = ""
    with open(inputfile, 'r', errors='ignore') as f:
        inputtext = f.readlines()
        print("Calling Summ fucntion")
		
    SummAbstrative (inputtext, outputfile)
    print("Summ function executed")
    with open(src_path.replace("_Action_PySummAbstractiveOnly.txt", "_PySummAbstractiveOutputOnly.txt"), 'w') as f:
        f.write("Dass")

# def listToString(s):
 
#     # initialize an empty string
#     str1 = " "
 
#     # return string
#     return (str1.join(s))
     

def ExtractiveMethod(inputtext, src_path):
    try:

        # nltk.download("stopwords")
        inputper = 0.3
        with open(src_path.replace("_Action_PySummAbstractive.txt", "_PySummParam.txt"), 'r', errors='ignore') as f:
            inputper = f.readlines()

        stop_words = stopwords.words('english')

        punctuation = '!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~' + '\n'
        #punctuation = '.' + '\n'
        # print(punctuation)
        # print("condentsssssss" + str(inputper))
        inputtext = listToString(inputtext)
        inputtext = inputtext.replace('\n', ' ').replace('\r', '')
        # print(inputtext)
        # print("condentsssssss")

        #contentstxt = re.sub('[^A-Za-z0-9.? %!@#$%^&*()\\r\\n{}"]+', ' ', str(inputtext))
        contentstxt = re.sub('[^A-Za-z0-9.? %!@#$%^&*()\\r\\n{}"]+', ' ', str(inputtext))
        #contentstxt = str(inputtext)

        #js.document.getElementById('textainput').value = contentstxt

        #with open('article.txt') as f:
        #    contentstxt = f.readlines()

        # print(contents)
        contents = re.sub("[^0-9a-zA-Z][.]\n\r", " ", str(contentstxt))
        contents = contents.replace('"', ' ')
        contents = contents.replace('  ', ' ')
        #contents = contentstxt
        print("RRRRRRRRRRRRRRRRRRRR" + contents)
        tokens = word_tokenize(contents)


        word_frequencies = {}
        for word in tokens:
            if word.lower() not in stop_words:
                if word.lower() not in punctuation:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        # print(word_frequencies)


        max_frequency = max(word_frequencies.values())
        # print(max_frequency)


        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word]/max_frequency

        # print(word_frequencies)

        sent_token = sent_tokenize(contents)
        # print(sent_token)


        sentence_scores = {}
        for sent in sent_token:
            sentence = sent.split(" ")
            for word in sentence:
                if word.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.lower()]

        # print(sentence_scores)

        #select_length = int(len(sent_token)*0.3)
        select_length = int(len(sent_token)*float(inputper[0]))
        print(select_length)
        print(inputper)
        summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)


        def getList(dict, dictper):
            list = []
            for key in dict.keys():
                if int(dict[key]) >= dictper:
                    list.append(key)
                    #list.append(key + "(" + str(format(float(dict[key]),'.2f')) + ")" )
            return list


        outputsumm = getList(sentence_scores, select_length)


        final_summary = [word for word in summary]
        summary = ' '.join(final_summary)
        summary2 = ' '.join(outputsumm)

        #print(contents)
        print("==============End of Content==============")

        print(summary)
        print("==============End of Summarization==============")
        #print("Content Length : " + str(len(contents)))
        #print("Summarization Length : " + str(len(summary)))
        outputvar = "\\n\\n\\nBefore Summarization Text Length : " + str(len(contents)) + "\\nAfter Summarization Text Length : " + str(len(summary))
        print(type(summary2))
        print(summary2)
        print("After summarization: " + summary)
        with open(src_path.replace("_Action_PySummAbstractive.txt", "_PySummExtractiveOutput.txt"), 'w') as f:
            f.write(summary)
        with open(src_path.replace("_Action_PySummAbstractive.txt", "_PySummExtractiveOutput2.txt"), 'w') as f:
            f.write(contents)
    except Exception as error:
        print("An exception occurred:", error)

def QA_Model(inputfile, inputfiletext, outputfile):
    
    try:
        inputtext = ""
        inputContenttext = ""

        with open(inputfile, 'r', errors='ignore') as f:
            inputtext = f.readlines()
        inputtext = listToString(inputtext)
        
        with open(inputfiletext, 'r', errors='ignore') as f:
            inputContenttext = f.readlines()
        inputContenttext = listToString(inputContenttext)

        QA_input = {
        'question': inputtext,
        'context': inputContenttext
        }

        res = nlp(QA_input)

        print(res)
        #print(res[0])
        print(inputContenttext)
        print(res['answer'])
        with open(outputfile, 'w') as f:
            if(float(res['score']) > 0.01):
                f.write("Score : " + str(res['score']) + " Ans: " + (res['answer']))
            else:
                f.write("Score : " + str(res['score']) + " Ans: I am still learning and not able to understand your question. " + (res['answer']))
    except Exception as error:
        print("An exception occurred:", error)

def on_created(src_path):
    print(f"hey, {src_path} has been created!")
    filename = os.path.basename(src_path).split('/')[-1]
    if "_Action_" in filename:
        x = filename.split("_")
        print(x[2])
        os.remove(src_path)
        if x[2] == "PySummAbstractive.txt":
            print("Hi dass")
            inputfile = src_path.replace("_Action_PySummAbstractive.txt", "_Input.txt")
            outputfile = src_path.replace("_Action_PySummAbstractive.txt", "_Output.txt")
            
            src_path = src_path
            thr = threading.Thread(target=ThreadCalling, args=(inputfile, outputfile,src_path, ), kwargs={})
            thr.start() # Will run "foo"
            print("Function called")
#            with open(inputfile, 'r', errors='ignore') as f:
#                inputtext = f.readlines()
#                SummAbstrative (inputtext, outputfile)
#                with open(event.src_path.replace("_Action_PySummAbstractive.txt", "_Action_PySummAbstractiveOutput.txt"), 'w') as f:
#                    f.write("Dass")


        elif x[2] == "PySummAbstractiveOnly.txt":
            print("QA called")
            inputfile = src_path.replace("_Action_PySummAbstractiveOnly.txt", "_Input.txt")
            outputfile = src_path.replace("_Action_PySummAbstractiveOnly.txt", "_Output.txt")
            
            src_path = src_path
            thr = threading.Thread(target=ThreadCallingOnly, args=(inputfile, outputfile,src_path, ), kwargs={})
            thr.start() # Will run "foo"
            print("Function called")
            
        # elif x[2] == "PyQA.txt":
        #     print("QA called")
        #     inputfile = src_path.replace("_Action_PyQA.txt", "_QAInput.txt")
        #     inputfiletext = src_path.replace("_Action_PyQA.txt", "_PyQAContent.txt")
        #     outputfile = src_path.replace("_Action_PyQA.txt", "_QAOutput.txt")
        #     QA_Model(inputfile, inputfiletext, outputfile)
        #     with open(src_path.replace("_Action_PyQA.txt", "_PyQAOutput.txt"), 'w') as f:
        #         f.write("Dass")

        elif x[2] == "ChequeRead.txt":
            print("ChequeRead called")
            ChequeReadMainV2.CheqeuMain(modelCheque, modelCheque2)
            # inputfile = src_path.replace("_Action_PyQA.txt", "_QAInput.txt")
            # inputfiletext = src_path.replace("_Action_PyQA.txt", "_PyQAContent.txt")
            # outputfile = src_path.replace("_Action_PyQA.txt", "_QAOutput.txt")
            # QA_Model(inputfile, inputfiletext, outputfile)
            with open(src_path.replace("_Action_ChequeRead.txt", "_ChequeReadOutput.txt"), 'w') as f:
                f.write("Dass")
        

        print(x[1])


    print(os.path.basename(src_path).split('/')[-1]) 

def on_deleted(event):
    print(f"what Someone deleted {event.src_path}!")
    print(os.path.basename(event.src_path).split('/')[-1]) 

def on_modified(event):
    print(f"hey buddy, {event.src_path} has been modified")

def on_moved(event):
    print(f"ok ok ok, someone moved {event.src_path} to {event.dest_path}")



if __name__ == "__main__":
    # patterns = ["*"]
    # ignore_patterns = None
    # ignore_directories = False
    # case_sensitive = True
    # my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

    # my_event_handler.on_created = on_created
    # # my_event_handler.on_deleted = on_deleted
    # # my_event_handler.on_modified = on_modified
    # # my_event_handler.on_moved = on_moved

    
    # go_recursively = True
    # my_observer = Observer()
    # my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    # my_observer.start()
    
    while True:
        time.sleep(.5)
        for x in os.listdir(path):
            if x.endswith(".txt"):
    # Prints only text file present in My Folder
                #print(x)
                if "_Action_" in x:
                    try:
                        on_created(path + "\\" + x)
                    except Exception as error:
                        print("An exception occurred:", error)

    
        # my_observer.stop()
        # my_observer.join()
     