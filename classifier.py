import nltk
from nltk.corpus import stopwords
from nltk.lm.preprocessing import flatten, pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import KneserNeyInterpolated, Laplace, WittenBellInterpolated
from nltk.util import bigrams,trigrams
from nltk.util import everygrams
from sklearn.model_selection import train_test_split
import string, sys
from sklearn.metrics import classification_report
#from transformers import pipeline
import argparse
import random
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score



# Load data function
def load_data(file_list):
    austen = ''
    dickens = ''
    tolstoy = ''
    wilde = ''
    for file in file_list:
        #print(f"Checking file: {file}")
        if file == 'austen_utf8.txt':
            #print("Found Austen file")
            with open(file, 'r') as f:
                austen = f.readlines()
            #print("Austen file",austen[0:10])
        elif file == 'dickens_utf8.txt':
            #print("Found Dickens file")
            with open(file, 'r') as f:
                dickens = f.readlines()
            #print("Read Dickens file",dickens[0:10])
        elif file == 'tolstoy_utf8.txt':
            #print("Found Tolstoy file")
            with open(file, 'r') as f:
                tolstoy = f.readlines()
            #print("Read Tolstoy file",tolstoy[0:10])
        elif file =='wilde_utf8.txt':
            print("Found Wilde file")
            with open(file, 'r') as f:
                wilde = f.readlines()
            #print("Read Wilde file",wilde[0:10])
        else:
            print(f"{file} is not found")
                
    return austen, dickens, tolstoy, wilde

def generative_approach(train,test,s,author):
    
    train_data,vocab = padded_everygram_pipeline(3, train)
    if s == 'k':
        model = KneserNeyInterpolated(3)
    elif s == 'w':
        model = WittenBellInterpolated(3)
    else:
        model = Laplace(3) 
    #model = Laplace(3)
    model.fit(train_data, vocab)
    #test_text=list(flatten(pad_both_ends(sent,n=3)for sent in test))
    #test_data1=list(bigrams(list(pad_both_ends(test_text,n=3))))
    #test_data1=list(trigrams(list(pad_both_ends(test,n=3))))
    #print("Perplexity on the 10% split test data is :")    
    perplexity=model.perplexity(test)
    #if (s == 'k'):
            #print(author,"Perplexity:",perplexity,"Smoothing used: KneserNeyInterpolated")
    #elif (s == 'w'):
            #print(author,"Perplexity:",perplexity,"Smoothing used: WittenBellInterpolated")
    #else:
           #print(author,"Perplexity:",perplexity,"Smoothing used: Laplace")
    #prompt="Once upon a time"
    #print(f"Austen generated text: {model.generate(prompt, max_length=500)}")
    threshold=float(500.000)
    correct_predictions = 0
    total_predictions = 0
    failure_cases=[]
    for sentence in test:
        testdata=list(trigrams(list(pad_both_ends(sentence,n=3))))
        perplexity = model.perplexity(testdata)
        total_predictions += 1
        if perplexity < threshold:
            predicted_author = author
        else:
            if(len(failure_cases)<5):
                original_sentence = ' '.join(sentence)
                failure_cases.append(original_sentence)
                
            predicted_author = None
        if predicted_author == author:
             correct_predictions += 1
    accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0
    print(f"{author} Accuracy: {accuracy * 100:.2f}%")
    #print(f"Five failure cases are:{failure_cases}\n)")
    return accuracy
    


def discriminative_approach():
    
    print("Discriminative Task\n")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    
    # A big thank you to Zhongxing0129 for uploading the dataset!!
    train = load_dataset('Zhongxing0129/authorlist_train')
    test = load_dataset("Zhongxing0129/authorlist_test")
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tokenizer = roberta_tokenizer
    
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)
    
    tokenized_train = train.map(preprocess_function, batched=True)
    tokenized_test = test.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #accuracy = evaluate.load('accuracy')
    
    def compute_label_wise_accuracy(predictions, references, label_indices):
        
        label_wise_accuracy = {}
        for label, index in label_indices.items():
            # Select only the predictions and references for the current label
            relevant_predictions = predictions[references == index]
            relevant_references = references[references == index]
            
            # Calculate accuracy for the current label
            label_acc = accuracy_score(relevant_references, relevant_predictions)
            label_wise_accuracy[label] = label_acc
        return label_wise_accuracy

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        overall_accuracy = accuracy_score(labels, predictions)
        
        label_wise_accuracy = compute_label_wise_accuracy(predictions, labels, label2id)
    
        return {"overall_accuracy": overall_accuracy, **label_wise_accuracy}
    
    
    labels = ['Austen', 'Wilde','Tolstoy','Dickens']
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in id2label.items()}
    #print('id2label:', id2label)
    #print('label2id:', label2id)
    
    #Importing the model
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=len(labels), id2label=id2label, label2id=label2id)
    
    # https://huggingface.co/transformers/v4.4.2/main_classes/trainer.html#trainingarguments
    
    #Passing the training arguments
    training_args = TrainingArguments(
    output_dir='Ritwickban/Roberta_classifier',
    learning_rate=2e-5,
    per_device_train_batch_size=49,
    per_device_eval_batch_size=49,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,)
    
    #Instantiating the Trainer class
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train['train'],
    eval_dataset=tokenized_test['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,)
    
    #Training the model
    trainer.train()
    
    count=0
    i=0
    model_name="Roberta_author_Ritwick"
    tokenizer = tokenizer
    model = model
    validation=tokenized_test['train']
    while count!=5:
      text=validation['text'][i]
      inputs = tokenizer(text,return_tensors="pt")
      with torch.no_grad():
        inputs = inputs.to(device)
        logits = model(**inputs).logits
      predicted_class_id = logits.argmax().item()
      if predicted_class_id!=validation['label'][i]:
        print(text)
        print('Confidence score:',torch.nn.functional.softmax(logits,dim=1))
        print('Predict:',model.config.id2label[predicted_class_id],"->Actual:",model.config.id2label[validation['label'][i]])
        count+=1
      i+=1
    
    
    
    
    
    
    
    

def preprocess_text(text):
    text_data= ''.join(text)
    # Tokenize the text into words
    sentences = sent_tokenize(text_data)

  # Tokenize each sentence into words
    tokenized_text = [word_tokenize(sentence) for sentence in sentences]
    #tokenized_text_test = [word_tokenize(test_sentence) for test_sentence in test_sentence]
    return tokenized_text

def preprocess_test_text(test_data):
    tokenized_text = [word_tokenize(sentence) for sentence in test_data]
    return tokenized_text


def main(args):
    #Takes in the authorlist, reads the file and loads respective data
    austen = ""
    dickens = ""
    tolstoy = ""
    wilde = ""
    flag=0
    
    
    if args.approach == 'generative':
        with open(args.authorlist, 'r') as f:
            file_list = f.read().splitlines()

        austen, dickens, tolstoy, wilde = load_data(file_list)
        data = {"Austen": austen, "Dickens": dickens, "Tolstoy": tolstoy, "Wilde": wilde}

        if args.test:
            with open(args.test, 'r', encoding='utf-8') as f:
                test_data = f.readlines()
            
            flag=1
            print("Training Models.....")
            for author, works in data.items():
                #print(f"{preprocess_text(works)}")
                #break
                #print(f"Size of data for {author}: {len(works)}")
                if author=='Austen':
                    #print("Hitting the right spot")
                    tokenized_text_aust=preprocess_text(works)
                    train_data,vocab = padded_everygram_pipeline(3, tokenized_text_aust)
                    #print(works[0:100])
                    aust_mod= KneserNeyInterpolated(3)
                    aust_mod.fit(train_data,vocab)
                    print("Austen model Trained")
                elif author=='Dickens':
                    #print("Hitting the right spot")
                    tokenized_text_dick = preprocess_text(works)
                    train_data,vocab = padded_everygram_pipeline(3, tokenized_text_dick)
                    dick_mod= KneserNeyInterpolated(3)
                    dick_mod.fit(train_data,vocab)
                    print("Dickens model Trained")
                elif author=='Tolstoy':
                    #print("Hitting the right spot")
                    tokenized_text_tol = preprocess_text(works)
                    train_data,vocab = padded_everygram_pipeline(3, tokenized_text_tol)
                    tol_mod= KneserNeyInterpolated(3)
                    tol_mod.fit(train_data,vocab)
                    print("Tolstoy model Trained")
                elif author=='Wilde':
                    #print("Hitting the right spot")
                    tokenized_text_wild = preprocess_text(works)
                    train_data,vocab = padded_everygram_pipeline(3, tokenized_text_wild)
                    wild_mod= KneserNeyInterpolated(3)
                    wild_mod.fit(train_data,vocab)
                    print("Wilde model Trained")
                else:
                    print('Author doesnt match')
            #Test Dataset
            print("Sentence-wise Classification")
            #sentences=test_data.split('\n\n')
            #test_text = ''.join(test_data)
            #sentences = test_text.split('\n\n')
            tokenized_text_test = preprocess_text(test_data)
            #tokenized_text_test=preprocess_text(test_data)
            #print(sentences1[0:100])
            #print(test_data[0:100])
            #print(tokenized_text_test[0:10])
            #test_text=list(flatten(pad_both_ends(sent,n=2)for sent in tokenized_text_test))
            #test_data1=list(bigrams(list(pad_both_ends(test_text,n=2))))
            '''t= "family may be. We ought to be acquainted with Enscombe."
            s1 = sent_tokenize(t)
            tt = [word_tokenize(sentence) for sentence in s1]
            print(tt)
            td=list(flatten(pad_both_ends(sent,n=2)for sent in tt))
            td1=list(bigrams(list(pad_both_ends(td,n=2))))
            print(td)
            print(td1)
            print(f"Perplexity: {aust_mod.perplexity(td1)})")
            print(f"Perplexity: {dick_mod.perplexity(td1)})")
            print(f"Perplexity: {tol_mod.perplexity(td1)})")
            print(f"Perplexity: {wild_mod.perplexity(td1)})")'''
            #print(tokenized_text_test[0:5])
            #print(test_text[0:10])
            #print(test_data1[0:10])
            for i, sentence_tokens in enumerate(tokenized_text_test, start=1):
                
                #test_text=list(flatten(pad_both_ends(sent,n=2)for sent in sentence_tokens))
                test_data1=list(trigrams(list(pad_both_ends(sentence_tokens,n=3))))
                #line=" ".join(sentence_tokens)
                #print(" ".join(sentence_tokens)[:100])  # Printing first 100 characters of the sentence
                aust_perplexity = aust_mod.perplexity(test_data1)
                dick_perplexity = dick_mod.perplexity(test_data1)
                tol_perplexity = tol_mod.perplexity(test_data1)
                wild_perplexity = wild_mod.perplexity(test_data1)
                min_perplexity = min(aust_perplexity, dick_perplexity, tol_perplexity, wild_perplexity)
                
                if min_perplexity == aust_perplexity:
                    min_author = "Austen"
                elif min_perplexity == dick_perplexity:
                    min_author = "Dickens"
                elif min_perplexity == tol_perplexity:
                    min_author = "Tolstoy"
                elif min_perplexity == wild_perplexity:
                    min_author = "Wilde"
                    
                print(f"Sentence {i}: {min_author}")
                #print(f"{min_author}")
            #print(vocab[0:100])
            prompt1=[["Once"],["Hello"],["I"],["They"],["You"]]
            #prompt2=["Hello"]
            #prompt3=["I"]
            #prompt4=["Actually"]
            #prompt5=["You"]
            for i in range(5):
                gt1 = aust_mod.generate(15, text_seed=prompt1[i], random_seed=3)
              
            # Join the generated words into a string
                gts1 = ' '.join(gt1)
            # Print the generated text
                td=list(trigrams(list(pad_both_ends(gt1,n=3))))
                perp1a=aust_mod.perplexity(td)
                perp1d=dick_mod.perplexity(td)
                perp1t=tol_mod.perplexity(td)
                perp1w=wild_mod.perplexity(td)
                print("\n")
                print(f"Austen Generated Text: {gts1}, \n Austen Perplexity:{perp1a}; \nDickens Perplexity:{perp1d}; \nTolstoy Perplexity:{perp1t}; \nWilde Perplexity:{perp1w}")
                
                
                gt2 = dick_mod.generate(15, text_seed=prompt1[i], random_seed=3)
            # Join the generated words into a string
                gts2 = ' '.join(gt2)
            # Print the generated text
                td1=list(trigrams(list(pad_both_ends(gt2,n=3))))
                perp2a=aust_mod.perplexity(td1)
                perp2d=dick_mod.perplexity(td1)
                perp2t=tol_mod.perplexity(td1)
                perp2w=wild_mod.perplexity(td1)
                print("\n")
                print(f"Dickens Generated Text: {gts2} \nAusten Perplexity:{perp2a}; \nDickens Perplexity:{perp2d}; \nTolstoy Perplexity:{perp2t}; \nWilde Perplexity:{perp2w}")
                gt3 = tol_mod.generate(15, text_seed=prompt1[i], random_seed=3)
            # Join the generated words into a string
                gts3 = ' '.join(gt3)
            # Print the generated text
                td2=list(trigrams(list(pad_both_ends(gt3,n=3))))
                perp3a=aust_mod.perplexity(td2)
                perp3d=dick_mod.perplexity(td2)
                perp3t=tol_mod.perplexity(td2)
                perp3w=wild_mod.perplexity(td2)
                print("\n")
                print(f"Tolstoy Generated Text:{gts3} \nAusten Perplexity:{perp3a}; \nDickens Perplexity:{perp3d}; \nTolstoy Perplexity:{perp3t}; \nWilde Perplexity:{perp3w}")
                gt4 = wild_mod.generate(15, text_seed=prompt1[i], random_seed=3)
            # Join the generated words into a string
                gts4 = ' '.join(gt4)
            # Print the generated text
                td3=list(trigrams(list(pad_both_ends(gt4,n=3))))
                perp4a=aust_mod.perplexity(td3)
                perp4d=dick_mod.perplexity(td3)
                perp4t=tol_mod.perplexity(td3)
                perp4w=wild_mod.perplexity(td3)
                print("\n")
                print(f"Wilde Generated Text: {gts4} \nAusten Perplexity:{perp4a}; \nDickens Perplexity:{perp4d}; \nTolstoy Perplexity:{perp4t}; \nWilde Perplexity:{perp4w}")
                print("\n")
                print("\n")
                #print(f"Austen generated text: {aust_mod.generate(prompt, num_tokens=500)}")
                #dev_predictions = generative_approach(works, test_data,'k',flag,author) 
        else:
                 for author, works in data.items():
                     print(f"Size of data for {author}: {len(works)}")
                     #print(f"{author} has perplexity")
                     train_data, test_data = train_test_split(works, test_size=0.1, random_state=42)
                     accuracies = []
                     #accuracy=
                     train=preprocess_text(train_data)
                     test=preprocess_text(test_data)
                     accuracy=generative_approach(train, test,'k',author)
                     accuracies.append((author, accuracy))
                 #print("\nModel Evaluation:")
                 #for author, accuracy in accuracies:
                    #print(f"{author} Accuracy: {accuracy * 100:.2f}%")
    elif args.approach == 'discriminative':
                discriminative_approach()
                
    else:
        
        print("Sorry!")
        '''
        for author, works in data.items():
                print(f"Size of data for {author}: {len(works)}")
                #print(f"{author} has perplexity")
                train_data, test_data = train_test_split(works, test_size=0.1, random_state=42)
                accuracies = []
                #accuracy=
                train=preprocess_text(train_data)
                test=preprocess_text(test_data)
                accuracy=generative_approach(train, test,'k',author)
                accuracies.append((author, accuracy))
            #print("\nModel Evaluation:")
            #for author, accuracy in accuracies:
               #print(f"{author} Accuracy: {accuracy * 100:.2f}%")'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Author Classification')
    parser.add_argument('authorlist', type=str, help='File containing list of author files')
    parser.add_argument('-approach', choices=['generative', 'discriminative'], required=True, help='Classification approach')
    parser.add_argument('-test', type=str, help='Test file for classification')
    args = parser.parse_args()
    main(args)



