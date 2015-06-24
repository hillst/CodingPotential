#!/usr/bin/env python
from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN
from passage.utils import save, load
from random import shuffle

def prepare_fasta(filename, label):
    seqs = []
    seq = ""
    for line in open(filename,'r'):
        if ">" in line:
            if seq != "":
                seqs.append((seq, label))
            seq = ""
        else:
            seq += line.strip()
    seqs.append(seq)
    return seqs

def test():
    coding = prepare_fasta("TAIR10_pep_20101214.fa", 1)
    noncoding = prepare_fasta("NONCODEv4_tair.pep.fa", 0)
    shuffle(coding)
    shuffle(noncoding)

    train = coding[:len(coding)-100] + noncoding[:len(noncoding)-200]
    test = coding[len(coding)-100:] + noncoding[len(noncoding)-200:]
    
    train_text = [val[0] for val in train]
    train_labels = [val[1] for val in train]
    
    test_text = [val[0] for val in test]
    test_labels = [val[0] for val in test]

    tokenizer = Tokenizer(character=True)
    train_tokens = tokenizer.fit_transform(train_text)

    layers = [
        Embedding(size=128, n_features=tokenizer.n_features),
        GatedRecurrent(size=128),
        Dense(size=1, activation='sigmoid')
    ]

    model = RNN(layers=layers, cost='BinaryCrossEntropy')
    model.fit(train_tokens, train_labels)

    model.predict(tokenizer.transform(test_text))
    save(model, 'save_test.pkl')
    #model = load('save_test.pkl') 
    
    """
    Our evaluation (crude)
    """
    correct, incorrect = 0,0
    interesting_genes = []
    for idx, prediction in enumerate(model.predict(tokenizer.transform(test_text))):
        print round(prediction), test_labels[idx]
        if round(prediction) == test_labels[idx]:
            correct += 1
        else:
            interesting_genes.append(idx)
            incorrect += 1
    print correct, incorrect, correct / float(correct + incorrect)
    fd = open('interesting.fa', 'w')
    for gene in interesting_genes:
        print >> fd, ">gene"
        print >> fd, test_text[gene]

def main():
    print "loading model"
    model = load('save_test.pkl')
    
    print "preparing data"
    
    print "testing data"
    tokenizer = Tokenizer(character=True)
    train_tokens = tokenizer.fit_transform(train_text)
    correct, incorrect = 0,0
    interesting_genes = []
    for idx, prediction in enumerate(model.predict(tokenizer.transform(test_text))):
        print round(prediction), test_labels[idx]
        if round(prediction) == test_labels[idx]:
            correct += 1
        else:
            interesting_genes.append(idx)
            incorrect += 1
    print correct, incorrect, correct / float(correct + incorrect)
    fd = open('interesting.fa', 'w')
    for gene in interesting_genes:
        print >> fd, ">gene"
        print >> fd, test_text[gene]
    
        

if __name__ == "__main__":
    test()
    #main()
