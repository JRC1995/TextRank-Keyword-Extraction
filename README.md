
# Implementation of TextRank for keyword Extraction

Based on: 

[TextRank: Bringing Order into Texts - by Rada Mihalcea and Paul Tarau](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

The input text is given below


```python
#Source of text:
#https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents

Text = "Compatibility of systems of linear constraints over the set of natural numbers. \
Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and \
nonstrict inequations are considered. \
Upper bounds for components of a minimal set of solutions and \
algorithms of construction of minimal generating sets of solutions for all \
types of systems are given. \
These criteria and the corresponding algorithms for constructing \
a minimal supporting set of solutions can be used in solving all the \
considered types of systems and systems of mixed types."
```

### Cleaning Text Data

The raw input text is cleaned off non-printable characters (if any) and turned into lower case.
The processed input text is then tokenized using NLTK library functions. 


```python

import nltk
from nltk import word_tokenize
import string

#nltk.download('punkt')

def clean(text):
    text = text.lower()
    printable = set(string.printable)
    text = filter(lambda x: x in printable, text) #filter funny characters, if any.
    return text

Cleaned_text = clean(Text)

text = word_tokenize(Cleaned_text)

print "Tokenized Text: \n"
print text
```

    Tokenized Text: 
    
    ['compatibility', 'of', 'systems', 'of', 'linear', 'constraints', 'over', 'the', 'set', 'of', 'natural', 'numbers', '.', 'criteria', 'of', 'compatibility', 'of', 'a', 'system', 'of', 'linear', 'diophantine', 'equations', ',', 'strict', 'inequations', ',', 'and', 'nonstrict', 'inequations', 'are', 'considered', '.', 'upper', 'bounds', 'for', 'components', 'of', 'a', 'minimal', 'set', 'of', 'solutions', 'and', 'algorithms', 'of', 'construction', 'of', 'minimal', 'generating', 'sets', 'of', 'solutions', 'for', 'all', 'types', 'of', 'systems', 'are', 'given', '.', 'these', 'criteria', 'and', 'the', 'corresponding', 'algorithms', 'for', 'constructing', 'a', 'minimal', 'supporting', 'set', 'of', 'solutions', 'can', 'be', 'used', 'in', 'solving', 'all', 'the', 'considered', 'types', 'of', 'systems', 'and', 'systems', 'of', 'mixed', 'types', '.']


### POS Tagging For Lemmatization

NLTK is again used for <b>POS tagging</b> the input text so that the words can be lemmatized based on their POS tags.

Description of POS tags: 


http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html


```python
#nltk.download('averaged_perceptron_tagger')
  
POS_tag = nltk.pos_tag(text)

print "Tokenized Text with POS tags: \n"
print POS_tag
```

    Tokenized Text with POS tags: 
    
    [('compatibility', 'NN'), ('of', 'IN'), ('systems', 'NNS'), ('of', 'IN'), ('linear', 'JJ'), ('constraints', 'NNS'), ('over', 'IN'), ('the', 'DT'), ('set', 'NN'), ('of', 'IN'), ('natural', 'JJ'), ('numbers', 'NNS'), ('.', '.'), ('criteria', 'NNS'), ('of', 'IN'), ('compatibility', 'NN'), ('of', 'IN'), ('a', 'DT'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'), ('diophantine', 'NN'), ('equations', 'NNS'), (',', ','), ('strict', 'JJ'), ('inequations', 'NNS'), (',', ','), ('and', 'CC'), ('nonstrict', 'JJ'), ('inequations', 'NNS'), ('are', 'VBP'), ('considered', 'VBN'), ('.', '.'), ('upper', 'JJ'), ('bounds', 'NNS'), ('for', 'IN'), ('components', 'NNS'), ('of', 'IN'), ('a', 'DT'), ('minimal', 'JJ'), ('set', 'NN'), ('of', 'IN'), ('solutions', 'NNS'), ('and', 'CC'), ('algorithms', 'NN'), ('of', 'IN'), ('construction', 'NN'), ('of', 'IN'), ('minimal', 'JJ'), ('generating', 'VBG'), ('sets', 'NNS'), ('of', 'IN'), ('solutions', 'NNS'), ('for', 'IN'), ('all', 'DT'), ('types', 'NNS'), ('of', 'IN'), ('systems', 'NNS'), ('are', 'VBP'), ('given', 'VBN'), ('.', '.'), ('these', 'DT'), ('criteria', 'NNS'), ('and', 'CC'), ('the', 'DT'), ('corresponding', 'JJ'), ('algorithms', 'NN'), ('for', 'IN'), ('constructing', 'VBG'), ('a', 'DT'), ('minimal', 'JJ'), ('supporting', 'NN'), ('set', 'NN'), ('of', 'IN'), ('solutions', 'NNS'), ('can', 'MD'), ('be', 'VB'), ('used', 'VBN'), ('in', 'IN'), ('solving', 'VBG'), ('all', 'PDT'), ('the', 'DT'), ('considered', 'VBN'), ('types', 'NNS'), ('of', 'IN'), ('systems', 'NNS'), ('and', 'CC'), ('systems', 'NNS'), ('of', 'IN'), ('mixed', 'JJ'), ('types', 'NNS'), ('.', '.')]


### Lemmatization

The tokenized text (mainly the nouns and adjectives) is normalized by <b>lemmatization</b>.
In lemmatization different grammatical counterparts of a word will be replaced by single
basic lemma. For example, 'glasses' may be replaced by 'glass'. 

Details about lemmatization: 
    
https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html


```python
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

adjective_tags = ['JJ','JJR','JJS']

lemmatized_text = []

for word in POS_tag:
    if word[1] in adjective_tags:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
    else:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
        
print "Text tokens after lemmatization of adjectives and nouns: \n"
print lemmatized_text
```

    Text tokens after lemmatization of adjectives and nouns: 
    
    ['compatibility', 'of', 'system', 'of', 'linear', 'constraint', 'over', 'the', 'set', 'of', 'natural', 'number', '.', 'criterion', 'of', 'compatibility', 'of', 'a', 'system', 'of', 'linear', 'diophantine', 'equation', ',', 'strict', 'inequations', ',', 'and', 'nonstrict', 'inequations', 'are', 'considered', '.', 'upper', 'bound', 'for', 'component', 'of', 'a', 'minimal', 'set', 'of', 'solution', 'and', 'algorithm', 'of', 'construction', 'of', 'minimal', 'generating', 'set', 'of', 'solution', 'for', 'all', 'type', 'of', 'system', 'are', 'given', '.', 'these', 'criterion', 'and', 'the', 'corresponding', 'algorithm', 'for', 'constructing', 'a', 'minimal', 'supporting', 'set', 'of', 'solution', 'can', 'be', 'used', 'in', 'solving', 'all', 'the', 'considered', 'type', 'of', 'system', 'and', 'system', 'of', 'mixed', 'type', '.']


### POS tagging for Filtering

The <b>lemmatized text</b> is <b>POS tagged</b> here. The tags will be used for filtering later on.


```python
POS_tag = nltk.pos_tag(lemmatized_text)

print "Lemmatized text with POS tags: \n"
print POS_tag
```

    Lemmatized text with POS tags: 
    
    [('compatibility', 'NN'), ('of', 'IN'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'), ('constraint', 'NN'), ('over', 'IN'), ('the', 'DT'), ('set', 'NN'), ('of', 'IN'), ('natural', 'JJ'), ('number', 'NN'), ('.', '.'), ('criterion', 'NN'), ('of', 'IN'), ('compatibility', 'NN'), ('of', 'IN'), ('a', 'DT'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'), ('diophantine', 'JJ'), ('equation', 'NN'), (',', ','), ('strict', 'JJ'), ('inequations', 'NNS'), (',', ','), ('and', 'CC'), ('nonstrict', 'JJ'), ('inequations', 'NNS'), ('are', 'VBP'), ('considered', 'VBN'), ('.', '.'), ('upper', 'JJ'), ('bound', 'NN'), ('for', 'IN'), ('component', 'NN'), ('of', 'IN'), ('a', 'DT'), ('minimal', 'JJ'), ('set', 'NN'), ('of', 'IN'), ('solution', 'NN'), ('and', 'CC'), ('algorithm', 'NN'), ('of', 'IN'), ('construction', 'NN'), ('of', 'IN'), ('minimal', 'JJ'), ('generating', 'VBG'), ('set', 'NN'), ('of', 'IN'), ('solution', 'NN'), ('for', 'IN'), ('all', 'DT'), ('type', 'NN'), ('of', 'IN'), ('system', 'NN'), ('are', 'VBP'), ('given', 'VBN'), ('.', '.'), ('these', 'DT'), ('criterion', 'NN'), ('and', 'CC'), ('the', 'DT'), ('corresponding', 'JJ'), ('algorithm', 'NN'), ('for', 'IN'), ('constructing', 'VBG'), ('a', 'DT'), ('minimal', 'JJ'), ('supporting', 'NN'), ('set', 'NN'), ('of', 'IN'), ('solution', 'NN'), ('can', 'MD'), ('be', 'VB'), ('used', 'VBN'), ('in', 'IN'), ('solving', 'VBG'), ('all', 'PDT'), ('the', 'DT'), ('considered', 'VBN'), ('type', 'NN'), ('of', 'IN'), ('system', 'NN'), ('and', 'CC'), ('system', 'NN'), ('of', 'IN'), ('mixed', 'JJ'), ('type', 'NN'), ('.', '.')]


## POS Based Filtering

Any word from the lemmatized text, which isn't a noun, adjective, or gerund (or a 'foreign word'), is here
considered as a <b>stopword</b> (non-content). This is based on the assumption that usually keywords are noun,
adjectives or gerunds. 

Punctuations are added to the stopword list too.


```python
stopwords = []

wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW'] 

for word in POS_tag:
    if word[1] not in wanted_POS:
        stopwords.append(word[0])

punctuations = list(str(string.punctuation))

stopwords = stopwords + punctuations
```

### Complete stopword generation

Even if we remove the aforementioned stopwords, still some extremely common nouns, adjectives or gerunds may
remain which are very bad candidates for being keywords (or part of it). 

An external file constituting a long list of stopwords is loaded and all the words are added with the previous
stopwords to create the final list 'stopwords-plus' which is then converted into a set. 

(Source of stopwords data: https://www.ranks.nl/stopwords)

Stopwords-plus constitute the sum total of all stopwords and potential phrase-delimiters. 

(The contents of this set will be later used to partition the lemmatized text into n-gram phrases. But, for now, I will simply remove the stopwords, and work with a 'bag-of-words' approach. I will be developing the graph using unigram texts as vertices)

```python
stopword_file = open("long_stopwords.txt", "r")
#Source = https://www.ranks.nl/stopwords

lots_of_stopwords = []

for line in stopword_file.readlines():
    lots_of_stopwords.append(str(line.strip()))

stopwords_plus = []
stopwords_plus = stopwords + lots_of_stopwords
stopwords_plus = set(stopwords_plus)

#Stopwords_plus contain total set of all stopwords
```

### Removing Stopwords 

Removing stopwords from lemmatized_text. 
Processeced_text condtains the result.


```python
processed_text = []
for word in lemmatized_text:
    if word not in stopwords_plus:
        processed_text.append(word)
print processed_text
```

    ['compatibility', 'system', 'linear', 'constraint', 'set', 'natural', 'number', 'criterion', 'compatibility', 'system', 'linear', 'diophantine', 'equation', 'strict', 'inequations', 'nonstrict', 'inequations', 'upper', 'bound', 'component', 'minimal', 'set', 'solution', 'algorithm', 'construction', 'minimal', 'generating', 'set', 'solution', 'type', 'system', 'criterion', 'algorithm', 'constructing', 'minimal', 'supporting', 'set', 'solution', 'solving', 'type', 'system', 'system', 'mixed', 'type']


## Vocabulary Creation

Vocabulary will only contain unique words from processed_text.


```python
vocabulary = list(set(processed_text))
print vocabulary
```

    ['upper', 'set', 'constructing', 'number', 'solving', 'system', 'compatibility', 'strict', 'criterion', 'type', 'minimal', 'supporting', 'generating', 'linear', 'diophantine', 'component', 'bound', 'nonstrict', 'inequations', 'natural', 'algorithm', 'constraint', 'equation', 'solution', 'construction', 'mixed']


### Building Graph

TextRank is a graph based model, and thus it requires us to build a graph. Each words in the vocabulary will serve as a vertex for graph. The words will be represented in the vertices by their index in vocabulary list.  

The weighetd_edge matrix contains the information of edge connections among all vertices.
I am building a graph with wieghted undirected edges.

weighted_edge[i][j] contains the weight of the connecting edge between the word vertex represented by vocabulary index i and the word vertex represented by vocabulary j.

If weighted_edge[i][j] is zero, it means no edge or connection is present between the words represented by index i and j.

There is a connection between the words (and thus between i and j which represents them) if the words co-occur within a window of a specified 'window_size' in the processed_text.

I am increasing value of the weighted_edge[i][j] is increased by (1/(distance between positions of words currently represented by i and j)) for every connection discovered between the same words in different locations of the text. 

The covered_coocurrences list (which is contain the list of pairs of absolute positions in processed_text of the words whose coocurrence at that location is already checked) is managed so that the same two words located in the same positions in processed_text are not repetitively counted while sliding the window one text unit at a time.

The score of all vertices are intialized to one. 

Self-connections are not considered, so weighted_edge[i][i] will be zero.


```python
import numpy as np
import math
vocab_len = len(vocabulary)

weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)

score = np.zeros((vocab_len),dtype=np.float32)
window_size = 3
covered_coocurrences = []

for i in xrange(0,vocab_len):
    score[i]=1
    for j in xrange(0,vocab_len):
        if j==i:
            weighted_edge[i][j]=0
        else:
            for window_start in xrange(0,(len(processed_text)-window_size)):
                
                window_end = window_start+window_size
                
                window = processed_text[window_start:window_end]
                
                if (vocabulary[i] in window) and (vocabulary[j] in window):
                    
                    index_of_i = window_start + window.index(vocabulary[i])
                    index_of_j = window_start + window.index(vocabulary[j])
                    
                    # index_of_x is the absolute position of the xth term in the window 
                    # (counting from 0) 
                    # in the processed_text
                      
                    if [index_of_i,index_of_j] not in covered_coocurrences:
                        weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                        covered_coocurrences.append([index_of_i,index_of_j])

```

### Calculating weighted summation of connections of a vertex

inout[i] will contain the total no. of undirected connections\edges associated withe the vertex represented by i.


```python
inout = np.zeros((vocab_len),dtype=np.float32)

for i in xrange(0,vocab_len):
    for j in xrange(0,vocab_len):
        inout[i]+=weighted_edge[i][j]
```

### Scoring Vertices

The formula used for scoring a vertex represented by i is:

score[i] = (1-d) + d x [ Summation(j) ( (weighted_edge[i][j]/inout[j]) x score[j] ) ] where j belongs to the list of vertices that has a connection with i. 

d is the damping factor.

The score is iteratively updated until convergence. 


```python
MAX_ITERATIONS = 50
d=0.85
threshold = 0.0001 #convergence threshold

for iter in xrange(0,MAX_ITERATIONS):
    prev_score = np.copy(score)
    
    for i in xrange(0,vocab_len):
        
        summation = 0
        for j in xrange(0,vocab_len):
            if weighted_edge[i][j] != 0:
                summation += (weighted_edge[i][j]/inout[j])*score[j]
                
        score[i] = (1-d) + d*(summation)
    
    if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
        print "Converging at iteration "+str(iter)+"...."
        break

```

    Converging at iteration 29....



```python
for i in xrange(0,vocab_len):
    print "Score of "+vocabulary[i]+": "+str(score[i])
```

    Score of upper: 0.816792
    Score of set: 2.27184
    Score of constructing: 0.667288
    Score of number: 0.688316
    Score of solving: 0.642318
    Score of system: 2.12032
    Score of compatibility: 0.944584
    Score of strict: 0.823772
    Score of criterion: 1.22559
    Score of type: 1.08101
    Score of minimal: 1.78693
    Score of supporting: 0.653705
    Score of generating: 0.652645
    Score of linear: 1.2717
    Score of diophantine: 0.759295
    Score of component: 0.737641
    Score of bound: 0.786006
    Score of nonstrict: 0.827216
    Score of inequations: 1.30824
    Score of natural: 0.688299
    Score of algorithm: 1.19365
    Score of constraint: 0.674411
    Score of equation: 0.799815
    Score of solution: 1.6832
    Score of construction: 0.659809
    Score of mixed: 0.235822


### Phrase Partitioning

Paritioning lemmatized_text into phrases using the stopwords in it as delimeters.
The phrases are also candidates for keyphrases to be extracted. 


```python
phrases = []

phrase = " "
for word in lemmatized_text:
    
    if word in stopwords_plus:
        if phrase!= " ":
            phrases.append(str(phrase).strip().split())
        phrase = " "
    elif word not in stopwords_plus:
        phrase+=str(word)
        phrase+=" "

print "Partitioned Phrases (Candidate Keyphrases): \n"
print phrases
```

    Partitioned Phrases (Candidate Keyphrases): 
    
    [['compatibility'], ['system'], ['linear', 'constraint'], ['set'], ['natural', 'number'], ['criterion'], ['compatibility'], ['system'], ['linear', 'diophantine', 'equation'], ['strict', 'inequations'], ['nonstrict', 'inequations'], ['upper', 'bound'], ['component'], ['minimal', 'set'], ['solution'], ['algorithm'], ['construction'], ['minimal', 'generating', 'set'], ['solution'], ['type'], ['system'], ['criterion'], ['algorithm'], ['constructing'], ['minimal', 'supporting', 'set'], ['solution'], ['solving'], ['type'], ['system'], ['system'], ['mixed', 'type']]


### Create a list of unique phrases.

Repeating phrases\keyphrase candidates has no purpose here, anymore. 


```python
unique_phrases = []

for phrase in phrases:
    if phrase not in unique_phrases:
        unique_phrases.append(phrase)

print "Unique Phrases (Candidate Keyphrases): \n"
print unique_phrases
```

    Unique Phrases (Candidate Keyphrases): 
    
    [['compatibility'], ['system'], ['linear', 'constraint'], ['set'], ['natural', 'number'], ['criterion'], ['linear', 'diophantine', 'equation'], ['strict', 'inequations'], ['nonstrict', 'inequations'], ['upper', 'bound'], ['component'], ['minimal', 'set'], ['solution'], ['algorithm'], ['construction'], ['minimal', 'generating', 'set'], ['type'], ['constructing'], ['minimal', 'supporting', 'set'], ['solving'], ['mixed', 'type']]


### Thinning the list of candidate-keyphrases.

Removing single word keyphrase-candidates that are present multi-word alternatives. 


```python
for word in vocabulary:
    #print word
    for phrase in unique_phrases:
        if (word in phrase) and ([word] in unique_phrases) and (len(phrase)>1):
            #if len(phrase)>1 then the current phrase is multi-worded.
            #if the word in vocabulary is present in unique_phrases as a single-word-phrase
            # and at the same time present as a word within a multi-worded phrase,
            # then I will remove the single-word-phrase from the list.
            unique_phrases.remove([word])
            
print "Thinned Unique Phrases (Candidate Keyphrases): \n"
print unique_phrases    
```

    Thinned Unique Phrases (Candidate Keyphrases): 
    
    [['compatibility'], ['system'], ['linear', 'constraint'], ['natural', 'number'], ['criterion'], ['linear', 'diophantine', 'equation'], ['strict', 'inequations'], ['nonstrict', 'inequations'], ['upper', 'bound'], ['component'], ['minimal', 'set'], ['solution'], ['algorithm'], ['construction'], ['minimal', 'generating', 'set'], ['constructing'], ['minimal', 'supporting', 'set'], ['solving'], ['mixed', 'type']]


### Scoring Keyphrases

Scoring the phrases (candidate keyphrases) and building up a list of keyphrases
by listing untokenized versions of tokenized phrases\candidate-keyphrases.
Phrases are scored by adding the score of their members (words\text-units that were ranked by the graph algorithm)



```python
phrase_scores = []
keywords = []
for phrase in unique_phrases:
    phrase_score=0
    keyword = ''
    for word in phrase:
        keyword += str(word)
        keyword += " "
        phrase_score+=score[vocabulary.index(word)]
    phrase_scores.append(phrase_score)
    keywords.append(keyword.strip())

i=0
for keyword in keywords:
    print "Keyword: '"+str(keyword)+"', Score: "+str(phrase_scores[i])
    i+=1
```

    Keyword: 'compatibility', Score: 0.944583714008
    Keyword: 'system', Score: 2.12031626701
    Keyword: 'linear constraint', Score: 1.94610738754
    Keyword: 'natural number', Score: 1.37661552429
    Keyword: 'criterion', Score: 1.2255872488
    Keyword: 'linear diophantine equation', Score: 2.83080631495
    Keyword: 'strict inequations', Score: 2.13201224804
    Keyword: 'nonstrict inequations', Score: 2.135455966
    Keyword: 'upper bound', Score: 1.60279768705
    Keyword: 'component', Score: 0.737640619278
    Keyword: 'minimal set', Score: 4.05876886845
    Keyword: 'solution', Score: 1.68319940567
    Keyword: 'algorithm', Score: 1.19365406036
    Keyword: 'construction', Score: 0.659808635712
    Keyword: 'minimal generating set', Score: 4.71141409874
    Keyword: 'constructing', Score: 0.66728836298
    Keyword: 'minimal supporting set', Score: 4.71247345209
    Keyword: 'solving', Score: 0.642318367958
    Keyword: 'mixed type', Score: 1.31682945788


### Ranking Keyphrases

Ranking keyphrases based on their calculated scores. Displaying top 'keywords_num' no. of keyphrases.


```python
sorted_index = np.flip(np.argsort(phrase_scores),0)

keywords_num = 10

print "Keywords:\n"

for i in xrange(0,keywords_num):
    print str(keywords[sorted_index[i]])+", ",
```

    Keywords:
    
    minimal supporting set,  minimal generating set,  minimal set,  linear diophantine equation,  nonstrict inequations,  strict inequations,  system,  linear constraint,  solution,  upper bound, 


# Input:

Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types.

# Extracted Keywords:

* minimal supporting set,  
* minimal generating set,  
* minimal set,  
* linear diophantine equation,  
* nonstrict inequations,  
* strict inequations,  
* system,  
* linear constraint,  
* solution,  
* upper bound, 
