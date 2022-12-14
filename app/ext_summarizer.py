from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from fastapi.logger import logger

async def read_article_file(file):
    contents = await file.read()
    logger.info('Provided file')
    logger.info(f'content: {contents}')
    text = contents.decode("utf-8")
    article = text.split(". ")

    sentences = []
    for sentence in article:
      print(sentence)
      sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    
    return sentences

def read_article(text):
    article = text.split(". ")
    sentences = []
    
    for sentence in article:
      print(sentence)
      sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

async def generate_summary(text, file, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text and split it
    if((text is not None) and (file is None)):
        sentences = read_article(text)
    elif((text is None) and (file is not None)):
        sentences =  await read_article_file(file)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]) + ".")

    # Step 5 - Offcourse, output the summarize text
    summarized = " ".join(summarize_text)
    
    return(summarized)