# Package that agg all data preprocess functions

import pandas as pd 
import numpy as np 
import torch
import collections
import itertools
import random

# Function that builds our vocabulary
def build_vocab(tokens, vocab_size):
    word_frequency = collections.Counter(tokens)

    vocab = {token: index for index, (token, _) in enumerate(
        word_frequency.most_common(vocab_size))}

    # Adicionamos o token "<unk>" para lidar com palavras não presentes no
    # vocabulário . O dataset text8 já contem este token, mas pode ser que ele
    # não tenha sido adicionado quando filtramos com `vocab_size`.
    if '<unk>' not in vocab:
        vocab['<unk>'] = len(vocab)

    # Adicionando o padding como um token
    if '<pad>' not in vocab:
        vocab['<pad>'] = len(vocab)
        
    return vocab 

# Function that generate n grams https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
def get_ngrams(tokens, vocab ,n = 5):
    '''
    Função que recebe uma lista de tokens e retorna os índices, de acordo com o vocab, dos n-grams de tamanho n com base no vocab passado

    '''

    # Primeiro adicionamos n-1 <pad> no começo dos tokens
    local_tokens = (n-1)*['<pad>'] + tokens.copy() 

    # Array que guardará o vetor de contextos
    context_df = []
    # Array que guardará os tokens target para cada contexto
    target_token = []

  # Para cada token de 0 a len(tokens) - n - 1(já que vamos usar até i+n tokens por contexto e o i+n+1 é o target)
    for i in range(len(local_tokens) - n):
    # Vetor auxiliar que será incrementado ao context_df
        aux_df = []

        # Loop que percorre os primeiros n tokens
        for j in range(i, i+n):
            if(local_tokens[j] not in vocab):
                aux_df.append(vocab["<unk>"])
            else:
                aux_df.append(vocab[local_tokens[j]])

        # Incrementa o context_df
        context_df.append(aux_df)

        # Incrementa o target_token com o i+n+1 token
        if(local_tokens[i+n] not in vocab):
            target_token.append(vocab["<unk>"])
        else:
            target_token.append(vocab[local_tokens[i+n]])

    # Retorno numpy arrays por comodidade minha, mas poderiam ser tensores direto
    return context_df, target_token

# Function that generate features, targets and authors
def generate_df(df, vocab, vocab_t, len_context = 5):
    results_list = []
    target_list = []
    author_list = []
    
    i = 0
    for text, author in zip(df.text, df.author):
#         print([text],author)
#         print((author+' '+text).split())
        encoded = get_ngrams((author+' '+text).split(), vocab ,n = len_context)
#         print([text],encoded[0])
#         i+=1
#         if(i>10):
#             break
        
        results_list.extend(encoded[0])
        target_list.extend(encoded[1])
        author_list.extend(len(encoded[1])*[vocab[author]])
        
    return results_list, target_list, author_list


def tokens2word(list_tokens, vocab_transposed):
    '''
        This function transform a list of tokens ids to string with the correspondent phrase.
    '''
    
    return ' '.join([vocab_transposed[t][0] for t in list_tokens])