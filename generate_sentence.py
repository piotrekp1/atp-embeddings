import numpy as np
import pandas as pd


def generate_sentence_from_word(df_scores, starting_word, length=10):
    sentence = [starting_word]
    word = starting_word
    while len(sentence) < length:
        df_node = df_scores[df_scores['prem'] == word]
        if df_node.shape[0] == 0:
            return sentence
        word = np.random.choice(df_node['conj'], p=df_node['prob'])
        sentence.append(word)
    return sentence


def generate_sentence(df_scores, initial_distribution, length=1000):  # no default length boundaries
    word = np.random.choice(initial_distribution.keys(), p=initial_distribution.values)
    return generate_sentence_from_word(df_scores, word, length)


def create_associations(sentence, window_size):
    associations = []
    for i, word in enumerate(sentence):
        for ass_word in sentence[i - window_size:i] + sentence[i+1:i+window_size + 1]:
            associations.append([word, ass_word])
    return associations


DATASET_SIZE = 10 ** 7

if __name__ == '__main__':
    df_scores = pd.read_csv('data/prems_mc.csv')
    init_distr = df_scores['prem'][~df_scores['prem'].isin(df_scores['conj'])].value_counts(normalize=True)

    big_data = []
    i = 0
    while len(big_data) < DATASET_SIZE:
        data = []
        while len(data) < DATASET_SIZE / 10:
            sentence = generate_sentence(df_scores, init_distr)
            data += create_associations(sentence, 3)
            if i % 200 == 0:
                print(len(data))
            i += 1
        big_data += data
        pd.DataFrame(big_data).to_csv('data/associations.csv', index=False)
