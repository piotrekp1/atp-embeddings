import numpy as np


def generate_sentence_from_word(df_scores, starting_word, length=10):
    sentence = [starting_word]
    while len(sentence) < length:
        df_node = df_scores[df_scores['prem'] == word]
        if df_node.shape[0] == 0:
            return sentence
        word = np.random.choice(df_node['conj'], p=df_node['prob'])
        sentence.append(word)
    return sentence


def generate_sentence(df_scores, df_scores_start, length=10):
    initial_distribution = df_scores_start['prem'].value_counts(normalize=True)
    word = np.random.choice(initial_distribution.keys(), p=initial_distribution.values)
    return generate_sentence_from_word(df_scores, word, length)
