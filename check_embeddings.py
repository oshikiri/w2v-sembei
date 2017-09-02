import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import pprint


def get_topn_df(vectors, query_list, topn=5):

    vectors = pd.DataFrame(normalize(vectors.values), index=vectors.index)

    dict_sims = dict()
    for query in query_list:
        if query not in vectors.index:
            print('"{0}" is not in `vectors.index`'.format(query))
            continue

        similarities = pd.Series(vectors.values @ vectors.loc[query, :].values, index=vectors.index)
        similarities_topn = similarities.sort_values(ascending=False).index[1:(topn + 1)]
        dict_sims[query] = '  '.join(similarities_topn.tolist())

    return dict_sims

if __name__ == '__main__':
    with open('./output/vocabulary.csv') as f:
        vocabulary = [v.replace('\n', '') for v in f.readlines()]

    df_embeddings = pd.read_csv('./output/embeddings_words.csv', header=None, sep=' ')
    vectors = pd.DataFrame(df_embeddings.values, index=vocabulary)
    vocabulary_choice = np.random.choice(vocabulary, size=20, replace=False).tolist()
    vocabulary_choice = ['英語', '漫画', '言語', 'スペイン', '科学',
                         '俳優', '手塚治虫', '理由', '状態', '夏',
                         'であった', '会社', 'サッカー', '人物', 'Java']

    pp = pprint.PrettyPrinter()
    pp.pprint(get_topn_df(vectors, vocabulary_choice, topn=10))
