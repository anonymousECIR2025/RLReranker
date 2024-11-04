import os 
import pandas as pd
from pathlib import Path
from loguru import logger
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder

from .helper_functions import set_manual_seed

set_manual_seed()


device = 'cuda:0'

def load_queries(queries_path):

    queries = {}
    with open(queries_path, 'r', encoding='utf8') as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query.strip()
    return queries


def load_corpus(corpus_file_path):

    corpus = {}
    with open(corpus_file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage.strip()

    return corpus

def get_vector(row, queries, corpus, cross_encoder, model):

    qid, doc_id = row[0], row[1] #str, str

    text = f'{queries[qid]}[SEP]{corpus[doc_id]}'
    tokens = cross_encoder.tokenizer(text, return_tensors="pt").to(device)
    vector = model(**tokens).pooler_output.detach().cpu().numpy().tolist()[0]
    return vector


def encode_date(queries, corpus, df, base_language_model_path):

    cross_encoder = CrossEncoder(base_language_model_path, device=device)
    model = cross_encoder.model.bert.to(device)

    enc = df.apply(get_vector, args=(queries, corpus, cross_encoder, model), axis=1, result_type='expand')
    
    df = pd.concat([df, enc], axis=1)

    return df

def load_dataset(cfg, stage= None) -> pd.DataFrame:

    if not stage:
        stage = cfg.stage
    if stage == 'TRAIN':
        input_file_path, df_path, queries_path = cfg.train_set_path, cfg.train_df_path, cfg.train_queries_path
    elif stage == "DL19":
        input_file_path, df_path, queries_path = cfg.dl19_test_set_path, cfg.dl19_test_df_path, cfg.test_queries_path
    elif stage == "DL20":
        input_file_path, df_path, queries_path = cfg.dl20_test_set_path, cfg.dl20_test_df_path, cfg.test_queries_path
    if not os.path.exists(df_path):

        logger.info(f"Loading from input_file_path:{input_file_path}")

        df = pd.read_csv(input_file_path, names = list(cfg.run_params.columns))

        df["qid"] = df["qid"].astype(str)
        df["doc_id"] = df["doc_id"].astype(str)

        queries = load_queries(queries_path)
        corpus = load_corpus(cfg.corpus_file_path)

        common_qid = set(queries.keys()).intersection(set(df["qid"].values))

        df = encode_date(queries, corpus, df, cfg.cross_encoder_path)

        df.columns = ['qid', 'doc_id', 'relevance'] + [str(i) for i in range(1, 769)]

        df.to_csv(df_path)

    else:

        logger.info(f"Loading from df_path:{df_path}")
        df = pd.read_csv(df_path, nrows=5000)  
        print(f"Number of unique qids in the df:{len(list(df['qid'].unique()))}")
        df["qid"] = df["qid"].astype(str)
        df["doc_id"] = df["doc_id"].astype(str)

        df = df.sort_values(["qid", "relevance"], ascending=False)

        logger.info(f"{stage}\n{df.head(5)}")

    logger.info(df.info())
    logger.info(f"Columns in the loaded dataset are :{df.columns}")
    return df


def get_features(qid, doc_id, features, dataset) -> List[float]:

    qid, doc_id = str(qid), str(doc_id)

    df = dataset[(dataset["doc_id"].str.contains(doc_id)) & (dataset["qid"] == qid)]
    assert len(df) != 0, "Fix the dataset"

    if 120 < len(df.columns) < 200:
        vector_size = 128
    elif 200 < len(df.columns) < 300:
        vector_size = 256
    elif 300 < len(df.columns) < 400:
        vector_size = 384
    elif 500 <  len(df.columns) < 900:
        vector_size = 768
    else:
        vector_size = 1024

    relevant_columns = [f"{i}" for i in range(1, vector_size+1)]

    if features:
        relevant_columns = relevant_columns + features

    return df[relevant_columns].values.tolist()[0]

def get_query_features(qid, doc_list, features, dataset) -> np.ndarray:
    """
    Get query features for the given query ID, list of docs, and dataset.
    """
    doc_set = set(doc_list)
    qid = str(qid)
    if len(doc_list) > 0:
        df = dataset[dataset["qid"] == qid]
        df = df[df["doc_id"].isin(doc_set)]
    else:
        df = dataset[dataset["qid"] == qid]
    assert len(df) != 0

    if 120 < len(df.columns) < 200:
        vector_size = 128
    elif 200 < len(df.columns) < 300:
        vector_size = 256
    elif 300 < len(df.columns) < 400:
        vector_size = 384
    elif 500 <  len(df.columns) < 900:
        vector_size = 768
    else:
        vector_size = 1024

    
    valid_columns = [str(x) for x in range(1,vector_size+1)]
    if features:
        valid_columns = valid_columns + features

    
    df = df.set_index('doc_id').loc[doc_list].reset_index()
    relevance_list = df["relevance"].values
    df = df[valid_columns]

    return df.values, relevance_list


def get_model_inputs(state, action, features, dataset, normalize=False) -> np.ndarray:
    temp = [state.t] + get_features(state.qid, action, features, dataset)
    temp = [float(x) for x in temp]

    temp_array = np.array(temp)
    
    if normalize:
        min_val = temp_array.min()
        max_val = temp_array.max()
        if max_val - min_val > 0:
            temp_array = (temp_array - min_val) / (max_val - min_val)

    return np.array(temp_array)

def get_multiple_model_inputs(state, doc_list, features, dataset) -> np.ndarray:

    features, relevance_list = get_query_features(state.qid, doc_list, features, dataset)

    return np.insert(features, 0, state.t, axis=1), relevance_list





