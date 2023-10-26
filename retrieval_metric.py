import numpy as np
from tqdm import tqdm


def average_precision_at_k(one_query_label, relevant_retrieved_labels):
    """ e.g., 
        one_query_label: 1, 
        relevant_retrieved_labels: [1,0,1,0,0,1,0,0,1,1] 
        average_precision_at_k = (1/1 + 2/3 + 3/6 + 4/9 + 5/10) * 1/5 
        """
    hits = 0
    precisions = []
    for i,label in enumerate(relevant_retrieved_labels):
        if label == one_query_label:
            hits += 1
            iterate_pos = i + 1
            precision = hits / (iterate_pos)
            precisions.append(precision)
    if hits > 0:
        average_precision = 0
        for i in precisions:
            average_precision += i
        average_precision /= hits
    else:
        average_precision = 0
    # print('hits:', hits, 'precisions:', precisions, 'average_precision:', average_precision)  ###
    return average_precision


# def mean_average_precision_at_k(query_labels, retrieved_labels):
#     mean_average_precision = 0
#     for i, q in tqdm(enumerate(query_labels)):
#         retrieved_labels_relevant_to_q = retrieved_labels[i]
#         average_precision = average_precision_at_k(q, retrieved_labels_relevant_to_q)
#         mean_average_precision += average_precision
    
#     mean_average_precision /= len(query_labels)
#     return mean_average_precision


def test():
    """ https://www.zhihu.com/question/30122195 """
    average_precision = average_precision_at_k(1, [1,0,1,0,0,1,0,0,1,1])
    print('average_precision', average_precision)

    query_labels = [1,1]
    retrieved_labels = [[1,0,1,0,0,1,0,0,1,1], [0,1,0,0,1,0,1,0,0,0]]
    mean_average_precision = mean_average_precision_at_k(query_labels, retrieved_labels)
    print('mean_average_precision', mean_average_precision)


##########################


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def search_k_nearest(target_embedding, embeddings, k):
    similarities = [cosine_similarity(target_embedding, emb) for emb in embeddings]
    sorted_indices = np.argsort(similarities)[::-1]
    return sorted_indices[1:k + 1]  # 排除第一个最相似的节点（自身）


def mean_precision_at_k(query_ids, labels, embeddings, k_values):
    results = {}
    for k in k_values:

        mean_precision = 0
        for query_id in tqdm(query_ids):
            query_label = labels[query_id]  # an integer
            query_embedding = embeddings[query_id]
            nearest_indices = search_k_nearest(query_embedding, embeddings, k)  # a list of integers
            retrieved_labels_relevant_to_q = labels[nearest_indices]
            # print('nearest_indices', nearest_indices, 
            #       'retrieved_labels_relevant_to_q', retrieved_labels_relevant_to_q)
            precision_at_k = 0
            for label in retrieved_labels_relevant_to_q:
                if label == query_label:
                    precision_at_k += 1
            precision_at_k /= k  # precision@k of current query
            
            mean_precision += precision_at_k
        mean_precision /= len(query_ids)
        results[k] = mean_precision
    return results


def mean_average_precision_at_k(query_ids, labels, embeddings, k_values):
    results = {}
    for k in k_values:

        mean_average_precision = 0
        for query_id in tqdm(query_ids):
            query_label = labels[query_id]  # an integer
            query_embedding = embeddings[query_id]
            nearest_indices = search_k_nearest(query_embedding, embeddings, k)  # a list of integers
            retrieved_labels_relevant_to_q = labels[nearest_indices]
            # print('nearest_indices', nearest_indices, 
            #       'retrieved_labels_relevant_to_q', retrieved_labels_relevant_to_q)
            average_precision = average_precision_at_k(query_label, retrieved_labels_relevant_to_q)
            mean_average_precision += average_precision

        mean_average_precision /= len(query_ids)
        results[k] = mean_average_precision
    return results


def test1(embeddings, labels, k_values = [10]):
    num_nodes = 100
    embeddings = np.random.rand(num_nodes, 128)  #
    labels = np.random.randint(0, 7, num_nodes)  #

    k_values = [98, 99]
    query_ids = [i for i in range(0, 80)]
    mean_average_precision_values = mean_average_precision_at_k(query_ids, labels, embeddings, k_values)
    print("MAP@k:", mean_average_precision_values)
    mean_precision_values = mean_precision_at_k(query_ids, labels, embeddings, k_values)
    print("precision@k:", mean_precision_values)


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

from numba import jit

@jit(nopython=True)
def hamming_dist(q, batch_visual_emb, batch_caption_emb):
    return 0.5 * (q - batch_visual_emb @ np.transpose(batch_caption_emb))

@jit(nopython=True)
def cos_sim(batch_visual_emb, batch_caption_emb):
    return batch_visual_emb @ np.transpose(batch_caption_emb)

def compute_distance(image_features, text_features, mode='hash', bs=1000):
    """ query, reference"""
    rows = image_features.shape[0]
    cols = text_features.shape[0]
    q = text_features.shape[1]  # max inner product value
    similarity_scores = np.zeros((rows, cols), dtype=np.float32)
    if mode == 'hash':
        for v in range(0, rows, bs):
            for t in range(0, cols, bs):
                batch_visual_emb = image_features[v:v+bs]
                batch_caption_emb = text_features[t:t+bs]

                # logits = 0.5 * (q - batch_visual_emb @ np.transpose(batch_caption_emb))
                logits = hamming_dist(q, batch_visual_emb, batch_caption_emb)
                similarity_scores[v:v+bs,t:t+bs] = logits
    else:
        image_features /= np.linalg.norm(image_features,axis=1)[:,np.newaxis]
        text_features /= np.linalg.norm(text_features,axis=1)[:,np.newaxis]

        for v in range(0, rows, bs):
            for t in range(0, cols, bs):
                batch_visual_emb = image_features[v:v+bs]
                batch_caption_emb = text_features[t:t+bs]

                # logits = 0.5 * (q - batch_visual_emb @ np.transpose(batch_caption_emb))
                logits = cos_sim(batch_visual_emb, batch_caption_emb)
                similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done vector distance')
    return similarity_scores


def compute_retrieval(a2b_sims, a_labels, b_labels, mode='hash', topk=20):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    topk_acc = {i:0 for i in range(1,topk+1)}
    query_num = a2b_sims.shape[0]
    
    for index in tqdm(range(query_num)):

        a_label = a_labels[index]
        # all_relevant_label_num =  np.sum(b_labels == a_label)

        # get order of similarities to target embeddings
        if mode == 'hash':  # a2b_sim is distance matrix
            b_indexes = np.argsort(a2b_sims[index])
        else:  # a2b_sim is cosine similarity matrix
            b_indexes = np.argsort(a2b_sims[index])[::-1]
        retrieved_labels = [b_labels[ind] for ind in b_indexes]

        retrieved_labels = retrieved_labels[:topk]
        for idx,pred_label in enumerate(retrieved_labels):
            if pred_label == a_label:
                for k in range(idx+1, topk+1):
                    topk_acc[k] += 1
                break

    for k in range(1, topk+1):
        topk_acc[k] = topk_acc[k] / len(a_labels)
    print('top-k accuracy:',topk_acc)
    #     for k in topk:
    #         retrieved_topk_labels = retrieved_labels[:k]
            
    #         # MAP@k
    #         average_precision = average_precision_at_k(a_label, retrieved_topk_labels)
    #         MAP_at_k[k] += average_precision
            
    #         # precision@k
    #         precision_at_k = 0
    #         for label in retrieved_topk_labels:
    #             if label == a_label:
    #                 precision_at_k += 1
    #         precision_at_k /= k
    #         mean_precision_at_k[k] += precision_at_k

    # for k in topk:
    #     MAP_at_k[k] /= query_num
    #     mean_precision_at_k[k] /= query_num
    
    # print('MAP_at_k:', MAP_at_k)
    # print('mean_precision_at_k:', mean_precision_at_k)
    # return MAP_at_k, mean_precision_at_k


def test2():
    num_nodes = 10000
    embeddings1 = np.random.rand(1000, 128)  #
    embeddings2 = np.random.rand(10000, 128)

    a_labels = np.random.randint(0, 7, num_nodes)
    b_labels = a_labels

    sim_distances = compute_distance(embeddings1, embeddings2)
    print(sim_distances)
    compute_retrieval(sim_distances, a_labels, b_labels)

