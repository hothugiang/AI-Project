import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import numpy as np
from rouge import Rouge
import string
import re
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from underthesea import sent_tokenize, word_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abstract_model_path = "bartpho-cp11000"
abstract_tokenizer_path = "bartpho-tokenizer"
stopword_path = "vietnamese-stopwords-dash.txt"
LDA_model_path = "LDA_models.pkl"
extractive_model_path = "e_25_0.3071.mdl"
contrastive_model_path = "c_25_0.3071.mdl"
with open('dict_map.json', 'r', encoding='utf-8') as f:
    dict_map = json.load(f)

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model_summarization = AutoModelForSeq2SeqLM.from_pretrained(abstract_model_path).to(device)
tokenizer_summarization = AutoTokenizer.from_pretrained(abstract_tokenizer_path)

"""# Extractive model"""


def getRouge2(ref, pred, kind):  # tokenized input
    try:
        return round(Rouge().get_scores(pred.lower(), ref.lower())[0]['rouge-2'][kind], 4)
    except ValueError:
        return 0.0


class MLP(nn.Module):
    def __init__(self, dims: list, layers=2, act=nn.LeakyReLU(), dropout_p=0.3, keep_last_layer=False):
        super(MLP, self).__init__()
        assert len(dims) == layers + 1
        self.layers = layers
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.keep_last = keep_last_layer

        self.mlp_layers = nn.ModuleList([])
        for i in range(self.layers):
            self.mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for i in range(len(self.mlp_layers) - 1):
            x = self.dropout(self.act(self.mlp_layers[i](x)))
        if self.keep_last:
            x = self.mlp_layers[-1](x)
        else:
            x = self.act(self.mlp_layers[-1](x))
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, docnum, secnum):
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))

        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float(-1e9))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)


class GAT(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor, docnum, secnum):
        x = x.squeeze(0)
        adj_mat = adj_mat.squeeze(0)
        adj_x = adj_mat.clone().sum(dim=1, keepdim=True).repeat(1, x.shape[1]).bool()
        adj_mat = adj_mat.unsqueeze(-1).bool()
        x = self.dropout(x)
        x = self.layer1(x, adj_mat, docnum, secnum)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x, adj_mat, docnum, secnum).masked_fill(adj_x == 0, float(0))
        return x.unsqueeze(0)


class StepWiseGraphConvLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout_p=0.3, act=nn.LeakyReLU(), nheads=6, iter=1, final="att"):
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.iter = iter
        self.in_dim = in_dim
        self.gat = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                      dropout=dropout_p, n_heads=nheads) for _ in range(iter)])
        self.gat2 = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                       dropout=dropout_p, n_heads=nheads) for _ in range(iter)])
        self.gat3 = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                       dropout=dropout_p, n_heads=nheads) for _ in range(iter)])

        self.out_ffn = MLP([in_dim * 3, hid_dim, hid_dim, in_dim], layers=3, dropout_p=dropout_p)

    def forward(self, feature, adj, docnum, secnum):
        sen_adj = adj.clone()
        sen_adj[:, -docnum - secnum - 1:, :] = sen_adj[:, :, -docnum - secnum - 1:] = 0
        sec_adj = adj.clone()
        sec_adj[:, :-docnum - secnum - 1, :] = sec_adj[:, -docnum - 1:, :] = sec_adj[:, :, -docnum - 1:] = 0
        doc_adj = adj.clone()
        doc_adj[:, :-docnum - 1, :] = 0

        feature_sen = feature.clone()
        feature_resi = feature

        feature_sen_re = feature_sen.clone()
        for i in range(0, self.iter):
            feature_sen = self.gat[i](feature_sen, sen_adj, docnum, secnum)
        feature_sen = F.layer_norm(feature_sen + feature_sen_re, [self.in_dim])

        feature_sec = feature_sen.clone()
        feature_sec_re = feature_sec.clone()
        for i in range(0, self.iter):
            feature_sec = self.gat2[i](feature_sec, sec_adj, docnum, secnum)
        feature_sec = F.layer_norm(feature_sec + feature_sec_re, [self.in_dim])

        feature_doc = feature_sec.clone()
        feature_doc_re = feature_doc.clone()
        for i in range(0, self.iter):
            feature_doc = self.gat3[i](feature_doc, doc_adj, docnum, secnum)
        feature_doc = F.layer_norm(feature_doc + feature_doc_re, [self.in_dim])

        feature_sec[:, :-docnum - secnum - 1, :] = adj[:, :-docnum - secnum - 1,
                                                   -docnum - secnum - 1:-docnum - 1] @ feature_sec[:,
                                                                                       -docnum - secnum - 1:-docnum - 1,
                                                                                       :]
        feature_doc[:, -docnum - secnum - 1:-docnum - 1, :] = adj[:, -docnum - secnum - 1:-docnum - 1,
                                                              -docnum - 1:] @ feature_doc[:, -docnum - 1:, :]
        feature_doc[:, :-docnum - secnum - 1, :] = adj[:, :-docnum - secnum - 1,
                                                   -docnum - secnum - 1:-docnum - 1] @ feature_doc[:,
                                                                                       -docnum - secnum - 1:-docnum - 1,
                                                                                       :]
        feature = torch.concat([feature_doc, feature_sec, feature_sen], dim=-1)
        feature = F.layer_norm(self.out_ffn(feature) + feature_resi, [self.in_dim])
        return feature


class Contrast_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, act=nn.LeakyReLU(0.1), dropout_p=0.3):
        super(Contrast_Encoder, self).__init__()
        self.graph_encoder = StepWiseGraphConvLayer(in_dim=input_dim, hid_dim=hidden_dim,
                                                    dropout_p=dropout_p, act=act, nheads=heads, iter=1)
        self.common_proj_mlp = MLP([input_dim, hidden_dim, input_dim], layers=2, dropout_p=dropout_p, act=act,
                                   keep_last_layer=False)

    def forward(self, p_gfeature, doc_lens, p_adj, docnum, secnum):
        posVec = torch.cat(
            [PositionVec[:l] for l in doc_lens] + [torch.zeros(secnum + docnum + 1, 768).float().to(device)], dim=0)
        p_gfeature = p_gfeature + posVec.unsqueeze(0)
        pg = self.graph_encoder(p_gfeature, p_adj, docnum, secnum)
        pg = self.common_proj_mlp(pg)
        return pg


class End2End_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, act=nn.LeakyReLU(0.1), dropout_p=0.3):
        super(End2End_Encoder, self).__init__()
        self.graph_encoder = StepWiseGraphConvLayer(in_dim=input_dim, hid_dim=hidden_dim,
                                                    dropout_p=dropout_p, act=act, nheads=heads, iter=1)
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj_layer_mlp = MLP([input_dim, hidden_dim, input_dim], layers=2, dropout_p=dropout_p, act=act,
                                      keep_last_layer=False)
        self.linear = MLP([input_dim, 1], layers=1, dropout_p=dropout_p, act=act, keep_last_layer=True)

    def forward(self, x, doc_lens, adj, docnum, secnum):
        x = self.graph_encoder(x, adj, docnum, secnum)
        x = x[:, :-docnum - secnum - 1, :]
        x = self.out_proj_layer_mlp(x)
        x = self.linear(x)
        return x


class Cluster:
    def __init__(self, sent_texts, sent_vecs, doc_lens, doc_sec_mask, sec_sen_mask):
        assert len(sent_vecs) == len(sent_texts)
        self.docnum = len(doc_sec_mask)
        self.secnum = len(sec_sen_mask)
        self.feature = torch.cat(
            (torch.stack(sent_vecs, dim=0), torch.zeros((self.secnum + self.docnum + 1, sent_vecs[0].shape[0]))),
            dim=0).to(device)
        self.adj = torch.from_numpy(self.mask_to_adj(doc_sec_mask, sec_sen_mask)).float().to(device)
        self.sent_text = np.array(sent_texts)
        self.doc_lens = doc_lens
        self.init_node_vec()
        self.feature = self.feature.float()

    def init_node_vec(self):
        docnum, secnum = self.docnum, self.secnum
        for i in range(-secnum - docnum - 1, -docnum - 1):
            mask = self.adj[i].clone()
            mask[-secnum - docnum - 1:] = 0
            self.feature[i] = torch.mean(self.feature[mask.bool()], dim=0)
        for i in range(-docnum - 1, -1):
            mask = self.adj[i].clone()
            mask[-docnum - 1:] = 0
            self.feature[i] = torch.mean(self.feature[mask.bool()], dim=0)
        self.feature[-1] = torch.mean(self.feature[-docnum - 1:-1], dim=0)

    def mask_to_adj(self, doc_sec_mask, sec_sen_mask):
        sen_num = sec_sen_mask.shape[1]
        sec_num = sec_sen_mask.shape[0]
        doc_num = doc_sec_mask.shape[0]
        adj = np.zeros((sen_num + sec_num + doc_num + 1, sen_num + sec_num + doc_num + 1))
        # section connection
        adj[-sec_num - doc_num - 1:-doc_num - 1, 0:-sec_num - doc_num - 1] = sec_sen_mask
        adj[0:-sec_num - doc_num - 1, -sec_num - doc_num - 1:-doc_num - 1] = sec_sen_mask.T
        for i in range(0, doc_num):
            doc_mask = doc_sec_mask[i]
            doc_mask = doc_mask.reshape((1, len(doc_mask)))
            adj[sen_num:-doc_num - 1, sen_num:-doc_num - 1] += doc_mask * doc_mask.T
        # doc connection
        adj[-doc_num - 1:-1, -sec_num - doc_num - 1:-doc_num - 1] = doc_sec_mask
        adj[-sec_num - doc_num - 1:-doc_num - 1, -doc_num - 1:-1] = doc_sec_mask.T
        adj[-doc_num - 1:, -doc_num - 1:] = 1

        #build sentence connection
        for i in range(0, sec_num):
            sec_mask = sec_sen_mask[i]
            sec_mask = sec_mask.reshape((1, len(sec_mask)))
            adj[:sen_num, :sen_num] += sec_mask * sec_mask.T
        return adj


def meanTokenVecs(text):
    sent = text.lower()
    input_ids = torch.tensor([phobert_tokenizer.encode(sent)])
    tokenized_text = phobert_tokenizer.tokenize(sent)
    with torch.no_grad():
        features = phobert(input_ids)
    wordVecs, buffer, buffer_str = {}, [], ''
    for token in zip(tokenized_text, features.last_hidden_state[0, 1:-1, :]):
        if token[0][-2:] == '@@':
            buffer.append(token[1])
            buffer_str += token[0][:-2]
            continue
        if buffer:
            buffer.append(token[1])
            buffer_str += token[0]
            wordVecs[buffer_str] = torch.mean(torch.stack(buffer), dim=0)
            buffer, buffer_str = [], ''
        else:
            wordVecs[token[0]] = token[1]

    return torch.mean(torch.stack([vec for w, vec in wordVecs.items() if w not in string.punctuation]), dim=0)


def getPositionEncoding(pos, d=768, n=10000):
    P = np.zeros(d)
    for i in np.arange(int(d / 2)):
        denominator = np.power(n, 2 * i / d)
        P[2 * i] = np.sin(pos / denominator)
        P[2 * i + 1] = np.cos(pos / denominator)
    return P


def removeRedundant(text):
    text = text.lower()
    words = [w for w in text.split(' ') if w not in stop_w]
    return ' '.join(words)


def divideSection(doc_text, category='Giáo dục'):
    sent_para, para_sec, sent_sec = {}, {}, {}

    paras = [para for para in doc_text.split('\n') if para != '']
    all_sents = []
    # prepare sent_Para
    sentcnt = 0
    for i, para in enumerate(paras):
        sents = [word_tokenize(sent, format="text") for sent in sent_tokenize(para) if sent != '' and len(sent) > 4]
        all_sents.extend(sents)
        for ii, sent in enumerate(sents):
            sent_para[sentcnt + ii] = i
            sent = removeRedundant(sent)
        sentcnt += len(sents)

    # prepare para_sec
    paras = [removeRedundant(para) for para in paras]
    tf, lda_model = cate_models[category]
    X = tf.transform(paras)
    lda_top = lda_model.transform(X)
    for i, para_top in enumerate(lda_top):
        para_sec[i] = para_top.argmax()

    # output sent_sec
    for k, v in sent_para.items():
        sent_sec[k] = para_sec[v]
    return sent_sec, all_sents


def loadClusterData(docs_org, category='Giáo dục'):  # docs_org: list of text for each document
    seclist, docs = {}, []
    for d, doc in enumerate(docs_org):
        seclist[d], sentTexts = divideSection(doc, category)
        docs.append(sentTexts)

    secnum = 0
    for k, val_dict in seclist.items():
        vals = set(val_dict.values())
        for ki, vi in val_dict.items():
            for i, v in enumerate(vals):
                if vi == v:
                    val_dict[ki] = i + secnum
                    break
        seclist[k] = val_dict
        secnum += len(vals)

    sents, sentVecs, secIDs, doc_lens = [], [], [], []
    sentnum = sum([len(doc.values()) for doc in seclist.values()])
    doc_sec_mask = np.zeros((len(docs), secnum))
    sec_sen_mask = np.zeros((secnum, sentnum))
    cursec, cursent = 0, 0

    for d, doc in enumerate(docs):
        doc_lens.append(len(doc))
        doc_endsec = max(seclist[d].values())
        doc_sec_mask[d][cursec:doc_endsec + 1] = 1
        cursec = doc_endsec + 1
        for s, sent in enumerate(doc):
            sents.append(sent)
            sentVecs.append(meanTokenVecs(sent))
            sec_sen_mask[seclist[d][s], cursent] = 1
            cursent += 1

    return Cluster(sents, sentVecs, doc_lens, doc_sec_mask, sec_sen_mask)


def val_e2e(data, model, c_model=None):
    feature = data.feature.unsqueeze(0)
    doc_lens = data.doc_lens
    adj = data.adj.unsqueeze(0)
    docnum = data.docnum
    secnum = data.secnum

    with torch.no_grad():
        feature = c_model(feature, doc_lens, adj, docnum, secnum)
        x = model(feature, doc_lens, adj, docnum, secnum)
        scores = torch.sigmoid(x.squeeze(-1))

    return scores[0], data.sent_text


PositionVec = torch.stack([torch.from_numpy(getPositionEncoding(i, d=768)) for i in range(200)], dim=0).float().to(
    device)
stop_w = ['...']
with open(stopword_path, 'r', encoding='utf-8') as f:
    for w in f.readlines():
        stop_w.append(w.strip())
stop_w.extend([c for c in '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~…“”’‘'])

with open(LDA_model_path, mode='rb') as fp:
    cate_models = pickle.load(fp)

c_model = Contrast_Encoder(768, 1024, 4).to(device)
model = End2End_Encoder(768, 1024, 4).to(device)
model.load_state_dict(torch.load(extractive_model_path, map_location=device), strict=False)
c_model.load_state_dict(torch.load(contrastive_model_path, map_location=device), strict=False)
model.eval()
c_model.eval()

"""# Abstractive model"""


class Abstractive_Summarization:
    def replace_all(text):
        for i, j in Summarization.dict_map.items():
            text = text.replace(i, j)
        return text

    @staticmethod
    def generateSummary(text, sentnum):
        model_summarization.eval()
        with torch.no_grad():
            text = [str(sentnum) + ' câu. Tên: <>. Nội dung: <' + text + '>']
            inputs = tokenizer_summarization(text, padding=True, max_length=1024, truncation=True,
                                             return_tensors='pt').to(device)
            outputs = model_summarization.generate(**inputs, max_length=512, num_beams=5,
                                                   early_stopping=True, no_repeat_ngram_size=3)
            prediction = tokenizer_summarization.batch_decode(outputs, skip_special_tokens=True)
        return prediction[0]


"""# Main function"""


def get_summary(scores, sents, max_sent=5):
    ranked_score_idxs = torch.argsort(scores, dim=0, descending=True)
    sents = [s.replace('_', ' ') for s in sents]
    summSentIDList = []
    for i in ranked_score_idxs:
        if len(summSentIDList) >= max_sent: break
        s = sents[i]

        replicated, delIDs = False, []
        for chosedID in summSentIDList:
            if getRouge2(s, sents[chosedID], 'p') >= 0.45:
                delIDs.append(chosedID)
            if getRouge2(sents[chosedID], s, 'p') >= 0.45:
                replicated = True
                break
        if replicated: continue

        for delID in delIDs:
            del summSentIDList[summSentIDList.index(delID)]
        summSentIDList.append(i)
    summSentIDList = sorted(summSentIDList)
    return ' '.join([s for i, s in enumerate(sents) if i in summSentIDList])


def MultiDocSummarizationAPI(texts, compress_ratio):
    """
    Summarizes a list of documents using both extractive and abstractive methods.

    Parameters:
    - texts (list of str): A list of document texts to be summarized.
    - compress_ratio (float): A ratio or count determining the number of sentences in the summary.
      If less than 1, it represents the fraction of the original sentences to include in the summary.
      If 1 or greater, it represents the exact number of sentences to include in the summary.

    Returns:
    - dict: A dictionary containing:
        - 'extractive_summ' (str): The extractive summary of the documents.
        - 'abstractive_summ' (str): The abstractive summary of the documents.
    """
    assert compress_ratio > 0, "Compress ratio need to be greater than 0."
    docs = [text.strip() for text in texts]
    data_tree = loadClusterData(docs)
    scores, sents = val_e2e(data_tree, model, c_model=c_model)

    output_sent_cnt = int(len(sents) * compress_ratio) if compress_ratio < 1 else int(compress_ratio)
    print('Expected sentence count:', output_sent_cnt)

    extractive_summ = get_summary(scores, sents, max_sent=output_sent_cnt)
    extractive_summ = re.sub(r'\s+([.,;:"?)/!?”])', r'\1', extractive_summ)
    extractive_summ = re.sub(r'([(“])\s+', r'\1', extractive_summ)

    abstract_inputs = get_summary(scores, sents, max_sent=25)
    abstract_summ = Abstractive_Summarization.generateSummary(abstract_inputs, output_sent_cnt)
    abstract_summ = re.sub(r'\s+([.,;:"?)/!?”])', r'\1', abstract_summ)
    abstract_summ = re.sub(r'([(“])\s+', r'\1', abstract_summ)
    return {'extractive_summ': extractive_summ,
            'abstractive_summ': abstract_summ}

