# Extract posts and items' embeddings with Transformers
import torch
from transformers import AutoTokenizer, AutoModel

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    outputs = outputs
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings.detach().numpy()

items_embs = get_embedding(sentences_bdi, model, tokenizer)
