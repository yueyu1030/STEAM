import torch
from transformers import *
from typing import List, Optional


def emb_extract(sentence: str, key_words: List[str],
                device: Optional[torch.device] = torch.device('cpu')) -> List[torch.tensor]:
    """
    :param sentence: input sentence;
    :param key_words: input keywords;
    :param device: cpu or cuda;
    :return: a list of embedding of keywords.
    """

    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens = tokenizer.tokenize(sentence)
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)], device=device)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

    token_list_new = list()
    idx_new2old_map = list()
    n = 0
    for i, token in enumerate(tokens):
        if '##' not in token:
            token_list_new.append(token)
            idx_new2old_map.append([i + 1])
            n += 1
        else:
            token_list_new[n - 1] += token.replace('##', '')
            idx_new2old_map[n - 1].append(i + 1)

    emb_list = list()
    for tgt in key_words:
        idx = token_list_new.index(tgt)
        old_idx = idx_new2old_map[idx]
        embs = last_hidden_states[0, old_idx, :].sum(dim=0) / len(old_idx)  # average emb of all tokens
        emb_list.append(embs.to('cpu'))

    return emb_list


def phrase_emb_extract(sentence: str, phrase_list: List[str],
                       device: Optional[torch.device] = torch.device('cpu')) -> List[torch.tensor]:
    """
    :param sentence: input sentence;
    :param phrase_list: a list of phrases whose embedding you want to compute;
    :param device: cpu or cuda;
    :return: a list of embedding of keywords.
    """
    emb_list = list()
    for phrase in phrase_list:
        words = phrase.strip().split(' ')
        embs = emb_extract(sentence, words, device)
        phrase_emb = torch.stack(embs).sum(dim=0) / len(embs)
        emb_list.append(phrase_emb)
    return emb_list


def main():
    strange_string = "'Here is some really sstrange' text... to encode???"
    key_words = ['sstrange', 'encode']
    phrase = ['some really']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('test keyword embedding')
    emb_list = emb_extract(strange_string, key_words, device)

    print(emb_list[0].size())
    print(emb_list[0])

    print('test phrase embedding')
    emb_list = phrase_emb_extract(strange_string, phrase, device)

    print(emb_list[0].size())
    print(emb_list[0])


if __name__ == '__main__':
    main()
