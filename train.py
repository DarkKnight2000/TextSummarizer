from datasets import load_dataset
from transformers import BertTokenizer, BertModel

import torch
from torch import nn
import re
from tqdm import tqdm, trange

dataset = load_dataset("ZhongshengWang/Alpaca-cnn-dailymail")
train_data = dataset["train"]
print("example: ", train_data[0].keys())

# print(dataset.keys())

bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

print('vocab size ', bert_tokenizer.vocab_size)

# example = "This is an example sentence. It has two sentences."
# tokens = tokenizer.tokenize(example)
# print(tokens)
# print(tokenizer.convert_tokens_to_ids(tokens))
# print("encoding: ", tokenizer.encode(example, add_special_tokens=True))
# print(tokenizer(
#     [i for i in re.split('[.?!]', example) if i != ''], # Sentence to encode.
#     add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#     padding = True,
#     return_attention_mask = True,   # Construct attn. masks.
#     return_tensors = 'pt',     # Return pytorch tensors.
# ))
# print(re.split('[.?!]', example))


def get_word_embeddings(tokenizer, sentence):
    encoding = tokenizer(
        [i for i in re.split('[.?!]', sentence) if i != ''], # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        padding = True,
        return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
    )

    with torch.no_grad():
        outputs = bert_model(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        word_embeddings = outputs.last_hidden_state  # This contains the embeddings
        # print(word_embeddings.shape)
        return word_embeddings.reshape(-1, 1, 768), encoding["input_ids"].reshape(-1)
    

EMBED_SIZE = 768
HIDDEN_SIZE = 64
encoder = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE, bidirectional=True)
decoder = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE)

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.weight_len = 8
        self.w_h = nn.Parameter(torch.randn(HIDDEN_SIZE, self.weight_len))
        self.w_s = nn.Parameter(torch.randn(HIDDEN_SIZE, self.weight_len))
        self.b_attn = nn.Parameter(torch.randn(1, self.weight_len))
        self.v = nn.Parameter(torch.randn(self.weight_len, 1))
        
    def get_attn(self, encoder_output: torch.Tensor, decoder_hidden: torch.Tensor):
        '''
            encoder_hidden -> (L, 1, HIDDEN_SIZE)
            decoder_hidden -> (1, HIDDEN_SIZE)
        '''
        encoder_hidden_len = encoder_output.shape[0]
        # print('enc shape', encoder_output.shape)
        # print('wh shape', self.w_h.shape)
        # print('matmul', torch.matmul(encoder_output.reshape(-1, EMBED_SIZE), self.w_h).shape)
        # print('matmul', torch.matmul(decoder_hidden.repeat(encoder_hidden_len, 1), self.w_s).shape)
        e_t = (torch.matmul(encoder_output.reshape(-1, HIDDEN_SIZE), self.w_h) +
                torch.matmul(decoder_hidden.repeat(encoder_hidden_len, 1), self.w_s) + self.b_attn.repeat(encoder_hidden_len, 1))
        # print('et shape', e_t.shape)
        e_t = torch.matmul(e_t, self.v)
        # print('et shape', e_t.shape)
        
        a_t = nn.functional.softmax(e_t, dim=0)
        # print(a_t.shape)
        return a_t
        

class VocabModel(nn.Module):
    def __init__(self, vocab_size):
        super(VocabModel, self).__init__()
        self.layer1 = 8
        self.layer2 = vocab_size
        self.v = nn.Parameter(torch.randn(2 * HIDDEN_SIZE, self.layer1))
        self.v_1 = nn.Parameter(torch.randn(self.layer1, self.layer2))
        self.b = nn.Parameter(torch.randn(1, self.layer1))
        self.b_1 = nn.Parameter(torch.randn(1, self.layer2))
        
    def get_prob(self, state: torch.Tensor):
        '''
            state -> (1, 2 * HIDDEN_SIZE)
        '''
        # encoder_hidden_len = encoder_output.shape[0]
        # print('enc shape', encoder_output.shape)
        # print('wh shape', self.w_h.shape)
        # print('matmul', torch.matmul(encoder_output.reshape(-1, EMBED_SIZE), self.w_h).shape)
        # print('matmul', torch.matmul(decoder_hidden.repeat(encoder_hidden_len, 1), self.w_s).shape)
        e_t = torch.matmul(state, self.v) + self.b
        # print('et shape', e_t.shape)
        e_t = torch.matmul(e_t, self.v_1) + self.b_1
        # print('et shape', e_t.shape)
        
        a_t = nn.functional.softmax(e_t, dim=1)
        # print(a_t.shape)
        return a_t
    

class PointerModel(nn.Module):
    def __init__(self):
        super(PointerModel, self).__init__()
        self.w_h = nn.Parameter(torch.randn(HIDDEN_SIZE, 1))
        self.w_s = nn.Parameter(torch.randn(HIDDEN_SIZE, 1))
        self.w_x = nn.Parameter(torch.randn(EMBED_SIZE, 1))
        self.b_attn = nn.Parameter(torch.randn(1, 1))
        
    def get_prob(self, context_vector: torch.Tensor, decoder_hidden: torch.Tensor, decoder_input: torch.Tensor):
        '''
            context_vector -> (1, HIDDEN_SIZE)
            decoder_hidden -> (1, HIDDEN_SIZE)
            decoder_input-> (1, EMBED_SIZE)
        '''
        # print('enc shape', encoder_output.shape)
        # print('wh shape', self.w_h.shape)
        # print('matmul', torch.matmul(encoder_output.reshape(-1, EMBED_SIZE), self.w_h).shape)
        # print('matmul', torch.matmul(decoder_hidden.repeat(encoder_hidden_len, 1), self.w_s).shape)
        e_t = (torch.matmul(context_vector, self.w_h) + torch.matmul(decoder_hidden, self.w_s) + torch.matmul(decoder_input, self.w_x) + self.b_attn)
        # print('et shape', e_t)
        
        a_t = nn.functional.sigmoid(e_t)
        # print(a_t.shape)
        return a_t
        


attn_model = AttentionModel()
vocab_model = VocabModel(vocab_size=bert_tokenizer.vocab_size)
pointer_model = PointerModel()
optimizer = torch.optim.SGD(list(attn_model.parameters()) + list(vocab_model.parameters()) + list(pointer_model.parameters()))

iter_loss_sum = 0
iter_loss_count = 0
iter_tqdm = tqdm(train_data, desc="Avg. loss 0", leave=True)
for paragh in iter_tqdm:
    # print(paragh)
    text_embeddings, text_input_ids = get_word_embeddings(bert_tokenizer, paragh["input"])
    summary_embeddings, summary_input_ids = get_word_embeddings(bert_tokenizer, paragh["output"])
    
    encoder_hidden = torch.randn(2, 1, HIDDEN_SIZE)
    encoder_state = torch.randn(2, 1, HIDDEN_SIZE)
    encoder_output, _ = encoder(text_embeddings, (encoder_hidden, encoder_state))
    # print('enc out ', encoder_output.shape, text_input_ids.shape)
    # print('enc out trunc ', encoder_output[:,:,:EMBED_SIZE].shape)
    encoder_output = encoder_output[:,:,:HIDDEN_SIZE]
    
    # while training, this is the previous
    # word of the reference summary; at test time it is
    # the previous word emitted by the decoder
    decoder_hidden = torch.randn(1, 1, HIDDEN_SIZE)
    decoder_state = torch.randn(1, 1, HIDDEN_SIZE)
    decoder_output, _ = decoder(summary_embeddings, (decoder_hidden, decoder_state))
    # print('dec out ', decoder_output.shape, summary_input_ids.shape)
    
    loss = 0
    optimizer.zero_grad()
    for timestep in range(decoder_output.shape[0]):
        a_t = attn_model.get_attn(encoder_output, decoder_output[timestep])
        context_vec = torch.sum(a_t.reshape(-1,1,1) * encoder_output, dim=0)
        # print('context vec ', context_vec.shape)
        appended_decoder = torch.cat((decoder_output[timestep], context_vec), dim=1)
        # print('appended dec ', appended_decoder.shape)
        
        p_vocab = vocab_model.get_prob(appended_decoder)
        # print('prob vocab sum ', torch.sum(p_vocab), p_vocab.shape)
        
        p_gen = pointer_model.get_prob(context_vec, decoder_output[timestep], summary_embeddings[timestep])
        # print((p_gen))
        
        # print('ber toekn vcb len ', len(bert_tokenizer.vocab))
        
        # is_word_oov = summary_input_ids[timestep] not in bert_tokenizer.vocab.values()
        # print(len(summary_input_ids))
        p_word = p_gen * p_vocab[0][summary_input_ids[timestep]]
        # print("attn sum ", sum(a_t), a_t.shape, text_input_ids.shape)
        p_word += (1 - p_gen) * sum([a_t[i][0] for i in range(len(text_input_ids)) if text_input_ids[i] == summary_input_ids[timestep]])
        # print('p_wrd', p_word, torch.log(p_word))
        
        loss += -1 * torch.log(p_word + pow(10, -10))
        
    loss /= (decoder_output.shape[0])
    # print("loss", loss)
    iter_loss_count += 1
    iter_loss_sum += loss.item()
    iter_tqdm.set_description(f"Avg. loss {iter_loss_sum/iter_loss_count}")
    loss.backward()
    optimizer.step()
