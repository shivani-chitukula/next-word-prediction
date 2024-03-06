import torch
import string
from transformers import BertTokenizer, BertForMaskedLM
import streamlit as st
from transformers import AutoProcessor, SeamlessM4TModel

def load_model(model_name):
  try:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
    return bert_tokenizer,bert_model
  except Exception as e:
    pass
# bert encode
def encode_bert(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx


# bert decode
def decode_bert(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])


def get_all_predictions(text_sentence,  model_name, top_clean=5):
  if model_name.lower() == "bert":
    input_ids, mask_idx = encode_bert(bert_tokenizer, text_sentence)
    with torch.no_grad():
      predict = bert_model(input_ids)[0]
    bert = decode_bert(bert_tokenizer, predict[0, mask_idx, :].topk(no_words_to_be_predicted).indices.tolist(), top_clean)
    return {'bert': bert}


def get_prediction_end_of_sentence(input_text, model_name):
  try:
    input_text += ' <mask>'
    #print(input_text)
    res = get_all_predictions(input_text, model_name, top_clean=int(no_words_to_be_predicted))
    return res
  except Exception as error:
    pass
no_words_to_be_predicted=5
bert_tokenizer, bert_model = load_model("bert")
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
#input="The restaurant was"

st.title("Next Word Prediction using BERT")
inp = st.text_input("User Input:", value=" ", help="Enter text here...")
button_conversation = st.button("predict and translate")
if button_conversation:
    res = get_prediction_end_of_sentence(inp, "bert")
    l = list(res['bert'].split("\n"))
    st.markdown(input)
    st.markdown(l)
    for i in l:
        text_inputs = processor(text=inp + i + ".", src_lang="eng", return_tensors="pt")
        output_tokens = model.generate(**text_inputs, tgt_lang="tel", generate_speech=False)
        translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        st.markdown(translated_text_from_text)
    for i in l:
        text_inputs = processor(text=inp + i + ".", src_lang="eng", return_tensors="pt")
        output_tokens = model.generate(**text_inputs, tgt_lang="hin", generate_speech=False)
        translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        st.markdown(translated_text_from_text)

