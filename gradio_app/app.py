import transformers
import gradio as gr
import torch
import gc
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, AutoConfig

AVAILABLE_MODELS = {
    '600M_base': 'el-izm/nllb_nivkh_rus_600M_base',
    '600M_extended': 'el-izm/nllb_nivkh_rus_600M_extended',
    '600M_amur_extended': 'el-izm/nllb_nivkh_rus_600M_amur_extended', 
    '600M_sakh_extended': 'el-izm/nllb_nivkh_rus_600M_sakh_extended',
    '1.3B_base': 'el-izm/nllb_nivkh_rus_1.3B_base'
}

current_model = None
current_tokenizer = None
current_model_name = None

def clear_memory():
    global current_model, current_tokenizer
    if current_model is not None:
        del current_model
        current_model = None
    if current_tokenizer is not None:
        del current_tokenizer
        current_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # ждём завершения всех gpu операций

def load_model(model_display_name):
    global current_model, current_tokenizer, current_model_name
    model_name = AVAILABLE_MODELS[model_display_name]
    if current_model_name == model_name:
        return 'Модель уже загружена'
    clear_memory()
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        current_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=cfg)
        current_tokenizer = NllbTokenizer.from_pretrained(model_name)
        fix_tokenizer(current_tokenizer)
        current_model_name = model_name
        return f"Модель '{model_display_name}' загружена"
    except Exception as e:
        return f'Ошибка при загрузке модели: {str(e)}'

def translate(
    text,
    src_lang,
    tgt_lang,
    model_choice,
    max_length='auto',
    num_beams=4,
    no_repeat_ngram_size=4,
    n_out=None,
    **kwargs
):
    model_name = AVAILABLE_MODELS[model_choice]
    if current_model_name != model_name:
        load_model(model_choice)
    
    if current_model is None or current_tokenizer is None:
        return 'Ошибка: модель не загружена'
    
    lang_map = {'Нивхский': 'nivkh_Cyrl', 'Русский': 'rus_Cyrl'}
    src_lang_code = lang_map[src_lang]
    tgt_lang_code = lang_map[tgt_lang]
    current_tokenizer.src_lang = src_lang_code
    encoded = current_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    if max_length == 'auto':
        max_length = int(32 + 2.0 * encoded.input_ids.shape[1])
        
    current_model.eval()
    generated_tokens = current_model.generate(
        **encoded.to(current_model.device),
        forced_bos_token_id=current_tokenizer.lang_code_to_id[tgt_lang_code],
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_return_sequences=n_out or 1,
        **kwargs
    )
    out = current_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    if isinstance(text, str) and n_out is None:
        return out[0]
    return out

LANGUAGE_TARGET_LABEL = 'nivkh_Cyrl'
def fix_tokenizer(tokenizer, new_lang=LANGUAGE_TARGET_LABEL):
    """ Add a new language token to the tokenizer vocabulary (this should be done each time after its initialization) """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}

default_model = '600M_base'
load_model(default_model)

lang_options = ['Нивхский', 'Русский']
model_options = list(AVAILABLE_MODELS.keys())

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label='Текст для перевода', lines=5, placeholder='Введите текст для перевода...'),
        gr.Dropdown(choices=lang_options, value='Нивхский', label='Исходный язык'),
        gr.Dropdown(choices=lang_options, value='Русский', label='Целевой язык'),
        gr.Dropdown(choices=model_options, value=default_model, label='Модель перевода'),
    ],
    outputs=gr.Textbox(label='Переведённый текст'),
    title='Нивхско ⇄ Русский переводчик',
    submit_btn='Перевести',
    clear_btn='Очистить'
)

demo.launch()