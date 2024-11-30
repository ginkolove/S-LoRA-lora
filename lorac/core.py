import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model_name = "meta-llama/Llama-2-7b-hf"
lora_name = ["iamshnoo/alpaca-2-7b-english","iamshnoo/alpaca-2-7b-chinese","jb-01/llama-2-7b-ai2-arc"]
model_path = "/home/featurize/work/Lorac_recom/models/" 
lora_path= ["/home/featurize/work/Lorac_recom/lora_models/lora1",
            "/home/featurize/work/Lorac_recom/lora_models/lora2",
            "/home/featurize/work/Lorac_recom/lora_models/lora3"]


def load_model_and_tokenizer(base_model_name, lora_model_name=None):
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer

def load_adapter(model,lora):
    model = PeftModel.from_pretrained(model, lora)
    return model
def prefill(model,inputs):
    # if lora == None:
    outputs = model(inputs.input_ids,use_cache=True, output_hidden_states=False)
    main_KV = outputs.past_key_values
        # generated_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        # output_text = tokenizer.decode(generated_ids)
    return main_KV
    # else:
    #     lora_model = load_adapter(model,lora)
    #     outputs = lora_model(inputs.input_ids,use_cache=True, output_hidden_states=False)
    #     merged_KV = outputs.past_key_values
    #     # generated_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
    #     # output_text = tokenizer.decode(generated_ids)
    #     return merged_KV
def get_k_cache(kv_cache):
    k_cache=[]
    for layer_idx, (key, value) in enumerate(kv_cache):
        k_cache.append(key)
    return k_cache
def get_v_cache(kv_cache):
    v_cache=[]
    for layer_idx, (key, value) in enumerate(kv_cache):
        v_cache.append(value)
    return v_cache
def B_inverse_weight(model,num_layers):
    W_b=[]
    W_b_inver =[]
    state_dict = model.state_dict()
    for i in range(num_layers):
        layer_key = f'base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.default.weight'
        layer_B = state_dict[layer_key]
        W_b.append(layer_B)
        layer_B_inverse = torch.linalg.pinv(layer_B)
        W_b_inver.append(layer_B_inverse)
    return W_b,W_b_inver

def token(tokenizer,prompt,device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    return inputs

def compute_miniKV(main_KV, merged_KV, B_inverse_weight):
    mini_KVs=[]
    res_kv = merged_KV-main_KV
    res_v = get_v_cache(res_kv)
    for i in range(len(res_kv)):
        mini_KV = torch.matmul(res_kv[i],B_inverse_weight[i])
        mini_KVs.append(mini_KV)
    return mini_KVs

def recompute(kv_cache,W_B):
    re_caches=[]
    for i in range(len(kv_cache)):
        re_cache=torch.matmul(kv_cache[i],W_B[i])
        re_caches.append(re_cache)
    return re_caches
def inference_with_kv(model,inputs,W_B,kv_cache,main_kv):
    re_v_cache = recompute(kv_cache,W_B)
    for i in range(len(main_kv)):
        main_KV[i][1]=re_v_cache[i]
    outputs = model.generate(inputs.input_ids,past_key_values=main_KV)
    generated_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
    generated_ids = torch.cat((inputs.input_ids, generated_ids), 0)
    output_text = tokenizer.decode(generated_ids)
    return output_text
if __name__ == "__main__":
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = model.config
    num_layers = config.num_hidden_layers
    system_prompt = "Once upon a time"
    inputs = tokenizer(text=system_prompt, return_tensors="pt")
    main_KV = prefill(model,inputs)
    lora0_model = PeftModel.from_pretrained(model,lora_path[0])
    merged_KV = prefill(lora0_model,inputs)
    W_B,W_B_inver=B_inverse_weight(lora0_model,num_layers)
    mini_KV = compute_miniKV(main_KV=main_KV,merged_KV=merged_KV,B_inverse_weight=W_B_inver)
    output_text = inference_with_kv(model=lora0_model,inputs=inputs,W_B=W_B,kv_cache=mini_KV,main_kv=main_KV)
    print(output_text)
