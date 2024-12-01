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
        k_cache.append(key.permute(0, 2, 1, 3).reshape(1, 5, -1))
    return k_cache
def get_v_cache(kv_cache):
    v_cache=[]
    for layer_idx, (key, value) in enumerate(kv_cache):
        v_cache.append(value.permute(0, 2, 1, 3).reshape(1, 5, -1))
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
    res_v = [get_v_cache(merged_KV)[i]-get_v_cache(main_KV)[i] for i in range(len(merged_KV))]
    for i in range(len(res_v)):
        # print(res_v[i].dtype)
        mini_KV = torch.matmul(B_inverse_weight[i].to(torch.float16),res_v[i].permute(0, 2, 1))  #v:1,4,4096      B_inver: 64,4096   v=BAx      1,64,4
        mini_KVs.append(mini_KV)
    return mini_KVs

def recompute(kv_cache,W_B):
    re_caches=[]
    for i in range(len(kv_cache)):
        re_cache=torch.matmul(W_B[i].to(torch.float16),kv_cache[i]).view(1, 32, -1, 128)  #1,64,4    4096,64 = [1,4096,4]   -> [1, 32, 4, 128]
        re_caches.append(re_cache)
    return re_caches
def tuple_to_list(kv):
    for i in range(len(kv)):
        for j in range(len(kv[i])):
            list_main_kv
def list_to_tuple(kv):
    

def inference_with_kv(model,inputs,W_B,kv_cache,main_kv):
    re_v_cache = recompute(kv_cache,W_B)
    for i in range(len(main_kv)):
        main_kv[i][1]=re_v_cache[i]
    outputs = model.generate(inputs.input_ids,past_key_values=main_KV)
    generated_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
    generated_ids = torch.cat((inputs.input_ids, generated_ids), 0)
    output_text = tokenizer.decode(generated_ids)
    return output_text
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    num_layers = model.config.num_hidden_layers
    system_prompt = "Once upon a time"
    inputs = tokenizer(text=system_prompt, return_tensors="pt").to(device)
    main_KV = prefill(model,inputs)
    lora0_model = PeftModel.from_pretrained(model,lora_path[0]).to(device)
    del model
    torch.cuda.empty_cache()
    merged_KV = prefill(lora0_model,inputs)
    W_B,W_B_inver=B_inverse_weight(lora0_model,num_layers)
    mini_KV = compute_miniKV(main_KV=main_KV,merged_KV=merged_KV,B_inverse_weight=W_B_inver)
    output_text = inference_with_kv(model=lora0_model,inputs=inputs,W_B=W_B,kv_cache=mini_KV,main_kv=main_KV)
    print(output_text)
