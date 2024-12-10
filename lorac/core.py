import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
from peft import PeftModel
torch.cuda.empty_cache()
model_name = "meta-llama/Llama-2-7b-hf"
lora_name = ["iamshnoo/alpaca-2-7b-english","iamshnoo/alpaca-2-7b-chinese","jb-01/llama-2-7b-ai2-arc"]
model_path = "/home/featurize/work/Lorac_recom/models/" 
lora_path= ["/home/featurize/work/Lorac_recom/lora_models/lora1",
            "/home/featurize/work/Lorac_recom/lora_models/lora2",
            "/home/featurize/work/Lorac_recom/lora_models/lora3"]
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
def get_v_cache(kv_cache):
    v_cache=[]
    for layer_idx, (key, value) in enumerate(kv_cache):
        v_cache.append(value.permute(0, 2, 1, 3).reshape(1, -1, 4096).contiguous())
    return v_cache
def compute_miniKV(main_KV, merged_KV, B_inverse_weight):
    mini_KVs=[]
    main_V = get_v_cache(main_KV)
    merged_V = get_v_cache(merged_KV)
    # print(len(main_V),main_V[0].shape)
    # print(len(merged_V),merged_V[0].shape)
    res_v = [merged_V[i]-main_V[i] for i in range(len(merged_V))]
    for i in range(len(res_v)):
        # print(res_v[i].dtype)
        mini_KV = torch.matmul(B_inverse_weight[i].to(torch.float32),res_v[i].permute(0, 2, 1))  #v:1,4,4096      B_inver: 64,4096   v=BAx      1,64,4
        mini_KVs.append(mini_KV)
    return mini_KVs
def recompute(kv_cache,W_B):
    re_caches=[]
    for i in range(len(kv_cache)):
        re_cache=torch.matmul(W_B[i].to(torch.float32),kv_cache[i]).view(1, 32, -1, 128)  #1,64,4    4096,64 = [1,4096,4]   -> [1, 32, 4, 128]
        re_caches.append(re_cache)
    return re_caches
def tuple_to_list(kv_tuple):
    main_kv_list = [list(item) for item in kv_tuple]
    return main_kv_list

def list_to_tuple(kv_list):
    main_kv_tuple =tuple(tuple(item) for item in kv_list)
    return main_kv_tuple
def inference_with_kv(model,inputs,W_B,tokenizer,kv_cache,main_kv):
    re_v_cache = recompute(kv_cache,W_B)
    main_kv = tuple_to_list(main_kv)
    for i in range(len(main_kv)):
        main_kv[i][1]+=re_v_cache[i]
    main_kv = list_to_tuple(main_kv)
    main_kv = DynamicCache.from_legacy_cache(main_kv)
    return main_kv
    # outputs = model.generate(inputs.input_ids,past_key_values=main_kv,use_cache= True,max_new_tokens=10)
    # output_text = tokenizer.decode(outputs[0],skip_special_tokens=True)
    # return output_text
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
SYSTEM_PROMPT = "You are a helpful assistant. "
inputs_initial_prompt = tokenizer(SYSTEM_PROMPT, return_tensors="pt").to("cuda")
base_cache =  DynamicCache()
with torch.no_grad():
     SYS_BASE_CACHE = model(**inputs_initial_prompt, past_key_values = base_cache).past_key_values
num_layers = model.config.num_hidden_layers
lora0_model = PeftModel.from_pretrained(model,lora_path[0], torch_dtype=torch.float32).to("cuda")
W_B,W_inverB = B_inverse_weight(lora0_model,num_layers)
lora0_cache =  DynamicCache()
with torch.no_grad():
     SYS_L0_CACHE = lora0_model(**inputs_initial_prompt, past_key_values = lora0_cache).past_key_values
MINI_KV0=compute_miniKV(SYS_BASE_CACHE,SYS_L0_CACHE,W_inverB)
RE0_CACHE = recompute(MINI_KV0,W_B)
main_kv= inference_with_kv(lora0_cache,inputs_initial_prompt,W_B,tokenizer,MINI_KV0,SYS_BASE_CACHE)
lora0_prompt ='Please introduce the University of Hong Kong to me. '
lora0_input = tokenizer(SYSTEM_PROMPT+lora0_prompt, return_tensors="pt").to("cuda")
# output = lora0_model.generate(**lora0_input, past_key_values=SYS_L0_CACHE,max_new_tokens=20)
# output_text = tokenizer.decode(output[0],skip_special_tokens=True)
output_re = lora0_model.generate(**lora0_input, past_key_values=main_kv,max_new_tokens=20)
output_re_text = tokenizer.decode(output_re[0],skip_special_tokens=True)
# print(output_text)
print('..........................')
print(output_re_text)