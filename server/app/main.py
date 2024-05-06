from fastapi import Depends, FastAPI
import uvicorn
import os
from huggingface_hub import HfApi
from fastapi.middleware.cors import CORSMiddleware


from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from fastapi import APIRouter, UploadFile, File, Request, Form
from io import BytesIO
import base64
import json


parser = argparse.ArgumentParser()
# "THUDM/cogagent-chat-hf"
# THUDM/cogagent-vqa-hf
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")

args = parser.parse_args()
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()

text_only_template = "USER: {} ASSISTANT:"

app = FastAPI()
#include COR permissions for API request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def welcome():
    return {"message": "Welcome to QA Agent Backend"}
# [
#                 {"role": "system", "content": [{"type": "text", "text": prompt0}]},
#                 {"role": "user",
#                  "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url":
#                                                                                                         f"data:image/jpeg;nase64,{base64_image}",
#                                                                                                     "detail": "high"},
#                                                                  }]},
#             ]
@app.post('/run_inference')
async def run_inference(imageFile: UploadFile, conv_data: str = Form(), prompt: str = Form() ):

    # request_params = await request.json()
    #image_path = input("image path >>>>> ")
    image = ''
    # image = Image.open(image_path).convert('RGB')
    
    history = []


    #query = input("Human:")
    # if query == "clear":
    #     break
    query = prompt
    # if image is None:
    #     if text_only_first_query:
    #         query = text_only_template.format(query)
    #         text_only_first_query = False
    #     else:
    #         old_prompt = ''
    #         for _, (old_query, response) in enumerate(history):
    #             old_prompt += old_query + " " + response + "\n"
    #         query = old_prompt + "USER: {} ASSISTANT:".format(query)


    # query = prompt
    # query = request_params['data'][-1]['content'][0]['text']

    conv_data = json.loads(conv_data)

    if len(conv_data) <= 1:
        history = []
    else:
        history = [(conv_data[idx], conv_data[idx + 1]) for idx in range(0, len(conv_data[0:len(conv_data) - 1]), 2)]
        # history = [(conv_data[idx]['content'][0]['text'], conv_data[idx + 1]['content'][0]['text']) for idx in range(0, len(conv_data[0:len(conv_data) - 1]), 2)]
    
    # image_history = [item['image_url']['url']  for conv in conv_data for item in conv['content'] if item['type'] == 'image_url']

    # if len(image_history) == 0:
    #     input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, template_version='base')
    # else:
    #input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[Image.open(BytesIO(base64.b64decode(image_history[-1]))).convert('RGB')])
    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[Image.open(BytesIO(await imageFile.read())).convert('RGB')])

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

    # add any transformers params here.
    gen_kwargs = {"max_length": 2048,
                    "do_sample": False} # "temperature": 0.9
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]
        # print("\nCog:", response)
        return {"response": response}
    #history.append((query, response))


if __name__ == '__main__':
    uvicorn.run(app, port=4000, host='0.0.0.0')

