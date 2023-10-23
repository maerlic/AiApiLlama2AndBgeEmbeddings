from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer,\
BitsAndBytesConfig, AutoTokenizer
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model

app = Flask(__name__)

# Check if CUDA (GPU) is available
use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name = "abhinand/llama-2-13b-hf-bf16-sharded"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)

lora_config = LoraConfig.from_pretrained('final_finetuned_model_13b')
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model.to(device)

embeddings_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

class GenerationConfig:

    generation_config = model.generation_config
    generation_config.max_new_tokens = 70
    generation_config.temperature = 0.0
    generation_config.top_p = 1.0
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

@app.route('/data_extraction', methods=['POST'])
def generate_text():

    data = request.get_json()

    prompt_for_inference = data['Messages'][0]['Content']

    encoding = tokenizer(prompt_for_inference, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids = encoding.input_ids,
            attention_mask = encoding.attention_mask,
            generation_config = GenerationConfig.generation_config
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    return jsonify({'generated_text': generated_text})

@app.route('/calculate_embeddings_similarity', methods=['POST'])
def calculate_embeddings_similarity():
    # Step 2: Extract Sentences and Initialize SentenceTransformer Model
    sentences = request.json.get('Sentences')
    
    # Validate input sentences
    if not isinstance(sentences, list) or len(sentences) < 2:
        return jsonify({'error': 'Invalid input sentences.'}), 400
    
    # Step 3: Compute Embeddings and Similarity
    sentence_1 = sentences[0]
    sentence_2 = sentences[1]
    embeddings_1 = embeddings_model.encode(sentence_1, normalize_embeddings=True)
    embeddings_2 = embeddings_model.encode(sentence_2, normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    
    # Step 4: Return Embeddings and Similarity in JSON Response
    response = {
        'similarity_matrix': similarity.tolist()
    }
    
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)