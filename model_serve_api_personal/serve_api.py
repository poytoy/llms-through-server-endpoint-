from fastapi import FastAPI,HTTPException,Body
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes
import torch
import yaml
import gc
#created by poyraz guler
app = FastAPI()

class PromptModel(BaseModel):
    prompt: str

class QuantizeConfig(BaseModel):
    enable: bool
    type: str 

class ModelConfig(BaseModel):
    repo_id: str
    torch_dtype: str 
    max_new_tokens: int
    temperature: float
    quantize: QuantizeConfig # Options: "4bit", "8bit"

class ModelHandler:
    def __init__(self,config: ModelConfig):
        self.model_cache = {}
        self.tokenizer_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_path = None
        self.current_model = None
        self.current_tokenizer = None
        self.config = config
        
        self.unload_model()
        self.load_model()

    def unload_model(self):
        """Unload all models from the GPU."""
        if self.current_model is not None:
            print("unloading")
            #if "4bit" not in self.current_model_path and "8bit" not in self.current_model_path:
            #    self.current_model.to('cpu')
            del self.current_model
            del self.current_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_path = None

    def load_model(self):
        print(f"Loading model from {self.config.repo_id}")
        # Check if the requested model is the current one
        if self.current_model_path == self.config.repo_id:
            # Return the already loaded model and tokenizer
            return 

        # Unload the current model if it's different from the requested model
        self.unload_model()

        # Load and cache the model and tokenizer
        model_path = self.config.repo_id
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        dtype_str = self.config.torch_dtype.lower()
        if dtype_str == "float16":
            dtype = torch.float16
        elif dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str =="bfloat32":
            dtype = torch.float32
        else:
            dtype = torch.float32
              # Default to float32 if not specified


        #if quantisized to.self() method wont work
        try:
            # Check if the model is already quantized
            if "4bit" in model_path or "8bit" in model_path:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
            elif self.config.quantize.enable:
                if self.config.quantize.type == "4bit":
                    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, torch_dtype=dtype)
                elif self.config.quantize.type == "8bit":
                    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, torch_dtype=dtype)
                else:
                    raise ValueError("Unsupported quantization type")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(self.device)

            
        except Exception as e:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        self.current_model_path = model_path
        self.current_model = model
        self.current_tokenizer = tokenizer

    def infer(self,input_text):
        inputs = self.current_tokenizer(input_text, return_tensors='pt').to(self.device)
        """Perform inference using the model and tokenizer."""
        #accomidate for different tokeniztions of different llm
        
        
        outputs = self.current_model.generate(
            inputs['input_ids'],
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            pad_token_id=self.current_tokenizer.pad_token_id,
            eos_token_id=self.current_tokenizer.eos_token_id
        )
        

        decoded_outputs = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return decoded_outputs
        # Post-process the decoded outputs
    def get_model_precision(self):
        """Get the precision of the model."""
        if self.current_model is not None:
            dtype = next(self.current_model.parameters()).dtype
            return dtype
        else:
            raise HTTPException(status_code=400, detail="Model not loaded")
    def get_model_size(self):

        """Get the size of the model in parameters."""
        if self.current_model is not None:
            num_params = sum(p.numel() for p in self.current_model.parameters())
            return num_params / 1e6  # size in millions of parameters
        else:
            raise HTTPException(status_code=400, detail="Model not loaded")
    def reload(self):
        self.unload_model()
        """Reload the model with a new configuration."""
        new_config = load_config("config.yaml")
        self.config = new_config
        self.load_model()


# Load the configuration from a yaml file
def load_config(config_path: str) -> ModelConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return ModelConfig(**config_dict['model'])

# Load configuration

try:#for local
    config = load_config("config.yaml")
except Exception as e:#for docker
    config = load_config("app/config.yaml")
model_handler = ModelHandler(config=config)
# Load model and tokenizer based on the configuration


@app.post("/generate/")
def generate_text(prompt_model: PromptModel):
    try:
        prompt = prompt_model.prompt
        output_text = model_handler.infer(prompt)
        return {"generated_text": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_status/")
def model_status():
    if model_handler.current_model is not None:
        return {"status": "Model is loaded", "model_path": model_handler.current_model_path}
    else:
        return {"status": "Model is not loaded"}

@app.get("/")
def read_root():
    return {"message": "API is running"}
@app.get("/gpu/")
def status_gpu_check()->dict[str,str]:
    gpu_msg="available" if torch.cuda.is_available() else "not available"
    return{"gpu status ":gpu_msg}
# Start the FastAPI app if this file is executed directly
@app.get("/model_precision/")
def model_precision():
    try:
        precision = model_handler.get_model_precision()
        return {"model_precision": str(precision)}
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
@app.get("/model_size/")
def model_size():
    try:
        size = model_handler.get_model_size()
        return {"model_size_million_parameters": size}
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
@app.post("/reload_config/")
def reload_config():
    try:
        model_handler.reload()
        return {"status": "Configuration reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1420)