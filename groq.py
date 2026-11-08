import config
from openai import OpenAI
GROQ_URL="https://api.groq.com/openai/v1"
MODELS=getattr(config ,"GROQ_MODELS",["opeanai/gpt-oss-20b","opeanai/gpt-oss-7b","opeanai/gpt-oss-3b","opeanai/gpt-oss-1b","openai/gpt-oss-120b","qwen/qwen-7b","qwen/qwen-14b","qwen/qwen-14b-chat","qwen/qwen-7b-chat","qwen/qwen-7b-v1.0","qwen/qwen-14b-v1.0","qwen/qwen-7b-v1.1","qwen/qwen-14b-v1.1","qwen/qwen-7b-v1.2","qwen/qwen-14b-v1.2","qwen/qwen-7b-v1.3","qwen/qwen-14b-v1.3","qwen/qwen-7b-v1.4","qwen/qwen3.8-27b","allam-2-7b"])
def generate_response(prompt:str,temperature:float=0.3,max_tokens:int=512)->str:
    key=getattr(config,"GROQ_API_KEY",None)
    if key is None:
        raise ValueError("GROQ_API_KEY is not set in config")
    c=OpenAI(api_key=key,base_url=GROQ_URL)
    last_err=None
    for m in MODELS:
        try:
            r=c.chat.completions.create(
                model=m,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            last_err=e
    return(
        "GROQ MODEL failed\n"
        f"Tried models: {MODELS}\n"
        "Fix\n"
        "1)Switch to hf by importing hf.py\n"
        "2)Replace Groq model in groq.py"
    )