from transformers import pipeline
import torch
import gradio as gr
from typing import List, Dict

pipe = pipeline(
    "text-generation",
    model="sarvamai/sarvam-1",
    torch_dtype=torch.float16,
    device_map="auto",
)

class ChatHistory:
    def __init__(self, max_tokens: int = 1024):
        self.history: List[Dict[str, str]] = []
        self.max_tokens = max_tokens
        
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self._trim_history()
    
    def _trim_history(self):
        while self._calculate_tokens() > self.max_tokens and len(self.history) > 1:
            self.history.pop(0)
    
    def _calculate_tokens(self) -> int:
        return sum(len(pipe.tokenizer.encode(msg["content"])) for msg in self.history)
    
    def get_context_prompt(self) -> str:
        return "\n".join(f"### {msg['role']}: {msg['content']}" for msg in self.history)

chat_history = ChatHistory()

def generate_response(message: str, history: List[List[str]]) -> str:
    try:
        chat_history.history = []
        for user_msg, bot_msg in history:
            chat_history.add_message("User", user_msg)
            chat_history.add_message("Assistant", bot_msg)
        
        chat_history.add_message("User", message)
        
        system_prompt = """### System: निम्नलिखित प्रश्न का उत्तर केवल हिंदी में दें। 
आपको अनिवार्य रूप से हिंदी में ही जवाब देना है।"""
        
        full_prompt = f"""{system_prompt}
{chat_history.get_context_prompt()}
### Assistant: मैं आपके प्रश्न का उत्तर हिंदी में दे रहा हूं:"""
        
        response = pipe(
            full_prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        
        output = response[0]['generated_text'].split("### Assistant:")[-1].strip()
        
        if not any(0x0900 <= ord(c) <= 0x097F for c in output):
            raise ValueError("Non-Hindi response generated")
            
        chat_history.add_message("Assistant", output)
        
        return output
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return "क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।"

iface = gr.ChatInterface(
    fn=generate_response,
    title="Sarvam AI",
    description="An LLM made for Indic Languages (हिंदी in this specific case)",
    examples=[
        ["आप कौन हैं?"],
        ["महाभारत के रचयिता कौन हैं?"],
        ["हिंदी में एक कहानी सुनाइए"],
    ],
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="अपना प्रश्न हिंदी में टाइप करें...", container=False, scale=7),
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)