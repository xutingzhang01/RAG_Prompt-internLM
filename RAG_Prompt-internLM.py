'''
    基于InternLM实现诸葛孔明智能问答数据集生成助手项目
'''
import json
import re
import os
import datetime
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, RagTokenForGeneration, RagConfig, DPRContextEncoder, DPRQuestionEncoder
from transformers import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast, BitsAndBytesConfig
import torch
from datasets import load_dataset
import logging
from typing import List, Dict
import pdfplumber
from docx import Document  # 新增支持.docx 文件
import pandas as pd  # 新增支持.xlsx 文件
import io

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='/root/ft/app.log')
logger = logging.getLogger(__name__)

# 初始化模型和 tokenizer
MODEL_PATH = "/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)

class QAGenerator:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = 2048  # 假设的最大长度

    def _generate_with_retry(self, prompt, max_length, max_new_tokens, top_k, top_p):
        logger.info(f"Generating response for prompt: {prompt[:100]}...")  # 记录提示的前100个字符
        try:
            with torch.no_grad():
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=0.7,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                return decoded_output
        except Exception as e:
            logging.error(f"生成时发生错误: {str(e)}")
            return "无法生成，请检查输入。"

    def adjust_max_new_tokens(self, input_length):
        logger.info(f"Adjusting max new tokens based on input length: {input_length}")
        return min(input_length // 10, 32)

    def preprocess_context(self, text):
        logger.info("Preprocessing context...")
        # 预处理逻辑
        preprocess_context = re.sub(r'[^\w\s]', '', text.lower())
        return preprocess_context

    def postprocess_outlines(self, outlines, num_outlines):
        # 后处理大纲
        outlines_list = outlines.split('\n')
        
        # 保留前 num_outlines 个大纲
        outlines_list = outlines_list[:num_outlines]

        # 清理每个大纲
        cleaned_outlines = [outline.strip() for outline in outlines_list if outline.strip()]
        
        return cleaned_outlines

    def read_file(self, file_path):
        # 确保文件路径有效
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 '{file_path}' 不存在")
        # 读取文件内容
        _, ext = os.path.splitext(file_path)
        if ext == '.pdf':
            return self.read_pdf(file_path)
        elif ext == '.docx':
            return self.read_docx(file_path)
        elif ext == '.txt':
            return self.read_text(file_path)
        elif ext == '.xlsx':
            return self.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def read_pdf(self, file_path: str) -> str:
        with pdfplumber.open(file_path) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages)
        return text

    def read_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        text = '\n'.join(para.text for para in doc.paragraphs)
        return text

    def read_text(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def read_excel(self, file_path: str) -> str:
        df = pd.read_excel(file_path, engine='openpyxl')
        text = "\n".join(df.iloc[:, 0].astype(str))
        return text

    def generate_questions_and_answers(self, text, custom_prompt, num_outlines, user_questions=None):
        logger.info("Generating questions and answers...")
        # 减少 max_new_tokens 来降低内存使用
        max_new_tokens = self.adjust_max_new_tokens(len(text))

        # 动态调整 top_k 和 top_p 来降低多样性并减少内存使用
        top_k = 10 if max_new_tokens <= 64 else 50
        top_p = 0.8 if max_new_tokens <= 64 else 0.95

        # 对用户上传的文档进行更详细的预处理
        preprocessed_context = self.preprocess_context(text)

        qa_pairs=[]
        # 生成问题
        if user_questions:
            logger.info("Using user-provided questions.")
            for question in user_questions:
                # 生成回答
                answer_prompt = f"请根据文档内容或自定义提示，回答问题：'{question}'。"
                if custom_prompt:
                    answer_prompt += f"\n\n自定义提示: {custom_prompt}"

                answer = self._generate_with_retry(answer_prompt, self.max_length, max_new_tokens, top_k, top_p)

                # 转换为 Alpaca 格式
                alpaca_format = {
                    "instruction": question,
                    "input": "",
                    "output": answer if answer != "" else "模型还在学习中，系统无法做出回答，谢谢您的支持"
                }
                qa_pairs.append(alpaca_format)
        else:
            logger.info("Generating outlines.")
            # 生成问题
            outlines_prompt = (f"请根据提供的文档结构和内容生成 {num_outlines} 个问题。\n"
                               f"{preprocessed_context}\n\n附加信息:\n{custom_prompt}")
            if custom_prompt:
                outlines_prompt += f"\n\n自定义提示: {custom_prompt}"

            outlines = self._generate_with_retry(outlines_prompt, self.max_length, max_new_tokens, top_k, top_p)
            # 确保问题以问号结束
            outlines = re.sub(r'(?<!\?)\s*$', '?', outlines)

            # 处理问题输出
            outlines_list = self.postprocess_outlines(outlines, num_outlines)

            # 生成问题和答案
            for outline in outlines_list:
                # 直接使用大纲作为问题
                question = outline

                # 生成回答
                answer_prompt = f"请根据文档内容或自定义提示，回答问题：'{question}'。"
                if custom_prompt:
                    answer_prompt += f"\n\n自定义提示: {custom_prompt}"

                answer = self._generate_with_retry(answer_prompt, self.max_length, max_new_tokens, top_k, top_p)

                # 转换为 Alpaca 格式
                alpaca_format = {
                    "instruction": question,
                    "input": "",
                    "output": answer if answer != "" else "模型还在学习中，系统无法做出回答，谢谢您的支持"
                }
                qa_pairs.append(alpaca_format)
                
        logger.info(f"Generated {len(qa_pairs)} QA pairs.")
        return qa_pairs

    def parse_qa_pair(self, qa_pair: str) -> Dict[str, str]:
        match = re.search(r"(?P<question>.*?\n)\s*(?P<answer>.*)", qa_pair, re.DOTALL)
        if match:
            question = match.group('question').strip()
            answer = match.group('answer').strip()
            return {"question": question, "answer": answer}
        else:
            return {"question": "", "answer": qa_pair.strip()}

def create_gradio_ui():
    def process_file(file, custom_prompt, num_outlines, user_questions):
        if not file:
            logger.warning("No file uploaded.")
            return ""

        text = generator.read_file(file.name)
        qa_pairs = generator.generate_questions_and_answers(text, custom_prompt, num_outlines, user_questions=None)
        json_data = json.dumps(qa_pairs, indent=4, ensure_ascii=False)
        return json_data

    def download_json(json_data):
        if not isinstance(json_data, str):
            raise TypeError("json_data 必须是字符串类型")

        filename = f"qa_pairs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_data)
        return filename, json_data

    with gr.Blocks() as demo:
        gr.Markdown("# 诸葛孔明智能数据集问答生成系统")
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="上传文件", file_types=[".pdf", ".docx", ".txt", ".xlsx"])
                custom_prompt_input = gr.Textbox(label="自定义提示", placeholder="可选")
                num_outlines_input = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="问题数量")
                user_questions_input = gr.Textbox(label="自定义问题（一行一个问题）", placeholder="可选")
                submit_button = gr.Button("生成问题和答案对")

                with gr.Column():
                    output_text = gr.Textbox(label="输出 JSON 数据")
                    download_button = gr.Button("下载 JSON 文件")
                    # 添加条件判断，如果没有文件上传则显示提示信息
                    output_text.change(lambda x: "请上传文件以继续操作。" if x == "" else x, inputs=[output_text], outputs=[output_text])

                submit_button.click(fn=process_file, inputs=[file_input, custom_prompt_input, num_outlines_input, user_questions_input], outputs=output_text)
                download_button.click(fn=download_json, inputs=[output_text], outputs=[gr.File(), gr.Textbox()])

                return demo

if __name__ == "__main__":
    generator = QAGenerator(MODEL_PATH)
    demo = create_gradio_ui()
    demo.launch(share=True)
