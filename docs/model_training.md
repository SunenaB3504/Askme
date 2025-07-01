# Model Training Guide - AskMe Voice Assistant

## Overview

This guide covers the complete process of training and fine-tuning models for the AskMe voice assistant, including LLM fine-tuning with QLoRA, voice cloning for TTS, and custom ASR adaptation.

## Table of Contents

1. [LLM Fine-tuning with QLoRA](#llm-fine-tuning-with-qlora)
2. [Custom Dataset Preparation](#custom-dataset-preparation)
3. [Training Process](#training-process)
4. [Model Optimization and Quantization](#model-optimization-and-quantization)
5. [TTS Voice Customization](#tts-voice-customization)
6. [ASR Model Adaptation](#asr-model-adaptation)
7. [Evaluation and Benchmarking](#evaluation-and-benchmarking)

## LLM Fine-tuning with QLoRA

### Prerequisites

```bash
# Install training dependencies
pip install accelerate bitsandbytes peft trl datasets
pip install transformers[training]
pip install wandb  # For experiment tracking (optional)
```

### 1. Environment Setup

```python
# training/setup_training_env.py
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

def setup_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Setup base model with 4-bit quantization for QLoRA training
    """
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def setup_lora_config():
    """
    Configure LoRA for parameter-efficient fine-tuning
    """
    lora_config = LoraConfig(
        r=64,  # Rank of adaptation
        lora_alpha=16,  # LoRA scaling parameter
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return lora_config
```

### 2. Dataset Preparation

#### Conversational Dataset Format

```python
# data/prepare_training_data.py
import json
from datasets import Dataset

def create_conversation_dataset():
    """
    Create a dataset in the required format for instruction tuning
    """
    conversations = [
        {
            "instruction": "You are AskMe, a helpful offline voice assistant.",
            "input": "What can you help me with?",
            "output": "I'm AskMe, your privacy-focused offline assistant! I can help you with various tasks like answering questions, planning activities, creative writing, problem-solving, and general conversation. Since I work entirely offline, your data stays completely private on your device. What would you like assistance with today?"
        },
        {
            "instruction": "You are AskMe, a helpful offline voice assistant.",
            "input": "How do I set a reminder?",
            "output": "I can help you think about setting reminders, but as an offline assistant, I don't have access to your device's reminder system. However, I can suggest several options: you could use your phone's built-in reminder app, write a note, set an alarm, or use voice commands with your device's assistant. What kind of reminder did you want to set?"
        },
        {
            "instruction": "You are AskMe, a helpful offline voice assistant.",
            "input": "Explain machine learning in simple terms",
            "output": "Machine learning is like teaching a computer to recognize patterns and make decisions, similar to how humans learn from experience. Instead of programming specific rules, we show the computer lots of examples and let it figure out the patterns on its own. For instance, by showing thousands of photos labeled 'cat' or 'dog', the computer learns to distinguish between cats and dogs in new photos. It's used everywhere - from email spam filters to voice assistants like me!"
        }
    ]
    
    return Dataset.from_list(conversations)

def format_prompt(example):
    """
    Format conversation data into a prompt template
    """
    prompt = f"""<s>[INST] {example['instruction']}

{example['input']} [/INST] {example['output']}</s>"""
    return {"text": prompt}

# Load and format dataset
dataset = create_conversation_dataset()
dataset = dataset.map(format_prompt)
```

#### Domain-Specific Training Data

```python
# data/domain_datasets.py
class DomainDatasetCreator:
    def __init__(self):
        self.domains = {
            "education": self.create_education_data,
            "healthcare": self.create_healthcare_data,
            "accessibility": self.create_accessibility_data
        }
    
    def create_education_data(self):
        """Educational domain conversations"""
        return [
            {
                "instruction": "You are AskMe, an educational assistant.",
                "input": "Explain photosynthesis for a 10-year-old",
                "output": "Photosynthesis is like how plants eat! Just like you need food to grow strong, plants need food too. But instead of eating with their mouths, plants use their leaves to 'eat' sunlight, air, and water. They mix these together like a recipe and make their own food (sugar) to grow big and strong. As a bonus, they also make oxygen - the air we breathe! So plants are like little food factories that also help us breathe."
            },
            {
                "instruction": "You are AskMe, an educational assistant.",
                "input": "Help me understand fractions with pizza",
                "output": "Great idea using pizza! Imagine you have a whole pizza - that's 1. If you cut it in half, each piece is 1/2 (one half). If you cut it into 4 equal slices, each slice is 1/4 (one quarter). The bottom number tells you how many equal pieces the whole pizza was cut into, and the top number tells you how many pieces you have. So 3/4 means you have 3 slices out of 4 total slices. Want to try some fraction problems with pizza?"
            }
        ]
    
    def create_healthcare_data(self):
        """Healthcare domain conversations (non-diagnostic)"""
        return [
            {
                "instruction": "You are AskMe, a health-conscious assistant. Never provide medical diagnosis.",
                "input": "I have a headache, what should I do?",
                "output": "I understand you're experiencing a headache, which can be uncomfortable. While I can't provide medical advice, here are some general wellness suggestions that many people find helpful: staying hydrated, resting in a quiet dark room, gentle neck stretches, or applying a cold or warm compress. If your headache is severe, persistent, or unusual for you, it's always best to consult with a healthcare professional who can properly assess your situation."
            }
        ]
    
    def create_accessibility_data(self):
        """Accessibility-focused conversations"""
        return [
            {
                "instruction": "You are AskMe, an accessibility-focused assistant.",
                "input": "Help me navigate my computer with voice commands",
                "output": "I'd be happy to help you learn about voice navigation! Most operating systems have built-in accessibility features. On Windows, you can use Speech Recognition or Narrator. On Mac, try Voice Control or VoiceOver. Common voice commands include 'click [button name]', 'scroll down', 'open [application]', and 'select [text]'. Would you like me to explain specific commands for your operating system?"
            }
        ]
```

### 3. Training Configuration

```python
# training/train_model.py
from transformers import TrainingArguments, Trainer
import wandb

def setup_training_arguments():
    """
    Configure training parameters for QLoRA fine-tuning
    """
    return TrainingArguments(
        output_dir="./results/askme-mistral-7b",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="wandb",  # Optional: for experiment tracking
        run_name="askme-training"
    )

def train_model():
    """
    Main training function
    """
    # Initialize wandb (optional)
    wandb.init(project="askme-voice-assistant", name="mistral-7b-finetune")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    dataset = load_dataset("json", data_files="data/training_data.json")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=2048,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Split dataset
    train_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = setup_training_arguments()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    model.save_pretrained("./models/askme-mistral-7b-lora")
    tokenizer.save_pretrained("./models/askme-mistral-7b-lora")

if __name__ == "__main__":
    train_model()
```

### 4. Training Script Execution

```bash
# Start training
python training/train_model.py

# Monitor training with wandb (optional)
wandb login
```

Expected training output:
```
trainable params: 83,886,080 || all params: 7,324,417,536 || trainable%: 1.1449
{'train_runtime': 2847.86, 'train_samples_per_second': 1.123}
{'eval_loss': 0.8234, 'eval_runtime': 145.32, 'eval_samples_per_second': 2.456}
```

## Model Optimization and Quantization

### 1. Merge LoRA Adapters

```python
# optimization/merge_adapters.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_lora_adapters(base_model_path, lora_path, output_path):
    """
    Merge LoRA adapters back into the base model
    """
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Merge adapters
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

# Usage
merge_lora_adapters(
    "mistralai/Mistral-7B-Instruct-v0.2",
    "./models/askme-mistral-7b-lora",
    "./models/askme-mistral-7b-merged"
)
```

### 2. Convert to GGUF Format

```python
# optimization/convert_to_gguf.py
import subprocess
import os

def convert_to_gguf(model_path, output_path, quantization="q4_0"):
    """
    Convert merged model to GGUF format for llama.cpp
    """
    # Download conversion script if not exists
    convert_script = "convert.py"
    if not os.path.exists(convert_script):
        subprocess.run([
            "wget", 
            "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert.py"
        ])
    
    # Convert to GGUF
    cmd = [
        "python", convert_script,
        "--input-dir", model_path,
        "--output-file", f"{output_path}/askme-{quantization}.gguf",
        "--quantization", quantization
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Model converted to GGUF format: {output_path}")

# Convert with different quantization levels
quantizations = ["q4_0", "q5_0", "q8_0"]
for quant in quantizations:
    convert_to_gguf(
        "./models/askme-mistral-7b-merged",
        "./models/gguf",
        quant
    )
```

## TTS Voice Customization

### 1. Voice Cloning Setup

```python
# tts/voice_cloning.py
from TTS.api import TTS
import os
import glob

class VoiceCloner:
    def __init__(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    
    def prepare_voice_data(self, audio_dir, speaker_name):
        """
        Prepare audio samples for voice cloning
        """
        audio_files = glob.glob(f"{audio_dir}/*.wav")
        
        # Validate audio requirements
        for audio_file in audio_files:
            duration = self.get_audio_duration(audio_file)
            if duration < 6 or duration > 12:
                print(f"Warning: {audio_file} duration {duration}s not optimal")
        
        return audio_files
    
    def clone_voice(self, reference_audio, target_text, output_path):
        """
        Generate speech with cloned voice
        """
        self.tts.tts_to_file(
            text=target_text,
            speaker_wav=reference_audio,
            language="en",
            file_path=output_path
        )
    
    def create_voice_profile(self, audio_samples, speaker_name):
        """
        Create a reusable voice profile
        """
        # Process audio samples
        processed_samples = []
        for sample in audio_samples:
            # Audio preprocessing
            processed = self.preprocess_audio(sample)
            processed_samples.append(processed)
        
        # Save voice profile
        profile_path = f"./models/voices/{speaker_name}"
        os.makedirs(profile_path, exist_ok=True)
        
        # Store voice embeddings
        self.save_voice_embeddings(processed_samples, profile_path)
        
        return profile_path
```

### 2. Voice Training Script

```bash
#!/bin/bash
# scripts/train_custom_voice.sh

# Collect voice samples
echo "Recording voice samples for training..."
python tts/record_voice_samples.py --speaker "user_voice" --samples 20

# Train custom voice
echo "Training custom voice model..."
python tts/train_voice.py \
    --input_dir "./data/voice_samples/user_voice" \
    --output_dir "./models/voices/user_voice" \
    --epochs 100 \
    --batch_size 8

# Test voice quality
echo "Testing voice quality..."
python tts/test_voice.py \
    --voice_model "./models/voices/user_voice" \
    --test_text "Hello, this is a test of my custom voice."
```

## ASR Model Adaptation

### 1. Custom Vocabulary Addition

```python
# asr/adapt_whisper.py
import whisper
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperAdapter:
    def __init__(self, model_name="openai/whisper-base"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    def add_custom_vocabulary(self, custom_words):
        """
        Add domain-specific vocabulary to the model
        """
        # Add new tokens to tokenizer
        new_tokens = []
        for word in custom_words:
            if word not in self.processor.tokenizer.vocab:
                new_tokens.append(word)
        
        if new_tokens:
            self.processor.tokenizer.add_tokens(new_tokens)
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
        
        return len(new_tokens)
    
    def fine_tune_for_accent(self, audio_dataset, transcriptions):
        """
        Fine-tune Whisper for specific accents or speaking styles
        """
        # Prepare training data
        train_data = self.prepare_training_data(audio_dataset, transcriptions)
        
        # Configure training
        training_args = {
            "num_epochs": 5,
            "learning_rate": 1e-5,
            "batch_size": 16,
            "warmup_steps": 100
        }
        
        # Fine-tune model
        self.train_model(train_data, training_args)
    
    def save_adapted_model(self, output_path):
        """
        Save the adapted model
        """
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

# Usage
adapter = WhisperAdapter()

# Add medical/technical terms
medical_terms = [
    "electrocardiogram", "stethoscope", "bronchoscopy",
    "pneumonia", "antibiotics", "hypertension"
]

adapter.add_custom_vocabulary(medical_terms)
adapter.save_adapted_model("./models/whisper-medical")
```

## Evaluation and Benchmarking

### 1. LLM Evaluation

```python
# evaluation/evaluate_llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json

class LLMEvaluator:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def evaluate_conversation_quality(self, test_conversations):
        """
        Evaluate conversation quality metrics
        """
        results = {
            "relevance_scores": [],
            "coherence_scores": [],
            "response_lengths": [],
            "generation_times": []
        }
        
        for conversation in test_conversations:
            start_time = time.time()
            
            # Generate response
            inputs = self.tokenizer(conversation["input"], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_time = time.time() - start_time
            
            # Calculate metrics
            relevance = self.calculate_relevance(conversation["input"], response)
            coherence = self.calculate_coherence(response)
            
            results["relevance_scores"].append(relevance)
            results["coherence_scores"].append(coherence)
            results["response_lengths"].append(len(response.split()))
            results["generation_times"].append(generation_time)
        
        return self.summarize_results(results)
    
    def benchmark_performance(self):
        """
        Benchmark model performance
        """
        metrics = {
            "avg_tokens_per_second": 0,
            "memory_usage_gb": 0,
            "first_token_latency_ms": 0
        }
        
        # Run performance tests
        test_prompts = [
            "Explain quantum computing",
            "Help me plan my day",
            "What is machine learning?"
        ]
        
        for prompt in test_prompts:
            perf_metrics = self.measure_performance(prompt)
            for key in metrics:
                metrics[key] += perf_metrics[key]
        
        # Average the results
        for key in metrics:
            metrics[key] /= len(test_prompts)
        
        return metrics
```

### 2. End-to-End System Evaluation

```python
# evaluation/system_evaluation.py
import time
import psutil
import whisper
from TTS.api import TTS

class SystemEvaluator:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.setup_components()
    
    def setup_components(self):
        """Initialize all system components"""
        self.whisper_model = whisper.load_model(self.config["asr"]["model"])
        self.tts_model = TTS(self.config["tts"]["model"])
        # LLM setup...
    
    def evaluate_end_to_end_pipeline(self, test_audio_files):
        """
        Evaluate complete audio-to-audio pipeline
        """
        results = []
        
        for audio_file in test_audio_files:
            start_time = time.time()
            
            # ASR
            asr_start = time.time()
            transcript = self.whisper_model.transcribe(audio_file)
            asr_time = time.time() - asr_start
            
            # LLM Processing
            llm_start = time.time()
            response = self.process_with_llm(transcript["text"])
            llm_time = time.time() - llm_start
            
            # TTS
            tts_start = time.time()
            self.tts_model.tts_to_file(
                text=response,
                file_path=f"temp_output_{len(results)}.wav"
            )
            tts_time = time.time() - tts_start
            
            total_time = time.time() - start_time
            
            results.append({
                "audio_file": audio_file,
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": tts_time,
                "total_time": total_time,
                "transcript": transcript["text"],
                "response": response,
                "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
            })
        
        return self.analyze_results(results)
    
    def analyze_results(self, results):
        """
        Analyze and summarize evaluation results
        """
        summary = {
            "avg_asr_time": sum(r["asr_time"] for r in results) / len(results),
            "avg_llm_time": sum(r["llm_time"] for r in results) / len(results),
            "avg_tts_time": sum(r["tts_time"] for r in results) / len(results),
            "avg_total_time": sum(r["total_time"] for r in results) / len(results),
            "avg_memory_usage": sum(r["memory_usage"] for r in results) / len(results),
            "real_time_factor": sum(r["total_time"] for r in results) / sum(self.get_audio_duration(r["audio_file"]) for r in results)
        }
        
        return summary
```

### 3. Running Evaluations

```bash
# Run complete evaluation suite
python evaluation/run_evaluation.py \
    --model_path "./models/askme-mistral-7b-merged" \
    --test_data "./data/evaluation_set.json" \
    --output_dir "./evaluation_results"

# Generate evaluation report
python evaluation/generate_report.py \
    --results_dir "./evaluation_results" \
    --output_format "html"
```

Expected evaluation output:
```
=== AskMe Model Evaluation Results ===
ASR Performance:
  - Word Error Rate: 3.2%
  - Real-time Factor: 0.8x

LLM Performance:
  - Average Response Time: 280ms
  - Relevance Score: 4.2/5.0
  - Coherence Score: 4.1/5.0

TTS Performance:
  - Synthesis Time: 1.2x real-time
  - Voice Quality (MOS): 4.3/5.0

System Performance:
  - Memory Usage: 6.5GB
  - CPU Usage: 45% average
  - Total Pipeline Latency: 750ms
```

This comprehensive training guide provides all the necessary steps to create and customize your offline voice assistant models, ensuring optimal performance while maintaining privacy and security.

## Educational Assistant for School Books (Nia's Learning Companion)

### Overview

This section provides a complete guide to create a personalized educational assistant for your 9-year-old daughter Nia using her school book PDFs. The system will:

- Extract content from PDF textbooks
- Generate age-appropriate summaries
- Create various types of questions (multiple choice, fill-in-the-blank, true/false)
- Provide interactive learning experiences through voice and text

### 1. PDF Processing and Content Extraction

```python
# education/pdf_processor.py
import PyPDF2
import fitz  # PyMuPDF for better text extraction
import re
from pathlib import Path
import json

class SchoolBookProcessor:
    def __init__(self):
        self.chapter_patterns = [
            r"Chapter\s+(\d+)",
            r"Unit\s+(\d+)",
            r"Lesson\s+(\d+)",
            r"Section\s+(\d+)"
        ]
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF with chapter detection
        """
        doc = fitz.open(pdf_path)
        chapters = {}
        current_chapter = "Introduction"
        current_text = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Check for chapter headers
            chapter_match = self.detect_chapter(text)
            if chapter_match:
                # Save previous chapter
                if current_text:
                    chapters[current_chapter] = "\n".join(current_text)
                
                current_chapter = chapter_match
                current_text = []
            
            # Clean and add text
            cleaned_text = self.clean_text(text)
            if cleaned_text:
                current_text.append(cleaned_text)
        
        # Save last chapter
        if current_text:
            chapters[current_chapter] = "\n".join(current_text)
        
        doc.close()
        return chapters
    
    def detect_chapter(self, text):
        """
        Detect chapter headers in text
        """
        for pattern in self.chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"Chapter {match.group(1)}"
        return None
    
    def clean_text(self, text):
        """
        Clean extracted text for better processing
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        filtered_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(filtered_lines)
    
    def extract_key_concepts(self, text):
        """
        Extract key concepts and vocabulary from chapter text
        """
        # Simple keyword extraction based on formatting clues
        key_concepts = []
        
        # Look for bold text patterns (common in textbooks)
        bold_patterns = [
            r'\*\*(.*?)\*\*',  # **bold**
            r'__(.*?)__',      # __bold__
            r'Key Terms?:?\s*(.*?)(?:\n|$)',  # Key Terms: ...
            r'Important:?\s*(.*?)(?:\n|$)'    # Important: ...
        ]
        
        for pattern in bold_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_concepts.extend(matches)
        
        # Clean and deduplicate
        key_concepts = list(set([concept.strip() for concept in key_concepts if len(concept.strip()) > 2]))
        
        return key_concepts
```

### 2. Content Processing for 9-Year-Old Learning

```python
# education/content_processor.py
import openai
from transformers import pipeline
import json

class EducationalContentProcessor:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.age_level = "9-year-old"
    
    def create_age_appropriate_summary(self, chapter_text, chapter_title):
        """
        Create summaries appropriate for a 9-year-old
        """
        # Split long text into chunks for processing
        chunks = self.split_text(chapter_text, max_length=1000)
        summaries = []
        
        for chunk in chunks:
            # Create summary
            summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        # Combine and simplify
        combined_summary = " ".join(summaries)
        simplified_summary = self.simplify_for_age(combined_summary)
        
        return {
            "chapter": chapter_title,
            "summary": simplified_summary,
            "key_points": self.extract_key_points(simplified_summary),
            "difficulty_level": "Elementary (Grade 4-5)"
        }
    
    def simplify_for_age(self, text):
        """
        Simplify text for 9-year-old comprehension
        """
        # Replace complex words with simpler alternatives
        replacements = {
            "utilize": "use",
            "demonstrate": "show",
            "comprehend": "understand",
            "substantial": "big",
            "facilitate": "help",
            "subsequent": "next",
            "approximate": "about",
            "fundamental": "basic",
            "significant": "important",
            "consequently": "so"
        }
        
        simplified = text
        for complex_word, simple_word in replacements.items():
            simplified = re.sub(r'\b' + complex_word + r'\b', simple_word, simplified, flags=re.IGNORECASE)
        
        return simplified
    
    def extract_key_points(self, text):
        """
        Extract 3-5 key learning points
        """
        sentences = text.split('.')
        # Select sentences that contain important concepts
        key_sentences = []
        
        important_keywords = ['important', 'key', 'main', 'because', 'remember', 'always', 'never']
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in important_keywords):
                key_sentences.append(sentence.strip())
        
        return key_sentences[:5]  # Return top 5
    
    def split_text(self, text, max_length=1000):
        """
        Split text into manageable chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
```

### 3. Question Generation System

```python
# education/question_generator.py
import random
import json
from typing import List, Dict

class EducationalQuestionGenerator:
    def __init__(self):
        self.question_templates = {
            "multiple_choice": [
                "What is {concept}?",
                "Which of the following describes {concept}?",
                "What happens when {action}?",
                "Why is {concept} important?"
            ],
            "fill_in_blank": [
                "{sentence_with_blank}",
                "Complete this sentence: {partial_sentence} ____.",
                "Fill in the missing word: {sentence_with_blank}"
            ],
            "true_false": [
                "{statement}",
                "Is this true or false: {statement}",
                "True or False: {statement}"
            ],
            "short_answer": [
                "Explain why {concept} is important.",
                "Describe what {concept} means.",
                "How does {process} work?",
                "What would happen if {scenario}?"
            ]
        }
    
    def generate_questions_from_chapter(self, chapter_content, chapter_title, num_questions=10):
        """
        Generate various types of questions from chapter content
        """
        questions = {
            "chapter": chapter_title,
            "multiple_choice": [],
            "fill_in_blank": [],
            "true_false": [],
            "short_answer": []
        }
        
        # Extract key concepts and facts
        key_concepts = self.extract_concepts(chapter_content)
        facts = self.extract_facts(chapter_content)
        
        # Generate different types of questions
        questions["multiple_choice"] = self.create_multiple_choice(key_concepts, facts)[:3]
        questions["fill_in_blank"] = self.create_fill_in_blank(chapter_content)[:3]
        questions["true_false"] = self.create_true_false(facts)[:2]
        questions["short_answer"] = self.create_short_answer(key_concepts)[:2]
        
        return questions
    
    def create_multiple_choice(self, concepts, facts):
        """
        Create multiple choice questions
        """
        questions = []
        
        for concept in concepts[:3]:  # Limit to 3 concepts
            question_text = f"What is {concept}?"
            
            # Create plausible wrong answers
            correct_answer = self.get_concept_definition(concept, facts)
            wrong_answers = self.generate_distractors(concept, concepts)
            
            choices = [correct_answer] + wrong_answers[:3]
            random.shuffle(choices)
            
            questions.append({
                "question": question_text,
                "choices": choices,
                "correct_answer": correct_answer,
                "explanation": f"{concept} is an important concept in this chapter."
            })
        
        return questions
    
    def create_fill_in_blank(self, chapter_content):
        """
        Create fill-in-the-blank questions
        """
        questions = []
        sentences = chapter_content.split('.')
        
        for sentence in sentences[:5]:  # Check first 5 sentences
            if len(sentence.split()) > 8:  # Ensure sentence is long enough
                words = sentence.split()
                # Remove a key word (noun or important term)
                for i, word in enumerate(words):
                    if len(word) > 4 and word.isalpha():  # Find important words
                        blank_sentence = " ".join(words[:i] + ["____"] + words[i+1:])
                        questions.append({
                            "question": f"Fill in the blank: {blank_sentence}",
                            "answer": word,
                            "sentence": sentence
                        })
                        break
        
        return questions[:3]
    
    def create_true_false(self, facts):
        """
        Create true/false questions
        """
        questions = []
        
        for fact in facts[:2]:  # Create 2 true/false questions
            # Create a true statement
            questions.append({
                "statement": fact,
                "answer": True,
                "explanation": "This statement is directly from the chapter."
            })
            
            # Create a false statement by modifying the fact
            false_statement = self.create_false_statement(fact)
            questions.append({
                "statement": false_statement,
                "answer": False,
                "explanation": "This statement contains incorrect information."
            })
        
        return questions
    
    def create_short_answer(self, concepts):
        """
        Create short answer questions
        """
        questions = []
        
        for concept in concepts[:2]:
            questions.append({
                "question": f"Explain what {concept} means in your own words.",
                "sample_answer": f"{concept} is an important idea that...",
                "keywords": [concept.lower()]
            })
        
        return questions
    
    def extract_concepts(self, text):
        """
        Extract key concepts from text
        """
        # Simple concept extraction - look for capitalized terms and repeated words
        words = text.split()
        concepts = []
        
        # Find capitalized words (likely proper nouns or important terms)
        for word in words:
            if word[0].isupper() and len(word) > 3 and word.isalpha():
                concepts.append(word)
        
        # Count frequency and return most common
        concept_counts = {}
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Return top concepts
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, count in sorted_concepts[:10]]
    
    def extract_facts(self, text):
        """
        Extract factual statements from text
        """
        sentences = text.split('.')
        facts = []
        
        # Look for sentences that contain factual information
        fact_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'contains', 'includes']
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in fact_indicators):
                if len(sentence.split()) > 5:  # Ensure it's a substantial sentence
                    facts.append(sentence.strip())
        
        return facts[:5]  # Return top 5 facts
    
    def get_concept_definition(self, concept, facts):
        """
        Find definition for a concept from facts
        """
        for fact in facts:
            if concept.lower() in fact.lower():
                return fact.strip()
        
        return f"{concept} is an important term from this chapter."
    
    def generate_distractors(self, concept, all_concepts):
        """
        Generate plausible wrong answers
        """
        distractors = [c for c in all_concepts if c != concept]
        return random.sample(distractors, min(3, len(distractors)))
    
    def create_false_statement(self, fact):
        """
        Create a false statement by modifying a true fact
        """
        # Simple modification - replace key words
        modifications = {
            "is": "is not",
            "are": "are not",
            "can": "cannot",
            "will": "will not",
            "always": "never",
            "hot": "cold",
            "big": "small",
            "fast": "slow"
        }
        
        modified = fact
        for original, replacement in modifications.items():
            if original in fact.lower():
                modified = fact.replace(original, replacement)
                break
        
        return modified if modified != fact else f"It is false that {fact.lower()}"
```

### 4. Training Data Generation for Nia's Assistant

```python
# education/training_data_generator.py
import json
from pathlib import Path

class NiaEducationDataGenerator:
    def __init__(self):
        self.assistant_personality = {
            "name": "Nia's Learning Helper",
            "age_group": "9-year-old",
            "tone": "friendly, encouraging, patient",
            "style": "simple language, positive reinforcement"
        }
    
    def create_training_dataset(self, processed_chapters, questions):
        """
        Create training data for Nia's educational assistant
        """
        training_data = []
        
        # Create conversation examples for each chapter
        for chapter_title, chapter_data in processed_chapters.items():
            # Summary conversations
            training_data.extend(self.create_summary_conversations(chapter_title, chapter_data))
            
            # Question-answer conversations
            training_data.extend(self.create_qa_conversations(chapter_title, questions.get(chapter_title, {})))
            
            # Encouragement and help conversations
            training_data.extend(self.create_support_conversations(chapter_title))
        
        return training_data
    
    def create_summary_conversations(self, chapter, chapter_data):
        """
        Create conversations about chapter summaries
        """
        conversations = [
            {
                "instruction": "You are Nia's helpful learning assistant. Explain things simply for a 9-year-old.",
                "input": f"Can you tell me about {chapter}?",
                "output": f"Hi Nia! Let me tell you about {chapter}. {chapter_data['summary']} The most important things to remember are: {'. '.join(chapter_data['key_points'][:3])}. Would you like me to explain anything more?"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Explain things simply for a 9-year-old.",
                "input": f"What did I learn in {chapter}?",
                "output": f"Great question, Nia! In {chapter}, you learned some really cool things! {chapter_data['summary']} You're doing such a good job learning about this topic!"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Explain things simply for a 9-year-old.",
                "input": f"I don't understand {chapter}",
                "output": f"That's okay, Nia! Learning can be tricky sometimes. Let me explain {chapter} in a simpler way: {self.simplify_further(chapter_data['summary'])} Don't worry - you're smart and you'll get it! Want me to explain any part again?"
            }
        ]
        
        return conversations
    
    def create_qa_conversations(self, chapter, chapter_questions):
        """
        Create question-answering conversations
        """
        conversations = []
        
        # Multiple choice practice
        for mc_q in chapter_questions.get('multiple_choice', []):
            conversations.append({
                "instruction": "You are Nia's helpful learning assistant. Ask questions and provide encouraging feedback.",
                "input": "Can you ask me a question about this chapter?",
                "output": f"Sure, Nia! Here's a question for you: {mc_q['question']} Your choices are: A) {mc_q['choices'][0]}, B) {mc_q['choices'][1]}, C) {mc_q['choices'][2]}, D) {mc_q['choices'][3]}. Take your time and think about it!"
            })
        
        # Short answer encouragement
        for sa_q in chapter_questions.get('short_answer', []):
            conversations.append({
                "instruction": "You are Nia's helpful learning assistant. Encourage thinking and provide hints.",
                "input": f"I don't know how to answer: {sa_q['question']}",
                "output": f"That's a great question to think about, Nia! Here's a hint: think about {sa_q['keywords'][0] if sa_q['keywords'] else 'what you learned'}. You can do this! Try to explain it in your own words, and I'll help you if you need it."
            })
        
        return conversations
    
    def create_support_conversations(self, chapter):
        """
        Create supportive and encouraging conversations
        """
        conversations = [
            {
                "instruction": "You are Nia's helpful learning assistant. Always be encouraging and supportive.",
                "input": "This is too hard for me",
                "output": "Oh Nia, I know learning can feel hard sometimes, but you are so smart and capable! Remember, every time you try, your brain gets stronger - just like when you exercise your muscles! Let's break this down into smaller, easier pieces. What part would you like me to help you with first?"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Always be encouraging and supportive.",
                "input": "I got the answer wrong",
                "output": "That's okay, Nia! Making mistakes is how we learn and grow. Even the smartest people in the world make mistakes - that's normal! What matters is that you tried. Let me help you understand the right answer, and then you'll know it for next time. You're doing great!"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Always be encouraging and supportive.",
                "input": "I'm ready for my quiz",
                "output": "Wow, Nia! I'm so proud of you for studying and being ready! You've worked really hard. Remember to take your time, read each question carefully, and trust yourself. You know more than you think you do! I believe in you!"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Always be encouraging and supportive.",
                "input": "Can you help me study?",
                "output": "Of course, Nia! I love helping you learn! We can review your chapter together, practice questions, or I can quiz you. What would you like to do? Remember, studying is like practicing for sports - the more you practice, the better you get!"
            }
        ]
        
        return conversations
    
    def simplify_further(self, text):
        """
        Simplify text even more for younger comprehension
        """
        # Break into shorter sentences
        sentences = text.split('.')
        simplified_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 10:  # If sentence is too long
                # Break it into smaller parts
                parts = sentence.split(',')
                simplified_sentences.extend([part.strip() + '.' for part in parts if part.strip()])
            else:
                simplified_sentences.append(sentence + '.')
        
        return ' '.join(simplified_sentences[:3])  # Keep it short
    
    def save_training_data(self, training_data, output_path):
        """
        Save training data in the required format
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved to {output_path}")
        print(f"Generated {len(training_data)} training examples")
```

### 5. Complete Processing Pipeline

```python
# education/process_nia_books.py
import argparse
import json
from pathlib import Path

def process_nia_school_books(pdf_directory, output_directory):
    """
    Complete pipeline to process Nia's school books
    """
    # Initialize processors
    pdf_processor = SchoolBookProcessor()
    content_processor = EducationalContentProcessor()
    question_generator = EducationalQuestionGenerator()
    data_generator = NiaEducationDataGenerator()
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all PDFs
    all_chapters = {}
    all_questions = {}
    
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        
        # Extract text from PDF
        chapters = pdf_processor.extract_text_from_pdf(pdf_file)
        
        # Process each chapter
        for chapter_title, chapter_text in chapters.items():
            # Create age-appropriate summary
            processed_chapter = content_processor.create_age_appropriate_summary(
                chapter_text, chapter_title
            )
            all_chapters[chapter_title] = processed_chapter
            
            # Generate questions
            questions = question_generator.generate_questions_from_chapter(
                chapter_text, chapter_title
            )
            all_questions[chapter_title] = questions
            
            print(f"  - Processed: {chapter_title}")
    
    # Generate training data
    training_data = data_generator.create_training_dataset(all_chapters, all_questions)
    
    # Save all outputs
    with open(output_path / "processed_chapters.json", 'w', encoding='utf-8') as f:
        json.dump(all_chapters, f, indent=2, ensure_ascii=False)
    
    with open(output_path / "generated_questions.json", 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    
    data_generator.save_training_data(training_data, output_path / "nia_training_data.json")
    
    # Create summary report
    create_processing_report(all_chapters, all_questions, output_path)
    
    print(f"\n✅ Processing complete! Files saved to {output_path}")
    return output_path

def create_processing_report(chapters, questions, output_path):
    """
    Create a summary report of processed content
    """
    report = {
        "processing_summary": {
            "total_chapters": len(chapters),
            "total_questions": sum(
                len(q.get('multiple_choice', [])) + 
                len(q.get('fill_in_blank', [])) + 
                len(q.get('true_false', [])) + 
                len(q.get('short_answer', []))
                for q in questions.values()
            ),
            "chapters_processed": list(chapters.keys())
        },
        "content_breakdown": {}
    }
    
    for chapter_title, chapter_data in chapters.items():
        chapter_questions = questions.get(chapter_title, {}
        report["content_breakdown"][chapter_title] = {
            "summary_length": len(chapter_data.get('summary', '')),
            "key_points": len(chapter_data.get('key_points', [])),
            "multiple_choice_questions": len(chapter_questions.get('multiple_choice', [])),
            "fill_in_blank_questions": len(chapter_questions.get('fill_in_blank', [])),
            "true_false_questions": len(chapter_questions.get('true_false', [])),
            "short_answer_questions": len(chapter_questions.get('short_answer', []))
        }
    
    with open(output_path / "processing_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Nia's school books for educational assistant")
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", default="./data/nia_education", help="Output directory")
    
    args = parser.parse_args()
    process_nia_school_books(args.pdf_dir, args.output_dir)
```

### 6. Usage Instructions for Nia's Educational Assistant

```bash
# Install additional dependencies for PDF processing
pip install PyPDF2 PyMuPDF transformers torch

# Process Nia's school book PDFs
python education/process_nia_books.py --pdf_dir "./pdfs/nia_books" --output_dir "./data/nia_education"

# Train the educational assistant
python training/train_educational_model.py --data_file "./data/nia_education/nia_training_data.json"

# Start the educational assistant
python main.py --config configs/nia_education_config.yaml
```

### 7. Special Features for Nia

- **Age-appropriate language**: All content simplified for 9-year-old comprehension
- **Encouraging feedback**: Positive reinforcement and growth mindset messaging
- **Multiple question types**: Variety to keep learning engaging
- **Progress tracking**: Monitor which chapters have been covered
- **Interactive learning**: Voice and text interaction for different learning styles
- **Patient explanations**: Multiple ways to explain difficult concepts

### Next Steps

1. **Provide PDF files**: Place Nia's school book PDFs in a folder
2. **Run processing**: Execute the processing pipeline
3. **Review generated content**: Check summaries and questions for accuracy
4. **Train the model**: Fine-tune with the generated educational data
5. **Test with Nia**: Have her interact with the assistant and gather feedback
6. **Iterate**: Refine based on what works best for her learning style

This educational assistant will be completely private and tailored specifically to Nia's curriculum, helping her learn at her own pace with encouraging, age-appropriate support!
