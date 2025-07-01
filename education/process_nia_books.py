#!/usr/bin/env python3
"""
Complete pipeline to process Nia's school books and create educational training data
"""

import argparse
import json
import logging
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

# PDF processing
try:
    import PyPDF2
    import fitz  # PyMuPDF
except ImportError:
    print("Please install PDF processing libraries: pip install PyPDF2 PyMuPDF")
    exit(1)

# NLP processing
try:
    from transformers import pipeline
except ImportError:
    print("Please install transformers: pip install transformers torch")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchoolBookProcessor:
    """Process PDF school books and extract structured content"""
    
    def __init__(self):
        self.chapter_patterns = [
            r"Chapter\s+(\d+)",
            r"Unit\s+(\d+)", 
            r"Lesson\s+(\d+)",
            r"Section\s+(\d+)"
        ]
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, str]:
        """Extract text from PDF with chapter detection"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            chapters = {}
            current_chapter = f"Introduction - {pdf_path.stem}"
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
                    
                    current_chapter = f"{chapter_match} - {pdf_path.stem}"
                    current_text = []
                
                # Clean and add text
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    current_text.append(cleaned_text)
            
            # Save last chapter
            if current_text:
                chapters[current_chapter] = "\n".join(current_text)
            
            doc.close()
            logger.info(f"Extracted {len(chapters)} chapters from {pdf_path}")
            return chapters
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {}
    
    def detect_chapter(self, text: str) -> str:
        """Detect chapter headers in text"""
        for pattern in self.chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"Chapter {match.group(1)}"
        return None
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text for better processing"""
        if not text or len(text.strip()) < 10:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        filtered_lines = [line.strip() for line in lines if len(line.strip()) > 15]
        
        return '\n'.join(filtered_lines)


class EducationalContentProcessor:
    """Process content to be age-appropriate for 9-year-olds"""
    
    def __init__(self):
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            logger.warning(f"Could not load summarizer: {e}")
            self.summarizer = None
        
        self.age_level = "9-year-old (Grade 4)"
        
        # Word replacements for simplification
        self.word_simplifications = {
            "utilize": "use", "demonstrate": "show", "comprehend": "understand",
            "substantial": "big", "facilitate": "help", "subsequent": "next",
            "approximate": "about", "fundamental": "basic", "significant": "important",
            "consequently": "so", "therefore": "so", "however": "but",
            "nevertheless": "but", "furthermore": "also", "moreover": "also",
            "participate": "take part", "investigate": "look into", "observe": "watch",
            "examine": "look at", "analyze": "study", "conclude": "decide"
        }
    
    def create_age_appropriate_summary(self, chapter_text: str, chapter_title: str) -> Dict:
        """Create summaries appropriate for a 9-year-old"""
        # Split text into manageable chunks
        chunks = self.split_text(chapter_text, max_length=1000)
        
        if self.summarizer and chunks:
            summaries = []
            for chunk in chunks[:3]:  # Limit to first 3 chunks to avoid overwhelming
                try:
                    summary = self.summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    logger.warning(f"Summarization failed for chunk: {e}")
                    # Fallback to first few sentences
                    sentences = chunk.split('.')[:3]
                    summaries.append('. '.join(sentences) + '.')
            
            combined_summary = " ".join(summaries)
        else:
            # Fallback: use first few sentences
            sentences = chapter_text.split('.')[:5]
            combined_summary = '. '.join(sentences) + '.'
        
        # Simplify language
        simplified_summary = self.simplify_for_age(combined_summary)
        
        return {
            "chapter": chapter_title,
            "summary": simplified_summary,
            "key_points": self.extract_key_points(simplified_summary),
            "difficulty_level": self.age_level,
            "word_count": len(simplified_summary.split()),
            "key_concepts": self.extract_concepts(chapter_text)
        }
    
    def simplify_for_age(self, text: str) -> str:
        """Simplify text for 9-year-old comprehension"""
        simplified = text
        
        # Replace complex words
        for complex_word, simple_word in self.word_simplifications.items():
            simplified = re.sub(r'\b' + complex_word + r'\b', simple_word, simplified, flags=re.IGNORECASE)
        
        # Break long sentences
        sentences = simplified.split('.')
        new_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) > 15:  # If sentence is too long
                # Try to break at natural points
                if ',' in sentence:
                    parts = sentence.split(',')
                    for i, part in enumerate(parts):
                        if i == 0:
                            new_sentences.append(part.strip() + '.')
                        else:
                            new_sentences.append(part.strip() + '.')
                else:
                    new_sentences.append(sentence + '.')
            elif sentence:
                new_sentences.append(sentence + '.')
        
        return ' '.join(new_sentences)
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract 3-5 key learning points"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Score sentences based on important keywords
        important_keywords = ['important', 'key', 'main', 'because', 'remember', 'always', 'never', 'helps', 'makes']
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in important_keywords if keyword in sentence.lower())
            if score > 0 or len(sentence.split()) > 5:  # Either has keywords or is substantial
                scored_sentences.append((sentence, score))
        
        # Sort by score and return top 5
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in scored_sentences[:5]]
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        words = text.split()
        concepts = []
        
        # Find capitalized words (likely important terms)
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if (clean_word and clean_word[0].isupper() and 
                len(clean_word) > 3 and clean_word.isalpha() and
                clean_word.lower() not in ['the', 'this', 'that', 'when', 'where', 'what', 'how']):
                concepts.append(clean_word)
        
        # Count frequency and return most common
        concept_counts = {}
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Filter out very common words and return top concepts
        filtered_concepts = {k: v for k, v in concept_counts.items() if v >= 2}
        sorted_concepts = sorted(filtered_concepts.items(), key=lambda x: x[1], reverse=True)
        
        return [concept for concept, count in sorted_concepts[:8]]
    
    def split_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class EducationalQuestionGenerator:
    """Generate various types of educational questions"""
    
    def __init__(self):
        self.question_types = ['multiple_choice', 'fill_in_blank', 'true_false', 'short_answer']
    
    def generate_questions_from_chapter(self, chapter_content: str, chapter_title: str, 
                                       processed_data: Dict) -> Dict:
        """Generate various types of questions from chapter content"""
        questions = {
            "chapter": chapter_title,
            "multiple_choice": [],
            "fill_in_blank": [],
            "true_false": [],
            "short_answer": []
        }
        
        key_concepts = processed_data.get('key_concepts', [])
        summary = processed_data.get('summary', '')
        key_points = processed_data.get('key_points', [])
        
        # Generate different types of questions
        if key_concepts:
            questions["multiple_choice"] = self.create_multiple_choice(key_concepts, summary)[:3]
            questions["short_answer"] = self.create_short_answer(key_concepts)[:2]
        
        if summary:
            questions["fill_in_blank"] = self.create_fill_in_blank(summary)[:3]
        
        if key_points:
            questions["true_false"] = self.create_true_false(key_points)[:4]
        
        return questions
    
    def create_multiple_choice(self, concepts: List[str], summary: str) -> List[Dict]:
        """Create multiple choice questions"""
        questions = []
        
        for concept in concepts[:3]:
            # Find sentences mentioning this concept
            sentences = summary.split('.')
            concept_sentence = None
            
            for sentence in sentences:
                if concept.lower() in sentence.lower():
                    concept_sentence = sentence.strip()
                    break
            
            if concept_sentence:
                question_text = f"What is {concept}?"
                correct_answer = concept_sentence
                
                # Generate plausible wrong answers
                wrong_answers = [
                    f"{concept} is a type of animal.",
                    f"{concept} is something you eat.",
                    f"{concept} is a place you visit."
                ]
                
                # Try to make more relevant wrong answers
                other_concepts = [c for c in concepts if c != concept]
                if other_concepts:
                    wrong_answers = [
                        f"{concept} is the same as {other_concepts[0]}." if len(other_concepts) > 0 else wrong_answers[0],
                        f"{concept} is used only by {other_concepts[1]}." if len(other_concepts) > 1 else wrong_answers[1],
                        f"{concept} happens before {other_concepts[2]}." if len(other_concepts) > 2 else wrong_answers[2]
                    ]
                
                choices = [correct_answer] + wrong_answers[:3]
                random.shuffle(choices)
                
                questions.append({
                    "question": question_text,
                    "choices": choices,
                    "correct_answer": correct_answer,
                    "explanation": f"The correct answer tells us what {concept} really means."
                })
        
        return questions
    
    def create_fill_in_blank(self, text: str) -> List[Dict]:
        """Create fill-in-the-blank questions"""
        questions = []
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 6]
        
        for sentence in sentences[:3]:
            words = sentence.split()
            
            # Find a good word to blank out (noun or important word)
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w]', '', word)
                if (len(clean_word) > 4 and clean_word.isalpha() and 
                    clean_word.lower() not in ['this', 'that', 'they', 'there', 'where', 'when']):
                    
                    blank_sentence = " ".join(words[:i] + ["____"] + words[i+1:])
                    questions.append({
                        "question": f"Fill in the blank: {blank_sentence}",
                        "answer": clean_word,
                        "hint": f"This word starts with '{clean_word[0]}' and has {len(clean_word)} letters.",
                        "sentence": sentence
                    })
                    break
        
        return questions
    
    def create_true_false(self, key_points: List[str]) -> List[Dict]:
        """Create true/false questions"""
        questions = []
        
        for point in key_points[:2]:
            if point and len(point.strip()) > 10:
                # Create a true statement
                questions.append({
                    "statement": point,
                    "answer": True,
                    "explanation": "This statement is true based on what we learned in the chapter."
                })
                
                # Create a false statement by modifying the fact
                false_statement = self.create_false_statement(point)
                questions.append({
                    "statement": false_statement,
                    "answer": False,
                    "explanation": "This statement contains incorrect information."
                })
        
        return questions
    
    def create_short_answer(self, concepts: List[str]) -> List[Dict]:
        """Create short answer questions"""
        questions = []
        
        question_templates = [
            "What is {concept} and why is it important?",
            "How would you explain {concept} to a friend?",
            "Can you describe what {concept} means?",
            "Why should we learn about {concept}?"
        ]
        
        for concept in concepts[:2]:
            template = random.choice(question_templates)
            question = template.format(concept=concept)
            
            questions.append({
                "question": question,
                "sample_answer": f"{concept} is an important concept that helps us understand...",
                "keywords": [concept.lower()],
                "hint": f"Think about what {concept} does or why it matters."
            })
        
        return questions
    
    def create_false_statement(self, fact: str) -> str:
        """Create a false statement by modifying a true fact"""
        modifications = {
            " is ": " is not ",
            " are ": " are not ",
            " can ": " cannot ",
            " will ": " will not ",
            " always ": " never ",
            " helps ": " does not help ",
            " makes ": " does not make ",
            " hot ": " cold ",
            " big ": " small ",
            " fast ": " slow ",
            " good ": " bad "
        }
        
        modified = fact
        for original, replacement in modifications.items():
            if original in fact.lower():
                modified = fact.replace(original, replacement)
                break
        
        return modified if modified != fact else f"It is not true that {fact.lower()}"


class NiaEducationDataGenerator:
    """Generate training data specifically for Nia's educational assistant"""
    
    def __init__(self):
        self.assistant_name = "Nia's Learning Helper"
        self.personality_traits = [
            "friendly and encouraging",
            "patient and understanding", 
            "uses simple, clear language",
            "celebrates learning achievements",
            "makes learning fun"
        ]
    
    def create_training_dataset(self, processed_chapters: Dict, all_questions: Dict) -> List[Dict]:
        """Create comprehensive training data for Nia's educational assistant"""
        training_data = []
        
        # Create different types of conversations
        for chapter_title, chapter_data in processed_chapters.items():
            chapter_questions = all_questions.get(chapter_title, {})
            
            # Summary and explanation conversations
            training_data.extend(self.create_summary_conversations(chapter_title, chapter_data))
            
            # Question practice conversations
            training_data.extend(self.create_qa_conversations(chapter_title, chapter_questions))
            
            # Study help conversations
            training_data.extend(self.create_study_help_conversations(chapter_title, chapter_data))
            
            # Encouragement conversations
            training_data.extend(self.create_encouragement_conversations())
        
        # Add general helpful conversations
        training_data.extend(self.create_general_conversations())
        
        return training_data
    
    def create_summary_conversations(self, chapter: str, chapter_data: Dict) -> List[Dict]:
        """Create conversations about chapter summaries"""
        summary = chapter_data.get('summary', '')
        key_points = chapter_data.get('key_points', [])
        
        conversations = [
            {
                "instruction": "You are Nia's helpful learning assistant. Always explain things in simple, encouraging words for a 9-year-old.",
                "input": f"Can you tell me about {chapter}?",
                "output": f"Hi Nia! I'd love to tell you about {chapter}! {summary} The most important things to remember are: {'. '.join(key_points[:3])}. You're doing such a great job learning! Is there anything you'd like me to explain more?"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Always explain things in simple, encouraging words for a 9-year-old.",
                "input": f"What are the main points of {chapter}?",
                "output": f"Great question, Nia! The main points of {chapter} are: {'. '.join(key_points[:4])}. These are really important ideas to remember. You're so smart for asking about the main points!"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Always explain things in simple, encouraging words for a 9-year-old.",
                "input": f"I need help understanding {chapter}",
                "output": f"Of course, Nia! I'm here to help you understand {chapter}. {summary} Don't worry if it seems tricky at first - that's completely normal! Learning new things takes time, and you're doing wonderfully. What part would you like me to explain again?"
            }
        ]
        
        return conversations
    
    def create_qa_conversations(self, chapter: str, chapter_questions: Dict) -> List[Dict]:
        """Create question-answering practice conversations"""
        conversations = []
        
        # Multiple choice practice
        for mc_q in chapter_questions.get('multiple_choice', [])[:2]:
            conversations.append({
                "instruction": "You are Nia's helpful learning assistant. Make questions fun and provide encouraging feedback.",
                "input": f"Can you ask me a question about {chapter}?",
                "output": f"Sure, Nia! Here's a fun question for you: {mc_q['question']} Your choices are: A) {mc_q['choices'][0]}, B) {mc_q['choices'][1]}, C) {mc_q['choices'][2]}, D) {mc_q['choices'][3]}. Take your time and think carefully!"
            })
        
        # Fill in the blank practice
        for fib_q in chapter_questions.get('fill_in_blank', [])[:2]:
            conversations.append({
                "instruction": "You are Nia's helpful learning assistant. Give helpful hints and encourage thinking.",
                "input": "Can you give me a fill-in-the-blank question?",
                "output": f"Here's a fill-in-the-blank question for you, Nia: {fib_q['question']} {fib_q.get('hint', '')} You can do this!"
            })
        
        return conversations
    
    def create_study_help_conversations(self, chapter: str, chapter_data: Dict) -> List[Dict]:
        """Create study assistance conversations"""
        concepts = chapter_data.get('key_concepts', [])
        
        conversations = [
            {
                "instruction": "You are Nia's helpful learning assistant. Help with studying in a fun, encouraging way.",
                "input": f"How should I study {chapter}?",
                "output": f"Great question, Nia! Here are some fun ways to study {chapter}: 1) Read the summary out loud, 2) Draw pictures of the main ideas, 3) Ask yourself questions about what you learned, 4) Explain it to a stuffed animal or family member. The key concepts to focus on are: {', '.join(concepts[:4])}. You're going to do amazing!"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Break down complex topics into simple steps.",
                "input": f"This chapter seems really hard",
                "output": f"I understand, Nia! Sometimes new topics can feel challenging, but that just means your brain is growing stronger! Let's break {chapter} down into smaller, easier pieces. Think of it like building with blocks - we'll start with one simple idea and add more. You're braver than you think and smarter than you know!"
            }
        ]
        
        return conversations
    
    def create_encouragement_conversations(self) -> List[Dict]:
        """Create encouraging and supportive conversations"""
        conversations = [
            {
                "instruction": "You are Nia's helpful learning assistant. Always be positive, encouraging, and supportive.",
                "input": "I got the answer wrong",
                "output": "That's totally okay, Nia! Making mistakes is actually how our brains learn best - it's like exercise for your mind! Every mistake teaches us something new. What matters most is that you tried, and I'm so proud of you for that. Let's figure out the right answer together, and then you'll know it forever!"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Always be positive, encouraging, and supportive.",
                "input": "I don't think I'm smart enough",
                "output": "Oh Nia, you ARE smart! Being smart isn't about knowing everything right away - it's about being curious, asking questions, and never giving up. You ask wonderful questions and you keep trying even when things are hard. That makes you very smart indeed! Remember, every expert was once a beginner, and every master was once a disaster. You're doing fantastic!"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Always be positive, encouraging, and supportive.",
                "input": "This is boring",
                "output": "I hear you, Nia! Sometimes learning can feel boring, but let's make it more fun! We could turn it into a game, make up silly stories about what we're learning, or find connections to things you love. What do you enjoy doing? Maybe we can connect your lesson to that! Learning is always more exciting when we make it our own adventure."
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Always be positive, encouraging, and supportive.", 
                "input": "I'm ready for my test",
                "output": "Wow, Nia! I'm so excited and proud of you! You've worked really hard and studied well. Remember these important things for your test: take deep breaths, read each question carefully, trust yourself (you know more than you think!), and don't rush. If you get stuck on a question, skip it and come back later. You've got this, superstar!"
            }
        ]
        
        return conversations
    
    def create_general_conversations(self) -> List[Dict]:
        """Create general helpful conversations"""
        conversations = [
            {
                "instruction": "You are Nia's helpful learning assistant. Introduce yourself warmly and explain how you can help.",
                "input": "Hello, who are you?",
                "output": "Hi Nia! I'm your very own learning helper, and I'm so happy to meet you! I'm here to help you with your schoolwork, answer your questions, and make learning fun. We can review your chapters together, practice with questions, or I can explain things in different ways until they make sense. I love helping you learn and grow! What would you like to work on today?"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Explain your purpose and capabilities in kid-friendly terms.",
                "input": "What can you help me with?",
                "output": "I can help you with so many things, Nia! I can explain your school lessons in simple ways, ask you practice questions, help you study for tests, and even make learning games! If something in your book is confusing, I can explain it differently until it clicks. I can also cheer you on when you're doing great and help you feel better when things are tough. Think of me as your personal learning buddy who's always here to help!"
            },
            {
                "instruction": "You are Nia's helpful learning assistant. Be encouraging about the learning process.",
                "input": "How do I become better at learning?",
                "output": "What a wonderful question, Nia! Here are some super tips to become an amazing learner: 1) Ask lots of questions (there are no silly questions!), 2) Practice a little bit every day, 3) Don't be afraid to make mistakes - they help you learn!, 4) Connect new things to stuff you already know, 5) Take breaks when your brain feels tired, and 6) Celebrate your progress, even small wins! Remember, your brain is like a muscle - the more you use it, the stronger it gets!"
            }
        ]
        
        return conversations


def process_nia_school_books(pdf_directory: str, output_directory: str):
    """Main processing pipeline for Nia's school books"""
    pdf_dir = Path(pdf_directory)
    output_dir = Path(output_directory)
    
    # Validate input directory
    if not pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {pdf_directory}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize all processors
    pdf_processor = SchoolBookProcessor()
    content_processor = EducationalContentProcessor()
    question_generator = EducationalQuestionGenerator()
    data_generator = NiaEducationDataGenerator()
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_directory}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process all PDFs
    all_chapters = {}
    all_questions = {}
    processing_stats = {"total_chapters": 0, "total_questions": 0, "processed_files": []}
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        
        try:
            # Extract text from PDF
            chapters = pdf_processor.extract_text_from_pdf(pdf_file)
            
            if not chapters:
                logger.warning(f"No content extracted from {pdf_file.name}")
                continue
            
            processing_stats["processed_files"].append(pdf_file.name)
            
            # Process each chapter
            for chapter_title, chapter_text in chapters.items():
                if len(chapter_text.strip()) < 50:  # Skip very short chapters
                    logger.warning(f"Skipping short chapter: {chapter_title}")
                    continue
                
                # Create age-appropriate summary
                processed_chapter = content_processor.create_age_appropriate_summary(
                    chapter_text, chapter_title
                )
                all_chapters[chapter_title] = processed_chapter
                
                # Generate questions
                questions = question_generator.generate_questions_from_chapter(
                    chapter_text, chapter_title, processed_chapter
                )
                all_questions[chapter_title] = questions
                
                # Update stats
                processing_stats["total_chapters"] += 1
                processing_stats["total_questions"] += sum(
                    len(q) for q in questions.values() if isinstance(q, list)
                )
                
                logger.info(f"  ✓ Processed: {chapter_title}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            continue
    
    if not all_chapters:
        logger.error("No chapters were successfully processed")
        return
    
    # Generate training data
    logger.info("Generating training data...")
    training_data = data_generator.create_training_dataset(all_chapters, all_questions)
    
    # Save all outputs
    logger.info("Saving processed data...")
    
    # Save processed chapters
    with open(output_dir / "processed_chapters.json", 'w', encoding='utf-8') as f:
        json.dump(all_chapters, f, indent=2, ensure_ascii=False)
    
    # Save generated questions
    with open(output_dir / "generated_questions.json", 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    
    # Save training data
    with open(output_dir / "nia_training_data.json", 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Create processing report
    processing_stats.update({
        "training_examples": len(training_data),
        "output_directory": str(output_dir),
        "chapter_titles": list(all_chapters.keys())
    })
    
    with open(output_dir / "processing_report.json", 'w', encoding='utf-8') as f:
        json.dump(processing_stats, f, indent=2, ensure_ascii=False)
    
    # Create a simple summary for Nia's parents
    create_parent_summary(processing_stats, all_chapters, output_dir)
    
    logger.info(f"✅ Processing complete!")
    logger.info(f"📁 Output saved to: {output_dir}")
    logger.info(f"📚 Processed {processing_stats['total_chapters']} chapters")
    logger.info(f"❓ Generated {processing_stats['total_questions']} questions") 
    logger.info(f"💬 Created {len(training_data)} training examples")
    
    return output_dir


def create_parent_summary(stats: Dict, chapters: Dict, output_dir: Path):
    """Create a simple summary report for parents"""
    summary = f"""
# Nia's Learning Assistant - Processing Summary

## What was processed:
- **Files**: {', '.join(stats['processed_files'])}
- **Chapters**: {stats['total_chapters']} chapters
- **Questions**: {stats['total_questions']} practice questions
- **Training examples**: {stats['training_examples']} conversations

## Chapters covered:
{chr(10).join('- ' + title for title in stats['chapter_titles'])}

## What Nia can now do:
- Ask questions about any chapter
- Get simple explanations of difficult concepts
- Practice with multiple choice, fill-in-the-blank, and other question types
- Get encouragement and study tips
- Review key concepts before tests

## How to use:
1. Start the assistant: `python main.py`
2. Open browser to: http://localhost:8000
3. Nia can ask: "Can you tell me about Chapter 1?" or "Can you ask me a question about plants?"

The assistant is trained to be encouraging, patient, and use age-appropriate language for Nia's level.
"""
    
    with open(output_dir / "parent_summary.md", 'w', encoding='utf-8') as f:
        f.write(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Process Nia's school books to create educational training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_nia_books.py --pdf_dir "./pdfs" --output_dir "./data/nia"
  python process_nia_books.py --pdf_dir "C:/Users/Admin/Documents/Nia_Books" --output_dir "./education_data"
        """
    )
    
    parser.add_argument(
        "--pdf_dir", 
        required=True,
        help="Directory containing Nia's school book PDF files"
    )
    parser.add_argument(
        "--output_dir", 
        default="./data/nia_education",
        help="Output directory for processed data (default: ./data/nia_education)"
    )
    
    args = parser.parse_args()
    
    try:
        process_nia_school_books(args.pdf_dir, args.output_dir)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
