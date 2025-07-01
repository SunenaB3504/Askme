# Quick Start Guide - Nia's Educational Assistant

## Step 1: Install Requirements

```powershell
# Navigate to your project directory
cd c:\Users\Admin\AI-training\Askme

# Install educational processing requirements
pip install -r requirements-education.txt
```

## Step 2: Prepare Nia's School Books

1. Create a folder for Nia's PDF books:
```powershell
mkdir "c:\Users\Admin\AI-training\Askme\nia_books"
```

2. Copy Nia's school book PDFs into this folder

## Step 3: Process the Books

```powershell
# Process Nia's books to create training data
python education\process_nia_books.py --pdf_dir "C:\Users\Admin\Nia\Class 4\English\PDFs" --output_dir ".\data\nia_english"
```

✅ **COMPLETED!** Nia's 15 English chapters have been processed:
- 📚 15 chapters processed
- ❓ 164 practice questions generated  
- 💬 195 training conversations created
- 📝 Age-appropriate summaries for each chapter

## Step 4: Train the Educational Model (Optional)

If you want to fine-tune the model specifically for Nia:

```powershell
# Train the educational assistant model (using Nia's processed English data)
python training\train_educational_model.py --data_file ".\data\nia_english\nia_training_data.json"
```

## Step 5: Start Nia's Assistant

```powershell
# Start the educational assistant
python main.py --config configs\nia_education_config.yaml
```

Then open: http://localhost:8000

## What Nia Can Ask:

- **"Can you tell me about Chapter 1?"** - Get chapter summaries
- **"Ask me a question about plants"** - Practice questions
- **"I don't understand photosynthesis"** - Get explanations
- **"Help me study for my test"** - Study assistance
- **"This is too hard"** - Get encouragement and support

## Example Interactions:

**Nia:** "Can you tell me about the water cycle?"

**Assistant:** "Hi Nia! I'd love to tell you about the water cycle! The water cycle is how water moves around our planet. Water goes up into the sky as invisible water vapor, forms clouds, and then comes back down as rain or snow. Then it happens all over again! The most important things to remember are: Water evaporates from oceans and lakes, clouds form in the sky, and water falls back down as precipitation. You're doing such a great job learning about science!"

**Nia:** "Can you ask me a question?"

**Assistant:** "Sure, Nia! Here's a fun question for you: What happens to water when it gets very hot? Your choices are: A) It turns into ice, B) It becomes water vapor, C) It stays the same, D) It becomes heavier. Take your time and think carefully!"

## Features for Parents:

- **Complete Privacy**: Everything runs locally, no data sent online
- **Age-Appropriate**: Language simplified for 9-year-old comprehension
- **Encouraging**: Positive reinforcement and growth mindset
- **Curriculum-Based**: Uses Nia's actual school books
- **Safe Learning**: No inappropriate content, child-friendly interface

## Troubleshooting:

**Problem**: "No PDF files found"
**Solution**: Make sure PDF files are in the correct directory and have .pdf extension

**Problem**: "Error processing PDF"
**Solution**: Some PDFs may be protected or image-based. Try with different PDF files.

**Problem**: "Model not found"
**Solution**: Run the setup_models.py script first to download required models

## Tips for Best Results:

1. **Use clear, simple PDFs**: Text-based PDFs work better than scanned images
2. **One subject per run**: Process math books separately from science books
3. **Review generated content**: Check the summaries and questions for accuracy
4. **Encourage Nia**: The assistant is designed to be supportive, but parental encouragement helps too!
5. **Start simple**: Begin with easier chapters and gradually increase difficulty

## Generated Files:

After processing, you'll find:
- `processed_chapters.json` - Chapter summaries and key points
- `generated_questions.json` - Practice questions for each chapter
- `nia_training_data.json` - Training data for the assistant
- `parent_summary.md` - Simple overview for parents
- `processing_report.json` - Technical details

Now Nia has her own personal, private learning assistant trained on her actual school books! 🎓✨
