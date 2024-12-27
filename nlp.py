import gradio as gr
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AlbertTokenizer, AlbertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import textract
import os
import numpy as np
import pandas as pd
# Model configurations
MODEL_CONFIGS = [
    {
        'name': 'bert-base-uncased',
        'tokenizer_class': BertTokenizer,
        'model_class': BertModel,
        'weight': 0.5  # Configurable weight for each model
    },
    {
        'name': 'roberta-base',
        'tokenizer_class': RobertaTokenizer,
        'model_class': RobertaModel,
        'weight': 0.5
    },
   
]
# Initialize models and tokenizers
def initialize_models():
    models = {}
    for config in MODEL_CONFIGS:
        name = config['name']
        tokenizer = config['tokenizer_class'].from_pretrained(name)
        model = config['model_class'].from_pretrained(name)
        models[name] = {
            'tokenizer': tokenizer,
            'model': model
        }
    return models

import torch
import numpy as np

def generate_embeddings(text, model_info):
    tokenizer = model_info['tokenizer']
    model = model_info['model']

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Get the outputs, including attention weights
        outputs = model(**inputs, output_attentions=True)

    # Extract the token embeddings and attention weights
    token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_dim)
    attention_weights = outputs.attentions[-1]   # Last layer attention, shape: (batch_size, num_heads, seq_length, seq_length)

    # Average the attention weights across all heads
    avg_attention = attention_weights.mean(dim=1)  # Shape: (batch_size, seq_length, seq_length)

    # Weight the token embeddings with the averaged attention
    weighted_embeddings = torch.matmul(avg_attention, token_embeddings)  # Shape: (batch_size, seq_length, hidden_dim)

    # Aggregate embeddings (e.g., mean pooling over the sequence length dimension)
    sentence_embedding = weighted_embeddings.mean(dim=1).squeeze(0).numpy()  # Shape: (hidden_dim,)

    return sentence_embedding

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Function to preprocess resume text into different sections
def preprocess_resume(text):
    try:
        # Initialize sections with expanded keywords
        sections = {
            'work_experience': '', 
            'education': '', 
            'skills': '',
            'contact_info': '',
            'summary': '',
            'certifications': '',
            'projects': '',
            'languages': ''
        }

        # Expanded keywords for section detection
        section_keywords = {
            'work_experience': [
                'work experience', 'professional experience', 'employment history', 
                'career', 'professional background', 'work history', 'experience'
            ],
            'education': [
                'education', 'academic background', 'educational qualifications', 
                'academic history', 'academic details', 'degrees'
            ],
            'skills': [
                'skills', 'technical skills', 'skills summary', 'core competencies', 
                'professional skills', 'key skills'
            ],
            'contact_info': [
                'contact information', 'contact details', 'personal information', 
                'contact', 'personal details'
            ],
            'summary': [
                'professional summary', 'profile', 'career objective', 
                'professional profile', 'career summary'
            ],
            'certifications': [
                'certifications', 'professional certifications', 'certificates', 
                'professional credentials'
            ],
            'projects': [
                'projects', 'project experience', 'professional projects', 
                'key projects'
            ],
            'languages': [
                'languages', 'language skills', 'spoken languages', 'language proficiency'
            ]
        }

        # Load stop words
        stop_words = set(stopwords.words('english'))

        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()

        # Tokenize text and remove stop words
        words = word_tokenize(text_lower)
        filtered_text = ' '.join(word for word in words if word not in stop_words)
        
        # Find starting indices for each section
        section_indices = {}
        for section, keywords in section_keywords.items():
            # Find the minimum index of any matching keyword
            indices = [filtered_text.find(keyword) for keyword in keywords if keyword in filtered_text]
            section_indices[section] = min(indices) if indices else -1

        # Sort the indices to help with section extraction
        sorted_sections = sorted(
            [(section, index) for section, index in section_indices.items() if index != -1], 
            key=lambda x: x[1]
        )

        # Extract sections based on their positions
        for i, (section, start_index) in enumerate(sorted_sections):
            # Determine the end index by looking at the start of the next section
            end_index = sorted_sections[i+1][1] if i+1 < len(sorted_sections) else len(filtered_text)

            # Extract the section content
            sections[section] = filtered_text[start_index:end_index].strip()
        print(sections)
        
        return sections

    except Exception as e:
        print(f"Error preprocessing resume text: {e}")
        return {
            'work_experience': '', 
            'education': '', 
            'skills': '',
            'contact_info': '',
            'summary': '',
            'certifications': '',
            'projects': '',
            'languages': ''
        }

import PyPDF2

# Function to extract text from a PDF

    # Open the uploaded PDF
    
# Function to extract text from a local file
def extract_text_from_file(file_path):
    print(file_path)
    try:
        
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Loop through all pages and extract text
            for page in reader.pages:
                text += page.extract_text()
        
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""
def search_resumes(job_description, resumes):
    """
    Search and rank resumes against a job description.
    
    Args:
        job_description (str): Job description to match against resumes
        resumes (dict): Dictionary of resume texts with filenames as keys
    
    Returns:
        tuple: Formatted output string and results DataFrame
    """
    # Initialize models
    models_dict = initialize_models()
    
    # Store results for each model
    all_results = []
    
    try:
        # Generate query embeddings for each model
        query_embeddings = {}
        for config in MODEL_CONFIGS:
            model_name = config['name']
            query_embeddings[model_name] = generate_embeddings(job_description, models_dict[model_name])
        
        # Process each resume
        for file_name, text in resumes.items():
            # Preprocess resume
            preprocessed_resume = preprocess_resume(text)
            
            # Create a result dictionary for this resume
            result = {
                'file_name': file_name, 
                'resume_content': preprocessed_resume,
                'job_description': job_description  # Add job description to each result
            }
            
            # Calculate similarity for each model
            for config in MODEL_CONFIGS:
                model_name = config['name']
                model_info = models_dict[model_name]
                
                # Generate resume embeddings with more sections
                embedding_sections = [
                    preprocessed_resume.get('work_experience', ''),
                    preprocessed_resume.get('education', ''),
                    preprocessed_resume.get('skills', ''),
                    preprocessed_resume.get('summary', ''),
                    preprocessed_resume.get('projects', ''),
                    preprocessed_resume.get('certifications', ''),
                ]
                
                # Combine embeddings
                resume_embeddings = [
                    generate_embeddings(section, model_info)
                    for section in embedding_sections
                    if section.strip()
                ]
                
                # Average embeddings if any exist
                if resume_embeddings:
                    resume_embedding = np.mean(resume_embeddings, axis=0)
                else:
                    # Fallback if no embeddings generated
                    resume_embedding = generate_embeddings('', model_info)
                
                # Calculate cosine similarity
                similarity = cosine_similarity([query_embeddings[model_name]], [resume_embedding])[0][0]
                
                # Store model-specific similarity
                result[f'{model_name}_similarity'] = similarity
            
            # Calculate weighted combined similarity
            combined_similarity = sum(
                result[f'{config["name"]}_similarity'] * config['weight']
                for config in MODEL_CONFIGS
            )
            
            # Normalize to 0-1 range
            result['combined_similarity'] = max(0, min(1, combined_similarity))
            
            # Append the result for the resume
            all_results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Sort the DataFrame by combined similarity
        results_df = results_df.sort_values('combined_similarity', ascending=False)
        
        # Generate text output (without job description)
        output = "Search Results:\n\n"
        for _, result in results_df.iterrows():
            output += f"File: {result['file_name']}\n"
            for config in MODEL_CONFIGS:
                model_name = config['name']
                output += f"{model_name} Similarity: {result[f'{model_name}_similarity']:.4f}\n"
            output += f"Combined Similarity: {result['combined_similarity']:.4f}\n\n"
        
        global search_results
        search_results = results_df
        return output, results_df
    
    except Exception as e:
        return f"Error occurred: {e}", None

def process_files(query, files):
    # Convert uploaded files to resumes dictionary
    resumes = {}
    for file in files:
        # Extract text from the file
        text = extract_text_from_file(file.name)
        resumes[os.path.basename(file.name)] = text
    
    # Call search_resumes with query and resumes
    output, results_df = search_resumes(query, resumes)
    
    return output

def parse_resumes(files):
    # Convert uploaded files to resumes dictionary
    resumes = {}
    for file in files:
        # Extract text from the file
        text = extract_text_from_file(file.name)
        resumes[os.path.basename(file.name)] = text
    
    return resumes

import together
import pandas as pd

def generate_ideal_resume(job_description):
    """
    Generate an ideal resume for a given job description using Together AI.
    
    Args:
        job_description (str): Job description to base the ideal resume on
    
    Returns:
        str: Ideal resume text
    """
    client = together.Together(api_key='804be1d7c2ed8aae76e383e83f5b28c1f2101d7fb4e31d15bf566f43029c44a1')
    
    # Prompt to generate an ideal resume
    prompt = f"""You are an expert resume writer. Create an ideal resume for the following job description:

{job_description}

Please create a comprehensive resume that highlights:
- Relevant work experience
- Key skills matching the job requirements
- Professional summary
- Notable achievements
- Technical and soft skills

Format the resume professionally, focusing on the most critical aspects for this role."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are an expert resume writer creating an ideal resume."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.0,
        stop=["<|eot_id|>","<|eom_id|>"]
    )
    
    return response.choices[0].message.content

def evaluate_resume(ideal_resume, candidate_resume, job_description, filename, similarities):
    ideal_resume=preprocess_resume(ideal_resume)
    """
    Evaluate a candidate's resume against an ideal resume using Together AI.
    
    Args:
        ideal_resume (str): Ideal resume text
        candidate_resume (str): Candidate's resume text
        job_description (str): Job description
        filename (str): Name of the resume file
        similarities (dict): Semantic similarities for different models
    
    Returns:
        str: Evaluation summary
    """
    client = together.Together(api_key='804be1d7c2ed8aae76e383e83f5b28c1f2101d7fb4e31d15bf566f43029c44a1')
    
    # Prepare similarity information
    similarity_info = "\n".join([
        f"{model}: {similarity:.4f}" 
        for model, similarity in similarities.items()
    ])
    
    # Prompt for resume evaluation
    prompt = f"""You are an expert HR recruiter. Compare the following resumes:

Job Description:
{job_description}

Semantic Similarities:
{similarity_info}

Ideal Resume:
{ideal_resume}

Candidate Resume:
{candidate_resume}

Provide a detailed evaluation following this format:
Name: {filename}
Score out of 10: [Numerical score based on job fit AND semantic similarities]
Summary: [Concise overview of candidate's strengths and weaknesses]
AI Suggestion: [Hire/Do Not Hire recommendation with specific reasoning]

Consider the provided semantic similarities as an additional factor in your evaluation. 
Focus primarily on experience and skills. Be objective and provide constructive insights.
Consider these weights
"work_experience": 0.5, "skills": 0.3, "education": 0.2, "summary": 0.1, "projects": 0.2, "certifications": 0.1
Do not care about formatting
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are an expert HR recruiter evaluating resumes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.0,
        stop=["<|eot_id|>","<|eom_id|>"]
    )
    
    return response.choices[0].message.content

def llm_processor(search_results):
    """
    Process and evaluate resumes from the search results.
    
    Args:
        search_results (pd.DataFrame): DataFrame with resume search results
    
    Returns:
        list: Evaluation summaries for each candidate
    """
    # Extract job description (assuming it's the same for all rows)
    job_description = search_results['job_description'].iloc[0]
    
    # Generate ideal resume
    ideal_resume = generate_ideal_resume(job_description)
    print("Ideal Resume Generated:")
    print(ideal_resume)
    print("\n--- Resume Evaluations ---\n")
    
    # Store evaluations
    evaluations = []
    
    # Evaluate each candidate resume
    for _, row in search_results.iterrows():
        # Extract resume content and similarities
        candidate_resume = str(row['resume_content'])
        filename = row['file_name']
        # Collect similarities
        similarities = {
            'bert-base-uncased': row['bert-base-uncased_similarity'],
            'roberta-base': row['roberta-base_similarity'],
            #'albert-base-v2': row['albert-base-v2_similarity'],
            'combined': row['combined_similarity']
        }
        
        # Evaluate resume
        evaluation = evaluate_resume(
            ideal_resume, 
            candidate_resume, 
            job_description, 
            filename, 
            similarities
        )
        #print(evaluation)
        print("\n--------------------------\n")
        
        evaluations.append(evaluation)
        print(candidate_resume)
    
    return evaluations


from g4f.client import Client

# Function to evaluate LLaMA's output using GPT-4
def evaluate_llama_output_with_gpt4(llama_evaluation, candidate_resume, job_description, filename):
    """
    Evaluate LLaMA's evaluation output using GPT-4 (via g4f.client).
    """
    # Prepare the prompt for GPT-4
    prompt = f"""
    You are an expert HR recruiter. Here is an evaluation of a candidate's resume generated by LLaMA:

    LLaMA Evaluation:
    {llama_evaluation}

    Job Description:
    {job_description}

    Candidate Resume:
    {candidate_resume}

    Please provide a critical analysis of the LLaMA evaluation.
    Focus on the accuracy, thoroughness, and relevance of LLaMA's output.
    Suggest areas of improvement and give a final recommendation (e.g., hire/do not hire)
    Do not care about formatting.
    """

    # Create a client instance to interact with GPT-4
    client = Client()

    # Call GPT-4 via the g4f.client API
    response = client.chat.completions.create(
        model="gpt-4o",  # Replace with the correct model identifier
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,
        top_p=0.7,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    
    return response.choices[0].message.content.strip()
import gradio as gr

# Updated Process Function
def process_resume(job_description, resume_file):
    # Process the resumes
    output = process_files(job_description, resume_file)
    
    # Get evaluations from the LLM processor
    evaluations = llm_processor(search_results)
    
    # Generate GPT-4 evaluations for each resume
    gpt4_evaluations = []
    for _, row in search_results.iterrows():
        llama_evaluation = f"""
        BERT Similarity: {row['bert-base-uncased_similarity']:.4f}
        RoBERTa Similarity: {row['roberta-base_similarity']:.4f}
        Combined Similarity: {row['combined_similarity']:.4f}
        """
        candidate_resume = row['resume_content']
        filename = row['file_name']
        
        # Call GPT-4 evaluation
        gpt4_evaluation = evaluate_llama_output_with_gpt4(
            llama_evaluation, 
            candidate_resume, 
            job_description, 
            filename
        )
        gpt4_evaluations.append(f"File: {filename}\n{gpt4_evaluation}")
    
    # Join the GPT-4 evaluations with two line breaks
    formatted_gpt4_output = "\n\n".join(gpt4_evaluations)
    
    # Clean up the LLM Processor Output to format text properly (replacing \n with actual line breaks)
    llm_output = evaluations[0].replace("\\n", "\n")  # Ensure newlines are rendered correctly
    
    return output, llm_output, formatted_gpt4_output

with gr.Blocks() as demo:
    # Add a title to the interface
    gr.Markdown(
        """
        # Resume Evaluation Tool
        Upload resumes and provide a job description to get similarity evaluations and detailed feedback.
        """
    )
    
    with gr.Row():
        with gr.Column():
            job_description_input = gr.Textbox(label="Job Description")
            resume_file_input = gr.File(
                file_types=[".pdf", ".doc", ".docx", ".txt"],
                file_count="multiple",
                label="Upload Resumes",
                interactive=True
            )
            process_files_output = gr.Textbox(label="Process Files Output", lines=10)
        with gr.Column():
            llm_output = gr.Textbox(label="LLM Processor Output", lines=10)
            gpt4_output = gr.Textbox(label="GPT-4 Evaluation Output", lines=10)
   
    process_button = gr.Button("Process")
    process_button.click(
        fn=process_resume,
        inputs=[job_description_input, resume_file_input],
        outputs=[process_files_output, llm_output, gpt4_output]
    )

demo.launch()



