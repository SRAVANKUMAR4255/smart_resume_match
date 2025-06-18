from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Resume, JobDescription
import PyPDF2
import docx
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import traceback
import json

# Set up logging
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logger.error("Error loading spaCy model. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def extract_text_from_pdf(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.error(f"Error extracting text from PDF page: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Error reading PDF file: {str(e)}")
                return ""
        return text.strip()
    except Exception as e:
        logger.error(f"Error opening PDF file {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text.strip() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_file(file_path):
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return ""
            
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return ""
            
        if not text:
            logger.error(f"No text extracted from file: {file_path}")
            return ""
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from file {file_path}: {str(e)}")
        return ""

def extract_education(text):
    education = []
    # Expanded education keywords and patterns
    education_patterns = [
        r'(?i)(bachelor|master|phd|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?sc|m\.?sc|bca|mca|b\.?com|m\.?com|b\.?ba|m\.?ba)',
        r'(?i)(computer science|information technology|software engineering|data science|artificial intelligence|machine learning)',
        r'(?i)(university|college|institute|school)',
        r'(?i)(degree|diploma|certification)'
    ]
    
    # Split text into lines and process each line
    lines = text.split('\n')
    current_education = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line contains education-related information
        is_education_line = any(re.search(pattern, line) for pattern in education_patterns)
        
        if is_education_line:
            # Clean and format the education information
            line = re.sub(r'\s+', ' ', line)  # Remove extra spaces
            line = line.strip()
            if line and len(line) > 5:  # Avoid very short lines
                current_education.append(line)
    
    # Combine related education information
    if current_education:
        education = current_education  # Return as list instead of joining
    
    return education

def extract_experience(text):
    experience = []
    # Expanded experience patterns with more variations
    exp_patterns = [
        r'(?i)(\d+\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience)',
        r'(?i)(experience:\s*\d+\+?\s*(?:years?|yrs?))',
        r'(?i)(worked\s*(?:for)?\s*\d+\+?\s*(?:years?|yrs?))',
        r'(?i)(\d+\+?\s*(?:years?|yrs?)\s*(?:in)?\s*(?:the)?\s*(?:field|industry))',
        r'(?i)(senior|lead|principal|architect|manager|director)',
        r'(?i)(\d+\+?\s*(?:years?|yrs?)\s*(?:as)?\s*(?:a)?\s*(?:developer|engineer|architect|manager))',
        r'(?i)(\d+\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:professional|technical|industry)\s*(?:experience))',
        r'(?i)(\d+\+?\s*(?:years?|yrs?)\s*(?:in)?\s*(?:software|web|mobile|cloud|data)\s*(?:development|engineering))',
        r'(?i)(\d+\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:hands-on|practical|relevant)\s*(?:experience))',
        r'(?i)(\d+\+?\s*(?:years?|yrs?)\s*(?:in)?\s*(?:project|team|product)\s*(?:management))',
        r'(?i)(\d+\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:leadership|management)\s*(?:experience))'
    ]
    
    # Extract experience using patterns
    for pattern in exp_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            exp = match.group().strip()
            if exp and exp not in experience:
                experience.append(exp)
    
    # Extract experience from job titles and roles
    job_titles = [
        r'(?i)(software engineer|developer|architect|manager|director|lead|principal)',
        r'(?i)(senior|junior|lead|principal)\s+(?:software|web|mobile|cloud|data)\s+(?:engineer|developer|architect)',
        r'(?i)(full stack|front end|back end|devops|data|cloud)\s+(?:engineer|developer|architect)',
        r'(?i)(project|product|technical|team)\s+(?:manager|lead|architect)',
        r'(?i)(senior|lead|principal)\s+(?:consultant|specialist|analyst)'
    ]
    
    for pattern in job_titles:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            exp = match.group().strip()
            if exp and exp not in experience:
                experience.append(exp)
    
    return experience

def extract_skills(text):
    # Expanded list of technical skills
    common_skills = [
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust',
        # Web Technologies
        'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'asp.net',
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'cassandra',
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform', 'ansible',
        # Frameworks & Tools
        'react native', 'flutter', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        # Methodologies
        'agile', 'scrum', 'waterfall', 'devops', 'microservices', 'rest api', 'graphql',
        # Additional Skills
        'machine learning', 'artificial intelligence', 'data science', 'big data', 'blockchain',
        'cybersecurity', 'cloud computing', 'mobile development', 'web development',
        'software architecture', 'system design', 'project management'
    ]
    
    skills = []
    text_lower = text.lower()
    
    # Check for exact matches
    for skill in common_skills:
        if skill in text_lower:
            skills.append(skill)
    
    # Check for variations (e.g., "python programming" or "python developer")
    for skill in common_skills:
        variations = [
            f"{skill} programming",
            f"{skill} developer",
            f"{skill} development",
            f"{skill} engineer",
            f"{skill} engineering"
        ]
        for variation in variations:
            if variation in text_lower and skill not in skills:
                skills.append(skill)
                break
    
    return skills

def calculate_score(resume_text, job_description, skills, education, experience):
    # Initialize scores with adjusted weights
    skill_score = 0
    education_score = 0
    experience_score = 0
    context_score = 0
    
    # Calculate skill match score (40% weight)
    job_skills = extract_skills(job_description)
    if job_skills:
        matched_skills = set(skills).intersection(set(job_skills))
        skill_score = (len(matched_skills) / len(job_skills)) * 40
        # Bonus for having more than 50% of required skills
        if len(matched_skills) / len(job_skills) > 0.5:
            skill_score *= 1.2
        # Additional bonus for having more than 80% of required skills
        if len(matched_skills) / len(job_skills) > 0.8:
            skill_score *= 1.3
    else:
        skill_score = 0
    
    # Calculate education match score (30% weight)
    job_education = extract_education(job_description)
    if job_education:
        education_matches = 0
        for job_edu in job_education:
            for resume_edu in education:
                # More lenient matching for education
                job_keywords = set(job_edu.lower().split())
                resume_keywords = set(resume_edu.lower().split())
                if len(job_keywords.intersection(resume_keywords)) >= 2:  # At least 2 matching keywords
                    education_matches += 1
                    break
        education_score = (education_matches / len(job_education)) * 30
        # Bonus for having all required education
        if education_matches == len(job_education):
            education_score *= 1.2
        # Additional bonus for having higher education than required
        if any('phd' in edu.lower() or 'doctorate' in edu.lower() for edu in education):
            education_score *= 1.1
    else:
        education_score = 0
    
    # Calculate experience match score (30% weight)
    job_experience = extract_experience(job_description)
    if job_experience and experience:
        exp_matches = 0
        for job_exp in job_experience:
            for resume_exp in experience:
                # More lenient matching for experience
                job_keywords = set(job_exp.lower().split())
                resume_keywords = set(resume_exp.lower().split())
                if len(job_keywords.intersection(resume_keywords)) >= 2:  # At least 2 matching keywords
                    exp_matches += 1
                    break
        experience_score = (exp_matches / len(job_experience)) * 30
        # Bonus for having all required experience
        if exp_matches == len(job_experience):
            experience_score *= 1.2
        # Additional bonus for having more experience than required
        if any('senior' in exp.lower() or 'lead' in exp.lower() or 'principal' in exp.lower() for exp in experience):
            experience_score *= 1.1
    else:
        experience_score = 0
    
    # Calculate context match score (20% weight)
    try:
        # Preprocess text for better matching
        def preprocess_text(text):
            # Remove special characters and extra spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.lower().strip()
        
        # Preprocess both texts
        processed_resume = preprocess_text(resume_text)
        processed_job = preprocess_text(job_description)
        
        # Use TF-IDF with better parameters
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Consider single words, pairs, and triplets
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            analyzer='word'  # Use word-level analysis
        )
        
        tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_job])
        context_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 20
        
        # Boost context score if there's significant overlap
        if context_score > 10:  # If already a good match
            context_score *= 1.2
            
        # Additional boost for keyword matches
        job_keywords = set(processed_job.split())
        resume_keywords = set(processed_resume.split())
        keyword_overlap = len(job_keywords.intersection(resume_keywords)) / len(job_keywords)
        if keyword_overlap > 0.3:  # If more than 30% keywords match
            context_score *= (1 + keyword_overlap)
            
    except:
        context_score = 0
    
    # Calculate total score
    total_score = skill_score + education_score + experience_score + context_score
    
    # Ensure scores are within reasonable ranges
    total_score = min(100, max(0, total_score))
    skill_score = min(40, max(0, skill_score))
    education_score = min(30, max(0, education_score))
    experience_score = min(30, max(0, experience_score))
    context_score = min(20, max(0, context_score))
    
    return {
        'total_score': round(total_score, 2),
        'skill_score': round(skill_score, 2),
        'education_score': round(education_score, 2),
        'experience_score': round(experience_score, 2),
        'context_score': round(context_score, 2)
    }

@csrf_exempt
def upload_resumes(request):
    if request.method == 'POST':
        try:
            files = request.FILES.getlist('resumes')
            job_description = request.POST.get('job_description', '')
            
            if not files:
                return JsonResponse({'error': 'No files uploaded'}, status=400)
            
            if not job_description:
                return JsonResponse({'error': 'Job description is required'}, status=400)
            
            # Delete all previous resumes and their files
            previous_resumes = Resume.objects.all()
            for resume in previous_resumes:
                if resume.file and os.path.isfile(resume.file.path):
                    os.remove(resume.file.path)
            previous_resumes.delete()
            
            # Delete all previous job descriptions
            JobDescription.objects.all().delete()
            
            # Save new job description
            job_desc = JobDescription.objects.create(description=job_description)
            
            results = []
            for file in files:
                try:
                    # Validate file type
                    if not file.name.lower().endswith(('.pdf', '.docx')):
                        logger.warning(f"Invalid file type: {file.name}")
                        continue
                    
                    # Save resume file
                    resume = Resume.objects.create(
                        file=file,
                        name=file.name
                    )
                    
                    # Extract text from resume
                    file_path = resume.file.path
                    logger.info(f"Processing file: {file_path}")
                    
                    text = extract_text_from_file(file_path)
                    
                    if not text:
                        logger.warning(f"No text extracted from {file.name}")
                        resume.delete()  # Delete the resume if no text was extracted
                        continue
                    
                    logger.info(f"Successfully extracted text from {file.name}")
                    
                    # Extract information
                    skills = extract_skills(text)
                    education = extract_education(text)
                    experience = extract_experience(text)
                    
                    logger.info(f"Extracted information from {file.name}:")
                    logger.info(f"Skills: {skills}")
                    logger.info(f"Education: {education}")
                    logger.info(f"Experience: {experience}")
                    
                    # Calculate scores
                    scores = calculate_score(text, job_description, skills, education, experience)
                    
                    # Update resume with extracted information and scores
                    resume.skills = ', '.join(skills)
                    resume.education = ' | '.join(education)
                    resume.experience = ' | '.join(experience)
                    resume.score = scores['total_score']
                    resume.skill_score = scores['skill_score']
                    resume.education_score = scores['education_score']
                    resume.experience_score = scores['experience_score']
                    resume.context_score = scores['context_score']
                    resume.save()
                    
                    results.append({
                        'id': resume.id,
                        'name': resume.name,
                        'total_score': scores['total_score'],
                        'skill_score': scores['skill_score'],
                        'education_score': scores['education_score'],
                        'experience_score': scores['experience_score'],
                        'context_score': scores['context_score'],
                        'skills': resume.skills,
                        'education': resume.education,
                        'experience': resume.experience
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing file {file.name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    if 'resume' in locals():
                        resume.delete()  # Clean up if resume was created
                    continue
            
            if not results:
                return JsonResponse({'error': 'No valid resumes were processed. Please check if the files are valid PDF or DOCX files.'}, status=400)
            
            # Sort results by total score
            results.sort(key=lambda x: x['total_score'], reverse=True)
            
            return JsonResponse({'results': results})
            
        except Exception as e:
            logger.error(f"Error in upload_resumes: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({'error': 'An error occurred while processing the resumes'}, status=500)
    
    return render(request, 'core/upload.html')

def selected_resumes(request):
    # Get the latest job description
    job_description = JobDescription.objects.order_by('-id').first()
    
    # Get all resumes with scores, ordered by total score and best match status
    resumes = Resume.objects.filter(
        score__isnull=False
    ).order_by('-is_best_match', '-score')
    
    return render(request, 'core/selected_resumes.html', {
        'job_description': job_description,
        'resumes': resumes
    })

@csrf_exempt
def remove_resume(request, resume_id):
    try:
        resume = Resume.objects.get(id=resume_id)
        resume.delete()
        return JsonResponse({'success': True})
    except Resume.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Resume not found'}, status=404)

def download_resume(request, resume_id):
    try:
        resume = Resume.objects.get(id=resume_id)
        file_path = resume.file.path
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=resume.name)
    except Resume.DoesNotExist:
        return JsonResponse({'error': 'Resume not found'}, status=404)

def home(request):
    return render(request, 'core/home.html')

def best_resumes(request):
    # Get the latest job description
    job_description = JobDescription.objects.order_by('-id').first()
    
    # Get all resumes with scores, ordered by total score
    resumes = Resume.objects.filter(
        score__isnull=False
    ).order_by('-score', '-is_best_match')
    
    return render(request, 'core/best_resumes.html', {
        'job_description': job_description,
        'resumes': resumes
    })

@csrf_exempt
def mark_as_best(request, resume_id):
    try:
        resume = Resume.objects.get(id=resume_id)
        resume.is_best_match = True
        resume.save()
        return JsonResponse({'success': True})
    except Resume.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Resume not found'}, status=404)

@csrf_exempt
def unmark_as_best(request, resume_id):
    try:
        resume = Resume.objects.get(id=resume_id)
        resume.is_best_match = False
        resume.save()
        return JsonResponse({'success': True})
    except Resume.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Resume not found'}, status=404)

@csrf_exempt
def mark_best_multiple(request):
    try:
        data = json.loads(request.body)
        resume_ids = data.get('resume_ids', [])
        
        Resume.objects.filter(id__in=resume_ids).update(is_best_match=True)
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@csrf_exempt
def remove_resumes(request):
    try:
        data = json.loads(request.body)
        resume_ids = data.get('resume_ids', [])
        
        # Delete the files first
        resumes = Resume.objects.filter(id__in=resume_ids)
        for resume in resumes:
            if resume.file:
                if os.path.isfile(resume.file.path):
                    os.remove(resume.file.path)
        
        # Delete the database entries
        resumes.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400) 