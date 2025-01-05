from flask import Flask, render_template, request
import os
import io
from werkzeug.utils import secure_filename
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Tuple

app = Flask(__name__)

# Directory for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'pdf'}

class MultiDomainResumeAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define domain skills database
        self.domain_skills_db = {
            'technology': {
                'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'php', 'scala', 'swift', 'kotlin', 
                              'r', 'matlab', 'typescript', 'go', 'rust', 'perl', 'shell scripting'],
                'databases': ['sql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'redis', 'cassandra', 
                            'dynamodb', 'sqlite', 'neo4j', 'elasticsearch'],
                'frameworks': ['django', 'flask', 'react', 'angular', 'vue', 'spring', 'nodejs', 
                             'express', 'fastapi', 'laravel', 'bootstrap', 'tailwind'],
                'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 
                         'ansible', 'cloud computing', 'serverless', 'microservices'],
                'machine_learning': ['tensorflow', 'pytorch', 'scikit-learn', 'machine learning', 'deep learning',
                                   'neural networks', 'nlp', 'computer vision', 'data mining'],
                'tools': ['tableau', 'excel', 'standardscaler', 'tfidfvectorizer', 'github', 'leetcode',
                         'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv', 'yolov8'],
                'nlp': ['text preprocessing', 'tokenization', 'tfidfvectorizer'],
                'data_visualization': ['tableau', 'excel', 'matplotlib', 'seaborn'],
                'image_processing': ['opencv', 'yolov8', 'image annotation']                   
            },
            'finance': {
                'accounting': ['financial accounting', 'bookkeeping', 'balance sheet', 'income statement', 'cash flow',
                   'accounts payable', 'accounts receivable', 'tax accounting', 'audit', 'forensic accounting',
                   'corporate finance', 'cost accounting'],
                'analysis': ['financial analysis', 'risk assessment', 'portfolio management', 'valuation',
                 'financial modeling', 'forecasting', 'budgeting', 'cost analysis', 'credit analysis',
                 'liquidity analysis', 'capital budgeting'],
                'tools': ['excel', 'quickbooks', 'sap', 'bloomberg terminal', 'tableau', 'power bi',
                    'hyperion', 'sage', 'xero', 'factset', 'oracle financials', 'netsuite', 'quicken'],
                'concepts': ['investment banking', 'private equity', 'hedge funds', 'derivatives', 'options trading',
                 'merger and acquisition', 'capital markets', 'fixed income', 'equity research',
                 'asset management', 'financial risk management', 'wealth management'],
                'regulations': ['basel', 'sarbanes-oxley', 'dodd-frank', 'ifrs', 'gaap',
                    'sec regulations', 'aml', 'kyc', 'finra', 'fatca', 'crs']
            },
            'hr': {
                'recruitment': ['talent acquisition', 'interviewing', 'job posting', 'candidate screening', 'onboarding',
                    'sourcing', 'employer branding', 'job description writing', 'recruitment marketing',
                    'headhunting', 'candidate relationship management', 'diversity recruitment'],
                'systems': ['hris', 'ats', 'workday', 'successfactors', 'bamboo hr', 'oracle hcm',
                'peoplesoft', 'kronos', 'ultipro', 'paycom', 'zenefits', 'sap successfactors'],
                'functions': ['benefits administration', 'compensation', 'employee relations', 'performance management',
                  'payroll', 'leave management', 'workplace safety', 'employee engagement',
                  'talent management', 'conflict resolution', 'organizational development'],
                'compliance': ['labor laws', 'employment law', 'osha', 'workplace safety', 'diversity and inclusion',
                   'ada compliance', 'eeoc', 'fmla', 'flsa', 'hipaa', 'gdpr', 'iso compliance'],
                'development': ['training', 'leadership development', 'succession planning', 'career development',
                    'mentoring', 'coaching', 'skill assessment', 'performance evaluation',
                    'learning management systems', 'team building']
            },
            'teaching': {
                'pedagogy': ['lesson planning', 'curriculum development', 'student assessment', 'classroom management',
                 'behavior management', 'educational psychology', 'inclusive education'],
                'methods': ['differentiated instruction', 'project-based learning', 'blended learning', 'inquiry-based learning',
                'flipped classroom', 'experiential learning', 'direct instruction', 'collaborative learning'],
                'tools': ['blackboard', 'canvas', 'moodle', 'google classroom', 'educational technology',
              'kahoot', 'quizlet', 'edmodo', 'nearpod', 'zoom for education'],
                'skills': ['student engagement', 'classroom management', 'special education', 'esl', 'iep development',
               'conflict resolution', 'cultural competency', 'parent communication'],
                'assessment': ['formative assessment', 'summative assessment', 'rubric development', 'standardized testing',
                   'peer assessment', 'self-assessment', 'portfolio assessment']
            },
            'marketing': {
                'digital': ['seo', 'sem', 'social media marketing', 'content marketing', 'email marketing',
                'affiliate marketing', 'video marketing', 'influencer marketing', 'mobile marketing'],
                'analytics': ['google analytics', 'marketing metrics', 'conversion optimization', 'a/b testing',
                  'data visualization', 'predictive analytics', 'customer segmentation'],
                'tools': ['hubspot', 'mailchimp', 'google ads', 'facebook ads', 'adobe creative suite',
              'canva', 'semrush', 'ahrefs', 'hootsuite', 'buffer', 'sprout social'],
                'skills': ['brand management', 'market research', 'campaign planning', 'copywriting',
               'product marketing', 'pricing strategies', 'competitive analysis'],
                'social': ['facebook', 'instagram', 'linkedin', 'twitter', 'tiktok',
               'youtube', 'snapchat', 'pinterest', 'reddit']
            }
        }

        self.common_skills = {
            'soft_skills': ['leadership', 'communication', 'problem solving', 'organization'],
            'project_management': ['project planning', 'team management', 'event management']
        }
    def get_skill_gaps(self, resume_skills: Dict[str, List[str]], 
                      job_skills: Dict[str, List[str]]) -> List[str]:
        """
        Identify missing skills by comparing resume skills with job requirements.
        Returns a flat list of missing skills.
        """
        missing_skills = []
        
        # Compare technical skills
        job_technical = set(job_skills.get('technical', []))
        resume_technical = set(resume_skills.get('technical', []))
        missing_technical = job_technical - resume_technical
        missing_skills.extend(list(missing_technical))
        
        # Compare soft skills
        job_soft = set(job_skills.get('soft', []))
        resume_soft = set(resume_skills.get('soft', []))
        missing_soft = job_soft - resume_soft
        missing_skills.extend(list(missing_soft))
        
        return sorted(missing_skills)
    def extract_text_from_pdf(self, file) -> str:
        """
        Extract text from a PDF file using PyPDF2.
        Args:
            file: File object from request.files
        Returns:
            str: Extracted text from PDF
        """
        text = ""
        try:
            # Create a BytesIO object from the file's content
            pdf_bytes = io.BytesIO(file.read())
            
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)
            
            # Extract text from each page
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
                
            print("Extracted text:", text[:200])  # Print first 200 chars for debugging
            return text
            
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def extract_skills(self, text: str, domain: str) -> Dict[str, List[str]]:
        text = text.lower()
        found_skills = {'technical': [], 'soft': []}
    
        for domain_cat in self.domain_skills_db.values():
            for category, skills in domain_cat.items():
                for skill in skills:
                    # More flexible pattern matching
                    pattern = r'\b' + re.escape(skill).replace('\\+', '[+]?') + r'\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        found_skills['technical'].append(skill)

        # Extract soft skills
        for domain_cat in self.domain_skills_db.values():
            for skill in skills:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text):
                    found_skills['soft'].append(skill)
    
        found_skills['technical'] = sorted(list(set(found_skills['technical'])))
        print(f"Found skills for domain {domain}:", found_skills)  # Debug print
        found_skills['soft'] = sorted(list(set(found_skills['soft'])))
        return found_skills

    def calculate_similarity_score(self, resume_skills: Dict[str, List[str]], 
                                 job_skills: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate weighted similarity scores between resume skills and job requirements.
        Returns a dictionary with different similarity metrics.
        """
        scores = {}
        
        # Calculate technical skills similarity
        job_technical = set(job_skills.get('technical', []))
        resume_technical = set(resume_skills.get('technical', []))
        
        if job_technical:
            # Calculate exact matches
            matching_technical = resume_technical.intersection(job_technical)
            technical_score = len(matching_technical) / len(job_technical) * 100
            
            # Calculate partial matches
            partial_matches = 0
            remaining_job_skills = job_technical - matching_technical
            remaining_resume_skills = resume_technical - matching_technical
            
            for job_skill in remaining_job_skills:
                for resume_skill in remaining_resume_skills:
                    if (job_skill in resume_skill) or (resume_skill in job_skill):
                        partial_matches += 0.5
            
            technical_score += (partial_matches / len(job_technical)) * 100
            scores['technical'] = min(technical_score, 100)
        else:
            scores['technical'] = 0
            
        # Calculate soft skills similarity
        job_soft = set(job_skills.get('soft', []))
        resume_soft = set(resume_skills.get('soft', []))
        
        if job_soft:
            matching_soft = resume_soft.intersection(job_soft)
            scores['soft'] = len(matching_soft) / len(job_soft) * 100
        else:
            scores['soft'] = 0
            
        # Domain-specific weights
        domain_weights = {
            'technology': {'technical': 0.8, 'soft': 0.2},
            'finance': {'technical': 0.7, 'soft': 0.3},
            'hr': {'technical': 0.6, 'soft': 0.4},
            'marketing': {'technical': 0.7, 'soft': 0.3},
            'teaching': {'technical': 0.6, 'soft': 0.4}
        }
        
        # Get domain
        domain = detect_job_domain(' '.join(job_technical))
        weights = domain_weights.get(domain, domain_weights['technology'])
        
        # Calculate weighted overall score
        overall_score = (
            scores['technical'] * weights['technical'] +
            scores['soft'] * weights['soft']
        )
        
        # Store all scores as pre-calculated percentages
        scores['technical'] = round(scores['technical'], 2)
        scores['soft'] = round(scores['soft'], 2)
        scores['overall'] = round(overall_score, 2)
        
        return scores
    def recommend_courses(self, skill_gaps: List[str], domain: str) -> Dict[str, List[str]]:
      """Recommend courses based on skill gaps"""
      course_recommendations = {
        'technology': {
          'python': [
            'Complete Python Bootcamp: From Zero to Hero',
            'Python for Data Science and Machine Learning',
            'Advanced Python Programming',
            'Python Web Development with Django'
          ],
          'java': [
            'Java Programming Masterclass',
            'Spring Framework Complete Course',
            'Java Enterprise Edition Development',
            'Android App Development with Java'
          ],
          'javascript': [
            'Modern JavaScript from the Beginning',
            'Full Stack JavaScript Development',
            'React & Node.js Complete Developer Course',
            'Advanced JavaScript Concepts'
          ],
          'c++': [
            'C++ Programming from Scratch',
            'Game Development with C++',
            'Advanced C++ Programming',
            'Data Structures & Algorithms in C++'
          ],
          'typescript': [
            'TypeScript Complete Developer Guide',
            'Angular with TypeScript',
            'Enterprise TypeScript Development',
            'React & TypeScript Projects'
          ],
          'sql': [
            'Complete SQL Bootcamp',
            'Database Design & Management',
            'Advanced SQL for Data Analytics',
            'PostgreSQL Administration'
          ],
          'mongodb': [
            'MongoDB Complete Developer Guide',
            'NoSQL Database Development',
            'MERN Stack Development',
            'MongoDB Performance Optimization'
          ],
          'redis': [
            'Redis Fundamentals',
            'Caching Strategies with Redis',
            'Redis for Microservices',
            'High-Performance Systems with Redis'
          ],
          'react': [
            'React Complete Guide',
            'Advanced React and Redux',
            'React Native Mobile Development',
            'Full Stack React Development'
          ],
          'angular': [
            'Angular Complete Course',
            'Enterprise Angular Development',
            'Angular with TypeScript',
            'Full Stack Angular Projects'
          ],
          'django': [
            'Django Web Development',
            'Python & Django Full Stack',
            'Django REST Framework',
            'Django Projects Bootcamp'
          ],
          'aws': [
            'AWS Certified Solutions Architect',
            'AWS Developer Associate',
            'AWS DevOps Professional',
            'Serverless Applications on AWS'
          ],
          'azure': [
            'Microsoft Azure Fundamentals',
            'Azure Solutions Architect',
            'Azure DevOps Engineer',
            'Cloud Computing with Azure'
          ],
          'kubernetes': [
            'Kubernetes for Beginners',
            'Certified Kubernetes Administrator',
            'Container Orchestration',
            'Microservices with Kubernetes'
          ],
          'tensorflow': [
            'Deep Learning with TensorFlow',
            'TensorFlow Developer Certificate',
            'Machine Learning Projects',
            'Neural Networks with TensorFlow'
          ],
          'pytorch': [
            'PyTorch for Deep Learning',
            'Computer Vision with PyTorch',
            'Natural Language Processing',
            'Advanced AI Projects'
          ],
          'scikit-learn': [
            'Machine Learning with Scikit-Learn',
            'Data Science Bootcamp',
            'Predictive Analytics',
            'Statistical Learning'
          ]
        },
        'finance': {
          'financial accounting': [
            'Financial Accounting Fundamentals',
            'CPA Exam Preparation',
            'Corporate Accounting',
            'Advanced Financial Statements'
          ],
          'bookkeeping': [
            'Bookkeeping Basics',
            'QuickBooks Professional',
            'Small Business Accounting',
            'Advanced Bookkeeping'
          ],
          'audit': [
            'Internal Audit Fundamentals',
            'External Audit Practice',
            'Risk-Based Auditing',
            'IT Audit and Controls'
          ],
          'financial analysis': [
            'Financial Analysis Masterclass',
            'Investment Analysis',
            'Corporate Finance',
            'Financial Modeling'
          ],
          'risk assessment': [
            'Risk Management Professional',
            'Financial Risk Analysis',
            'Enterprise Risk Management',
            'Credit Risk Assessment'
          ]
        },
        'hr': {
          'talent acquisition': [
            'Strategic Talent Acquisition',
            'Recruitment and Selection',
            'HR Analytics for Recruiting',
            'Advanced Sourcing Techniques'
          ],
          'interviewing': [
            'Effective Interview Techniques',
            'Behavioral Interviewing',
            'Technical Recruitment',
            'Interview Assessment Methods'
          ],
          'ats': [
            'Mastering Applicant Tracking Systems',
            'Optimizing Recruitment with ATS Tools',
            'Advanced ATS Strategies for HR',
            'ATS Implementation and Best Practices'
        ],
        'hris': [
            'HRIS Fundamentals: A Complete Guide',
            'Implementing and Managing HRIS Systems',
            'HRIS for Workforce Analytics',
            'Advanced HRIS Configuration and Management'
        ],
        'payroll': [
            'Payroll Management Essentials',
            'Advanced Payroll Processing and Compliance',
            'Payroll Systems Implementation',
            'Global Payroll and Taxation Strategies'
        ],
        'training': [
            'Designing Effective Training Programs',
            'Corporate Training and Development',
            'Training Needs Analysis Masterclass',
            'E-Learning Development for Trainers'
        ],
          'workday': [
            'Workday HCM Administration',
            'Workday Implementation',
            'Workday Report Writing',
            'Workday Integration'
          ],
          'labor laws': [
            'Employment Law Essentials',
            'Labor Relations Management',
            'HR Compliance Certificate',
            'Workplace Investigations'
          ],
          'diversity and inclusion': [
            'DEI Strategy Development',
            'Inclusive Leadership',
            'Building Inclusive Workplaces',
            'Cultural Competence Training'
          ]
        },
        'marketing': {
            'digital': [
                'Complete Digital Marketing Course',
                'SEO Mastery for Beginners',
                'Social Media Marketing Fundamentals',
                'Content Marketing Strategy'
            ],
            'analytics': [
                'Marketing Analytics Mastery',
                'Google Analytics Certification',
                'A/B Testing and Optimization',
                'Predictive Marketing Analytics'
            ]
        }
      }


      recommendations = {}
      domain_courses = course_recommendations.get(domain, {})

      for skill in skill_gaps:
        found_course = False
        # Try to find exact match in current domain first
        if skill in domain_courses:
            recommendations[skill] = domain_courses[skill]
            found_course = True
        else:
            # Try partial matches in current domain
            for course_skill in domain_courses.keys():
                if (skill in course_skill) or (course_skill in skill):
                    recommendations[skill] = domain_courses[course_skill]
                    found_course = True
                    break
        
        # If no course found in current domain, search other domains
        if not found_course:
            for other_domain, other_courses in course_recommendations.items():
                if other_domain != domain:
                    if skill in other_courses:
                        recommendations[skill] = other_courses[skill]
                        break
                    else:
                        # Try partial matches in other domains
                        for course_skill in other_courses.keys():
                            if (skill in course_skill) or (course_skill in skill):
                                recommendations[skill] = other_courses[course_skill]
                                found_course = True
                                break
                if found_course:
                    break

      return recommendations

def detect_job_domain(job_description: str) -> str:
    description = job_description.lower()
    
    domain_keywords = {
        'technology': ['software', 'developer', 'programming', 'code', 'technical', 'engineer', 
                      'python', 'machine learning', 'data science', 'algorithm'],  # Added more tech keywords
        'finance': ['financial', 'accounting', 'finance', 'banking', 'investment'],
        'hr': ['hr', 'human resources', 'recruitment', 'hiring', 'talent'],
        'marketing': ['seo', 'marketing', 'campaign', 'digital', 'branding'],
        'teaching': ['teaching', 'pedagogy', 'curriculum', 'education', 'training']
    }
    
    # Count occurrences of keywords for each domain
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in description)
        domain_scores[domain] = score
    if all(score == 0 for score in domain_scores.values()):
        return "Error: Could not detect job domain from the description."
    # Return the domain with the highest score
    return max(domain_scores.items(), key=lambda x: x[1])[0]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        resume_file = request.files['resume']
        job_description = request.form['job_description']
        
        if resume_file and allowed_file(resume_file.filename):
            analyzer = MultiDomainResumeAnalyzer()
            
            # Extract text from resume
            resume_text = analyzer.extract_text_from_pdf(resume_file)
            if not resume_text:
                return "Error: Could not extract text from resume"
            
             # Detect domain from resume_text instead of job_description
            domain = detect_job_domain(resume_text)  # Changed this line
            print(f"Detected domain from resume: {domain}")  # Debug print
            if "Error" in domain:
                return f"Error: {domain}"
            
            # Extract skills
            resume_skills = analyzer.extract_skills(resume_text, domain)
            job_skills = analyzer.extract_skills(job_description, domain)
            
            # Calculate similarity
            similarity_scores = analyzer.calculate_similarity_score(resume_skills, job_skills)
            overall_score = similarity_scores['overall']
            # Get flat list of missing skills
            missing_skills = analyzer.get_skill_gaps(resume_skills, job_skills)
            
            # Get course recommendations for missing skills
            course_recommendations = analyzer.recommend_courses(missing_skills, domain)

            
            return render_template(
                'result.html',
                similarity_score=overall_score,
                domain=domain,
                resume_skills=resume_skills['technical'],
                job_skills=job_skills['technical'],
                skill_gaps=course_recommendations  # Changed from skill_gaps to course_recommendations
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)