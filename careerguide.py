import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------
# 1. Simulated dataset of career profiles
data = [
    {
        "job_title": "Data Scientist",
        "skills": "python, machine learning, statistics, data visualization",
        "education": "bachelor",
        "interests": "data, analytics, research",
        "experience": "mid",
        "description": "Analyzes data to extract insights and build predictive models."
    },
    {
        "job_title": "Software Engineer",
        "skills": "java, c++, algorithms, problem solving",
        "education": "bachelor",
        "interests": "coding, development, problem solving",
        "experience": "junior",
        "description": "Designs and develops software applications and systems."
    },
    {
        "job_title": "Graphic Designer",
        "skills": "photoshop, creativity, adobe illustrator, visual design",
        "education": "associate",
        "interests": "art, creativity, media",
        "experience": "entry",
        "description": "Creates visual concepts to communicate ideas."
    },
    {
        "job_title": "Project Manager",
        "skills": "leadership, communication, scheduling, budgeting",
        "education": "bachelor",
        "interests": "management, organization, planning",
        "experience": "senior",
        "description": "Oversees projects to ensure timely delivery within budget."
    },
    {
        "job_title": "Marketing Specialist",
        "skills": "seo, content creation, social media, communication",
        "education": "bachelor",
        "interests": "marketing, branding, communication",
        "experience": "mid",
        "description": "Develops strategies to promote products and brands."
    },
    {
        "job_title": "Cybersecurity Analyst",
        "skills": "network security, python, risk assessment, cryptography",
        "education": "bachelor",
        "interests": "security, technology, risk management",
        "experience": "mid",
        "description": "Protects an organization's computer systems and networks."
    },
    {
        "job_title": "Mechanical Engineer",
        "skills": "cad, thermodynamics, mechanics, problem solving",
        "education": "bachelor",
        "interests": "engineering, mechanics, design",
        "experience": "mid",
        "description": "Designs and tests mechanical devices and systems."
    },
    {
        "job_title": "Financial Analyst",
        "skills": "excel, finance, accounting, data analysis",
        "education": "bachelor",
        "interests": "finance, economics, data",
        "experience": "junior",
        "description": "Provides investment and financial recommendations."
    },
    {
        "job_title": "Teacher",
        "skills": "communication, patience, subject knowledge, mentoring",
        "education": "bachelor",
        "interests": "teaching, education, helping others",
        "experience": "mid",
        "description": "Educates and supports students in learning."
    },
    {
        "job_title": "UX Designer",
        "skills": "wireframing, user research, creativity, prototyping",
        "education": "bachelor",
        "interests": "design, user experience, psychology",
        "experience": "mid",
        "description": "Improves user satisfaction with products by enhancing usability."
    },
]

df = pd.DataFrame(data)

# ---------------------------------
# 2. Data preparation
def combine_text_features(row):
    return f"{row['skills']} {row['education']} {row['interests']} {row['experience']}"

df['combined_features'] = df.apply(combine_text_features, axis=1)

# Vectorize the combined features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined_features'])

# Encode job titles
le = LabelEncoder()
y = le.fit_transform(df['job_title'])

# ---------------------------------
# Model trainingy7
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ---------------------------------
# Streamlit UI
st.title("🎯 Career Path Recommendation System")

st.markdown("Enter your profile information below to get the **top 3 career path recommendations.**")

# User inputs
skills_input = st.text_input("Enter your skills (comma separated):", "")
education_input = st.selectbox("Select your highest educat\;ion level:", ['highschool', 'associate', 'bachelor', 'master', 'phd'])
interests_input = st.text_input("Enter your interests (comma separated):", "")
experience_input = st.selectbox("Select your experience level:", ['entry', 'junior', 'mid', 'senior'])

if st.button("🔍 Recommend Careers"):
    if not skills_input.strip() or not interests_input.strip():
        st.warning("Please fill in both your skills and interests.")
    else:
        user_features = f"{skills_input} {education_input} {interests_input} {experience_input}"
        user_vector = vectorizer.transform([user_features])

        pred_probs = model.predict_proba(user_vector)[0]

        # Cosine similarity
        def text_to_vector(text):
            skill_vectorizer = CountVectorizer(vocabulary=vectorizer.vocabulary_)
            return skill_vectorizer.transform([text])

        user_skills_vec = text_to_vector(skills_input.lower())
        job_skills_vec = vectorizer.transform(df['skills'].str.lower())

        similarities = cosine_similarity(user_skills_vec, job_skills_vec)[0]
        combined_scores = 0.7 * pred_probs + 0.3 * similarities
        top3_idx = combined_scores.argsort()[::-1][:3]

        st.subheader("Top 3 Career Recommendations 🎓")
        for idx in top3_idx:
            st.markdown(f"### {df.iloc[idx]['job_title']}")
            st.write(df.iloc[idx]['description'])
            st.markdown(f"**Required Skills:** {df.iloc[idx]['skills']}")
            st.markdown(f"**Typical Education:** {df.iloc[idx]['education'].capitalize()}")
            st.markdown(f"**Interests:** {df.iloc[idx]['interests']}")
            st.markdown(f"**Experience Level:** {df.iloc[idx]['experience'].capitalize()}")
            st.markdown("---")

# ---------------------------------
# Sidebar Explanation
st.sidebar.title("🧠 How it works")
st.sidebar.info("""
This system uses a simulated dataset of career profiles including skills, education, interests, and experience.

- All features are combined and vectorized.
- A machine learning model (Random Forest) is trained to classify career paths.
- Your input is matched against this model.
- Cosine similarity is added for better matching based on your skills.
- Top 3 most relevant careers are shown based on your profile.
""")
