import streamlit as st
import pickle
import re

#loading models

clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def cleanResume(txt):
    cleanTxt = re.sub('http\S+\s'," ",txt)
    cleanTxt = re.sub('RT|CC'," ",cleanTxt)
    cleanTxt = re.sub('@\S+'," ",cleanTxt)
    cleanTxt = re.sub('#\S+\s'," ",cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%'()*+,-./:;<=>?@[\]^_`{|}~""")," ",cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]'," ",cleanTxt)
    cleanTxt = re.sub('\s+'," ",cleanTxt)

    return cleanTxt

def main():
    st.title("Resume Screening App")
    file = st.file_uploader('Upload Your Resume',type=['pdf','txt'])

    if file is not None:
        try:
            resume_bytes = file.read()
            resumeTxt = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resumeTxt = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resumeTxt)
        modified_resume = tfidf.transform([cleaned_resume])
        prediction = clf.predict(modified_resume)[0]
        

        category_mapping = {
            15:"Java Developer",
            23:"Testing",
            8: "Devops Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12:"HR",
            13:"Hadoop",
            3:"Blockchain",
            10:"ETL Developer",
            18: "Opertaions Manager",
            6:"Data Science",
            22:"Sales",
            16:"Mechnaical Engineer",
            1: "Arts",
            7:"Database",
            11:"Electrical Engineering",
            14:"Health and Fitness",
            19:"PMO",
            4:"Bussiness Analyst",
            9:"Donet Developer",
            2:"Automation Testing",
            17:"Network Security Engineer",
            21: "SAP Developer",
            5:"Civil Engineer",
            0:"Advocate"
    
        }

        category = category_mapping.get(prediction,"unknown")
        st.write("Predicted resume category:  ",category)


if __name__ == "__main__":
    main() 
