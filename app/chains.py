import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Ekta, an AI Intern at Rajasthan Police HeadQuarter. I have worked on multiple projects like- Big Bang ChatBot: Custom-Generated Conversations for ’The Big Bang Theory §
• Developed a sophisticated chatbot leveraging a custom transformer model in the pre-training stage to simulate engaging and informative
conversations about the TV show ”The Big Bang Theory.”
• Collected and preprocessed extensive data from the show’s scripts and transcripts and integrated the transformer model into a chatbot framework and
created comprehensive conversation flows using CUDA virtual environment.
RajBot: A Generative AI model for the RP website §
• Rajbot is basically a Chatbot that assists users on the Rajasthan Police website, enabling efficient retrieval of essential information such as filing
reports, office contact details, and complaint status checks.
• Utilized Dialogflow ES to design the conversational flow and integrate key functionalities like FIR assistance, service information, and navigation
support.
I have the following skills-
Languages:Python,SQL,C++ ,Html 5, CSS.
Machine Learning and AI: TensorFlow, PyTorch, Scikit-learn,Transformers,Natural Language Processing (NLP), Computer Vision.
Developer Tools:V S Code,Android Studio,PyCharm, Eclipse.
CS Fundamentals:DBMS, Object Oriented Programming, Operating System, Computer Networking,Information Cybersecurity System
         
        Your job is to write a cold email to the company regarding the job mentioned above describing the capability of Ekta 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Ekta's Resume: {link_list}
        Remember you are Ekta, AI Intern at Rajasthan Police HeadQuarter. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))