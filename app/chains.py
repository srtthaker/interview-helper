import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

class Chain:
    def __init__(self):
        self.model = ChatOllama(model="gemma3:12b")
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            text: {cleaned_text}
            The scraped text is from the careers page of a website. Your job is to extract important information from the job postings and return them in a JSON format
            containing the following keys: `role`, `experience`, `skills`, and `description`.

            Only return the valid JSON.
            """
        )
        chain_extract = prompt_extract | self.model
        res = chain_extract.invoke(input={"cleaned_text": cleaned_text})
        #print(res)
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException as e:
            print(f"Error parsing JSON: {e}")
            return None
        return res if isinstance(res, list) else [res]

    def generate_questions(self, job_details):
        prompt_questions = PromptTemplate.from_template(
            """
            Job details:
            role: {role}
            experience: {experience}
            skills: {skills}
            description: {description}

            Generate 7 interview questions based on the job details provided. The questions should be relevant to the role and skills required. Include a mix of technical and behavioral questions.

            Return the questions as a numbered list. No preamble and no postamble.
            """
        )
        chain_questions = prompt_questions | self.model | self.str_parser
        try:
            questions = chain_questions.invoke(
                {
                    "role": job_details.get("role", "N/A"),
                    "experience": job_details.get("experience", "N/A"),
                    "skills": ", ".join(job_details.get("skills", [])),
                    "description": job_details.get("description", "N/A"),
                }
            )
            return questions
        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return "Could not generate interview questions."

    def generate_answers(self, questions, job_details, portfolio):
        prompt_answers = PromptTemplate.from_template(
            """
            interview questions: {questions}

            job details:
            role: {role}
            experience: {experience}
            skills: {skills}
            description: {description}

            portfolio: {portfolio}

            Imagine you are a candidate applying for the job described above. Your task is to generate clear and concise answers to the interview questions provided. Use projects from your portfolio to support your answers if possible. 

            Return the answers as a numbered list, mirroring the order of the questions. No preamble and no postamble.
            """
        )
        chain_answers = prompt_answers | self.model | self.str_parser

        portfolio_str = (
            "\n".join(f"- {desc}" for desc in portfolio)
            if portfolio
            else "No specific portfolio projects provided for context."
        )

        try:
            answers = chain_answers.invoke(
                {
                    "questions": questions,
                    "role": job_details.get("role", "N/A"),
                    "experience": job_details.get("experience", "N/A"),
                    "skills": ", ".join(job_details.get("skills", [])),
                    "description": job_details.get("description", "N/A"),
                    "portfolio": portfolio_str,
                }
            )
            return answers
        except Exception as e:
            print(f"Error generating answers: {e}")
            return "Could not generate answers."
