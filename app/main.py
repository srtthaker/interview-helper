import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_app(model, portfolio, clean_text):
    st.title("Interview Question Generator")

    url_input = st.text_input(
        "Enter the URL of the job posting to generate a set of interview questions."
    )
    submit_button = st.button("Submit")

    if submit_button:
        loader = WebBaseLoader([url_input])
        data = clean_text(loader.load().pop().page_content)
        portfolio.load_portfolio()
        jobs = model.extract_jobs(data)
        for job in jobs:
            skills = job.get("skills", [])
            projects = portfolio.query_portfolio(skills)

            st.subheader("Interview Questions:")
            questions = model.generate_questions(job)
            st.code(questions, language="markdown")

            st.subheader("Interview Question Answers:")
            answers = model.generate_answers(questions, job, projects)
            st.code(answers, language="markdown")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Interview Question Generator")
    create_app(chain, portfolio, clean_text)
