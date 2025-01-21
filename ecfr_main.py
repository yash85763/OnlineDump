from ecfr_api_wrapper import ECFRAPIWrapper
from utils import write_regulation_to_file
import requests
from utils.qa_processor import QAProcessor


def main():
    # Initialize with OpenAI API key
    openai_api_key = "your_api_key"  # Replace with actual API key
    api = ECFRAPIWrapper(openai_api_key=openai_api_key)
    qa_processor = QAProcessor(api_key=openai_api_key)
    
    try:
        # Define parameters
        date = "2024-12-12"
        title = "12"
        subtitle = "A"
        chapter = "II"
        subchapter = "A"
        part = "211"
        subpart = "A"
        section = "211.3"

        # Create filename
        filename = f"LLM_POWERED_ecfr_output_{date}_{title}_{subtitle}_{chapter}_{subchapter}_{part}_{subpart}_{section}.txt"
        answers_filename = f"ANSWERS_ecfr_qa_{date}_{title}_{subtitle}_{chapter}_{subchapter}_{part}_{subpart}_{section}.txt"
        
        # Get regulation data
        result = api.get_structured_regulation(
            date=date,
            title=title,
            subtitle=subtitle,
            chapter=chapter,
            subchapter=subchapter,
            part=part,
            subpart=subpart,
            section=section
        )
        
        # Write to file using utility function
        write_regulation_to_file(result, filename)
        print(f"Output has been written to {filename}")

        # Read content and process questions
        content = qa_processor.read_regulation_file(filename)
        questions = [
            "question 1?",
            "question 2?",
            "question 3"
        ]
        
        # Get and print answers
        print("\nAnswers to your questions:")
        print("-" * 40)
        answers = qa_processor.answer_questions(content, questions)
        print(answers[0])

        # Write answers to file
        with open(answers_filename, 'w', encoding='utf-8') as f:
            f.write("Questions and Answers for Regulation Section:\n")
            f.write(f"Date: {date}\n")
            f.write(f"Title: {title}\n")
            f.write(f"Section: {section}\n\n")
            f.write(answers[0])
        
        print(f"Answers written to: {answers_filename}")
        
        # Also print answers to console
        print("\nAnswers to your questions:")
        print("-" * 40)
        print(answers[0])
        
    except (ValueError, requests.exceptions.RequestException) as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
