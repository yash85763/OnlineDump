import os
import json
import requests
from utils.llm_processor import LLMProcessor
from ecfr_api_wrapper import ECFRAPIWrapper
from utils.configuration import (
    DATA_PATH,
    client_id,
    client_secret,
    rai_base_url,
    base_url,
    PROXIES,
    context_json_file_path,
    consolidated_ans_json_file_path,
    results_path_ext_cont
)

# Assuming load_questions is defined elsewhere
questions = load_questions("questions.txt")

def main():
    api = ECFRAPIWrapper(
        base_url=base_url,
        client_secret=client_secret,
        client_id=client_id,
        rai_base_url=rai_base_url,
        proxies=PROXIES
    )

    llm_processor = LLMProcessor(
        client_secret=client_secret,
        client_id=client_id,
        rai_base_url=rai_base_url
    )

    try:
        # [Context building remains the same]
        title = "12"
        date = "2025-03-06"
        subtitle = "A"
        chapter = "II"
        subchapter = "A"
        part = "211"
        subpart = "A"
        sections = [f"211.{i}" for i in range(1, 14)]

        context = {}
        os.makedirs(context_json_file_path, exist_ok=True)
        context_file_path = os.path.join(context_json_file_path, "context.json")

        if os.path.exists(context_file_path):
            logger.info("\n\nContext file already exists, loading context from file...\n\n")
            with open(context_file_path, "r") as f:
                context = json.load(f)
        else:
            logger.info("\n\nCONTEXT FILE NOT FOUND → STARTING FROM SCRATCH!!\n\n")
            for section in sections:
                filename = f"CONTEXT_ecfr_qa_{date}_{title}_{subtitle}_{chapter}_{subchapter}_{part}_{subpart}_{section}.txt"
                logger.info(f"\nBuilding context for section: {section}")
                result = api.get_structured_regulation(
                    date=date, title=title, subtitle=subtitle, chapter=chapter,
                    subchapter=subchapter, part=part, subpart=subpart, section=section
                )
                reg_contents_path = os.path.join(context_json_file_path, filename)
                write_regulation_to_file(result, reg_contents_path)
                context[section] = read_regulation_file(reg_contents_path)
            with open(context_file_path, "w") as f:
                json.dump(context, f)

        # [Question answering remains the same]
        answers = {question: [] for question in questions}
        for section, content in context.items():
            section_answers = llm_processor.answer_from_sections(content, questions)
            for i, answer in enumerate(section_answers):
                answers[questions[i]].append(answer)

        logger.info("\n\nSAVING FILES!!!\n\n")
        answers_file_path = os.path.join(results_path_ext_cont, "answers_dict.json")
        with open(answers_file_path, "w") as f:
            json.dump(answers, f)

        # Consolidation with debugging
        os.makedirs(consolidated_ans_json_file_path, exist_ok=True)
        consolidated_answers_path = os.path.join(consolidated_ans_json_file_path, "consolidated_data.json")

        # Reset the list explicitly
        consolidated_answers = []
        print(f"Starting consolidation. Initial consolidated_answers: {consolidated_answers}")
        
        with open(consolidated_answers_path, "w") as f:
            for i, (question, answers_nested) in enumerate(answers.items()):
                answers_list = [answer for answer in answers_nested]
                print(f"Processing question {i+1}: {question}")
                print(f"Answers list: {answers_list}")
                
                consolidated_answer = llm_processor.consolidator(question, answers_list)
                print(f"Consolidated answer: {consolidated_answer}")
                
                qa_block = {
                    "question": question,
                    "answer": consolidated_answer
                }
                print(f"qa_block: {qa_block}")
                
                consolidated_answers.append(qa_block)
                print(f"Current consolidated_answers: {consolidated_answers}")

            print(f"Final consolidated_answers before writing: {consolidated_answers}")
            json.dump(consolidated_answers, f, indent=4)
            print(f"Data written to {consolidated_answers_path}")

        logger.info(f"\n\n {consolidated_answers_path} SAVED!!! \n\n")

    except (ValueError, requests.exceptions.RequestException) as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()