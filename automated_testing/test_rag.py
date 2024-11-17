import os
import requests
import json
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"

# Test cases with extended retrieval contexts as lists of strings
test_cases = [
    {
        "question": "What room is COMP690 in?",
        "expected_answer": "Room P142",
        "retrieval_context": [
            "Name: COMP690 Internship Experience",
            "Credits: 4",
            "Term: Fall 2024",
            "Location: Rm P142",
            "Time: M2 Section: Wednesday 9:10am-12pm",
            "M2 Section: Wednesday 1:10-4pm",
            "Instructor Information",
            "Name: Karen Jin, Associate Professor, Department of Applied Engineering and Sciences",
            "Office: Rm139, Pandora Mill building",
            "Zoom: https://unh.zoom.us/j/4858446046",
            "Email: karen.jin@unh.edu",
            "Office Hours: Monday 1-4pm and Friday 9-noon. Available in person or over Zoom."
        ]
    },
    {
        "question": "What is the instructor's email?",
        "expected_answer": "karen.jin@unh.edu",
        "retrieval_context": [
            "Instructor Information",
            "Name: Karen Jin, Associate Professor, Department of Applied Engineering and Sciences",
            "Office: Rm139, Pandora Mill building",
            "Zoom: https://unh.zoom.us/j/4858446046",
            "Email: karen.jin@unh.edu",
            "Office Hours: Monday 1-4pm and Friday 9-noon.",
            "CaPS office: Website: https://manchester.unh.edu/careers/career-professional-success",
            "Phone: (603) 641-4394",
            "Email: unhm.career@unh.edu"
        ]
    },
    {
        "question": "What time is COMP893?",
        "expected_answer": "COMP893 has two sections:\n- M1 Section: Wednesday 9:10am-12pm\n- M2 Section: Wednesday 1:10-4pm",
        "retrieval_context": [
            "Name: COMP893 Team Project Internship",
            "Credits: 1-3",
            "Term: Fall 2024",
            "Location: Rm P142",
            "Time: M1 Section: Wednesday 9:10am-12pm",
            "M2 Section: Wednesday 1:10-4pm",
            "Instructor Information",
            "Name: Karen Jin, Associate Professor, Department of Applied Engineering and Sciences",
            "Office: Rm139, Pandora Mill building",
            "Zoom: https://unh.zoom.us/j/4858446046",
            "Email: karen.jin@unh.edu",
            "Office Hours: Monday 1-4pm and Friday 9-noon."
        ]
    },
    {
        "question": "When is week 1?",
        "expected_answer": "Week 1 starts on 8/28",
        "retrieval_context": [
            "Week 1: 8/28",
            "Class Introduction / Development Team (DT) Setup",
            "Intro to Project Management",
            "Intro to Scrum workflow",
            "Project Goal",
            "Week 2: 9/4",
            "Project Kickoff",
            "Environment Setup: Jira",
            "Create Project backlog",
            "Create user stories, tasks, and bugs"
        ]
    },
    {
        "question": "How much of the grade is class attendance?",
        "expected_answer": "10%.",
        "retrieval_context": [
            "Your final grade consists of the following three components:",
            "10% Class Attendance of all required meetings",
            "60% Sprint Grade is calculated as: Teamwork Grade * Sprint Grade",
            "Teamwork Grade is based on peer evaluation for each of the three sprints.",
            "Sprint Grades: You will receive a team grade for each of the three sprints, based on the technical aspect of the product and team project management.",
            "10% Homework: additional homework in project management and development tools.",
            "20% Final Project Report: See Appendix A for the report format."
        ]
    },
    {
        "question": "How is sprint grade calculated?",
        "expected_answer": "The sprint grade is calculated as the Teamwork Grade multiplied by the Sprint Grade.",
        "retrieval_context": [
            "Your final grade consists of the following three components:",
            "10% Class Attendance of all required meetings",
            "60% Sprint Grade is calculated as: Teamwork Grade * Sprint Grade",
            "Teamwork Grade is based on peer evaluation for each of the three sprints.",
            "Sprint Grades: You will receive a team grade for each of the three sprints, based on the technical aspect of the product and team project management."
        ]
    },
    {
        "question": "CaPS office website?",
        "expected_answer": "The website for the CaPS office is [Career and Professional Success](https://manchester.unh.edu/careers/career-professional-success).",
        "retrieval_context": [
            "CaPS office:",
            "Website: https://manchester.unh.edu/careers/career-professional-success",
            "Phone: (603) 641-4394",
            "Email: unhm.career@unh.edu",
            "The Office of International Students and Scholars (OISS):",
            "Website: https://www.unh.edu/global/international-students-scholars",
            "Email: oiss@unh.edu"
        ]
    },
    {
        "question": "How many credits is COMP890?",
        "expected_answer": "COMP890 is a 1-credit course in the UNH internship program.",
        "retrieval_context": [
            "Graduate students:",
            "COMP890: Internship and Career Planning.",
            "This is a 1 cr course you need to take after the first semester to help you plan for the internship search process.",
            "The course is offered in fall and spring semesters.",
            "COMP891: Internship Practice.",
            "This is a variable credit 1-3 crs course that you will take when you have an external internship."
        ]
    },
    {
        "question": "When is COMP890 offered?",
        "expected_answer": "COMP890 is offered in both the fall and spring semesters.",
        "retrieval_context": [
            "COMP890: Internship and Career Planning.",
            "This is a 1 cr course you need to take after the first semester to help you plan for the internship search process.",
            "The course is offered in fall and spring semesters."
        ]
    },
    {
        "question": "Who do you need to email for internship courses?",
        "expected_answer": "To register for internship courses, you need to email the Internship Advisor, Karen Jin.",
        "retrieval_context": [
            "All internship courses require instructor’s permission.",
            "You will need to email the faculty internship coordinator, or the course instructor to register you for the course.",
            "For more details, you may email Prof. Karen Jin.",
            "How to register for Internship Courses?",
            "Email the Internship Advisor, Karen Jin (karen.jin@unh.edu).",
            "Provide details about your internship or project requirements.",
            "Ensure to include your student ID in the email."
        ]
    }
]

# Remaining evaluation functions stay the same (from the previous script)
# Ensure that 'evaluate_test_case' and 'run_tests' are updated to handle the test cases

def get_chatbot_response(question):
    response = requests.post('http://127.0.0.1:5000/ask', json={"message": question})
    if response.status_code == 200:
        return response.json().get("response", "")
    return None

def evaluate_test_case(test_case):
    actual_output = get_chatbot_response(test_case["question"])
    if not actual_output:
        print(f"Error fetching chatbot response for question: {test_case['question']}")
        return None

    llm_test_case = LLMTestCase(
        input=test_case["question"],
        actual_output=actual_output,
        expected_output=test_case["expected_answer"],
        retrieval_context=test_case["retrieval_context"]
    )

    relevancy_metric = AnswerRelevancyMetric(threshold=0.9, model="gpt-4")
    faithfulness_metric = FaithfulnessMetric(threshold=0.9, model="gpt-4")

    relevancy_metric.measure(llm_test_case)
    faithfulness_metric.measure(llm_test_case)

    result = {
        "question": test_case["question"],
        "actual_answer": actual_output,
        "expected_answer": test_case["expected_answer"],
        "relevancy_score": relevancy_metric.score,
        "faithfulness_score": faithfulness_metric.score,
        "retrieval_context": test_case["retrieval_context"]
    }

    # Update the test case with the actual output
    test_case["actual_answer"] = actual_output

    return result

def run_tests():
    results = []
    updated_test_cases = []

    for test_case in test_cases:
        result = evaluate_test_case(test_case)
        if result:
            results.append(result)
            updated_test_cases.append(test_case)

    
    with open("evaluation_results.json", "w") as results_file:
        json.dump(results, results_file, indent=4)

    # Save updated test cases to another JSON file
    with open("updated_test_cases.json", "w") as test_cases_file:
        json.dump(updated_test_cases, test_cases_file, indent=4)

    print("\nEvaluation completed. Results saved to 'evaluation_results.json'.")
    print("Updated test cases saved to 'updated_test_cases.json'.")

if __name__ == "__main__":
    run_tests()
