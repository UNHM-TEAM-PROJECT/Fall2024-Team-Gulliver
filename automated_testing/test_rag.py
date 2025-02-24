"""
File: test_rag.py
Author: Sindhu Priya Itukulapati
Contributors: Nicholas Phenner
Date: 11-30-2024

"""

import os
import requests
import json
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Set the OpenAI API key as an environment variable

apikey = os.getenv("OPENAI_API_KEY")

# Predefined test cases for evaluating the chatbot
# Each test case includes a question, expected answer, and relevant retrieval context
test_cases = [
    {
        "question": "what majors are required to take internship courses?",
        "expected_answer": "Students from all computing majors are required to take internship courses. This includes undergraduate Computer Information Systems Major (CIS), Computer Science (CS) Major, and at the graduate level: M.S. Information Technology Major and M.S. Cybersecurity Engineering Major.",
        "retrieval_context": [
            "Internship courses are required for all computing major students. This includes undergraduate Computer Information Systems Major (CIS), Computer Science (CS) Major, and at the graduate level: M.S. Information Technology Major and M.S. Cybersecurity Engineering Major."
        ]
    },
    {
        "question": "what should I do to get an internship?",
        "expected_answer": "You can first start your search from UNH Handshake website. You can also apply for jobs directly through the website of the company or organization you are interested in, or through job websites such as LinkedIn, Indeed and Hired. Do attend internship fair on both Manchester and Durham campus, and speak with family, friends, and faculty. Career and Professional Success office can help you with resume writing, interview coaching and other career advice.",
        "retrieval_context": [
            "Students are encouraged to start their search on Handshake as most of the employers who posted their jobs are looking for UNH students. Students can also apply for jobs through the website of the company/organization they are interested in, or through job websites such as LinkedIn, Indeed and Hired. Do attend internship fair on both Manchester and Durham campus, and speak with family, friends, and faculty. Career and Professional Success office can help you with resume writing, interview coaching and other career advice."
        ]
    },
    {
        "question": "How can I register my internship experience on Handshake?",
        "expected_answer": " To register your internship experience on Handshake, please follow these steps: 1. Login to Handshake. 2. On the top right corner, under Career Center, go to Experiences. 3. Click on 'Request an Experience' and fill out the form. 4. Your internship experience must be approved by your site supervisor and your course instructor. 5. Make sure to include at least three well-developed learning objectives. If you have any questions related to registering your internship experience on Handshake, please contact the Career and Professional Success office.",
        "retrieval_context": [
            " To register your internship experience on Handshake, please follow these steps: 1. Login to Handshake. 2. On the top right corner, under Career Center, go to Experiences. 3. Click on 'Request an Experience' and fill out the form. 4. Your internship experience must be approved by your site supervisor and your course instructor. 5. Make sure to include at least three well-developed learning objectives. If you have any questions related to registering your internship experience on Handshake, please contact the Career and Professional Success office."
        ]
    },
    {
        "question": "what are the internship course options for undergrads?",
        "expected_answer": "If you are an undergraduate student, you need to take COMP690 Internship Experience. If you are currently working, you should take the course with the applied research option. If you are in your last semester, you may take the course with the team project option.",
        "retrieval_context": [
            "Internship Courses: Undergraduate students: • COMP690 Internship Experience The course has an applied research option for students who are currently working, and a team project option for students in their last semester of program."
        ]
    },
    {
        "question": "what are the internship course options for graduate students? ",
        "expected_answer": "If you are a graduate student, the internship course options include: - COMP890: Internship and Career Planning - COMP891: Internship Practice - COMP892: Applied Research Internship - COMP893: Team Project Internship",
        "retrieval_context": [
            "For graduate students, the internship course options include: - COMP890: Internship and Career Planning - COMP891: Internship Practice - COMP892: Applied Research Internship - COMP893: Team Project Internship"
        ]
    },
    {
        "question": "Tell me more about COMP690",
        "expected_answer": "COMP690 Internship Experience is for undergraduate students. It has an applied research option for students who are currently working, and a team project option for students in their last semester of program.",
        "retrieval_context": [
            "Undergraduate students: COMP690 Internship Experience has an applied research option for students who are currently working, and a team project option for students in their last semester of program."
        ]
    },
    {
        "question": "Tell me more about COMP890?",
        "expected_answer": "COMP890 Internship and Career Planning is for graduate students. You need to take this 1 cr course after the first semester to help you plan for the internship search process. It is offered in fall and spring semesters.",
        "retrieval_context": [
            "Graduate students: COMP890: Internship and Career Planning. This is a 1 cr course you need to take after the first semester to help you plan for the internship search process. The course is offered in fall and spring semesters."
        ]
    },
    {
        "question": "Tell me more about COMP891",
        "expected_answer": "COMP891 Internship Practice is for graduate students. This is a variable credit 1-3 crs course that you will take when you have an external internship. You will need to register in this course for at least 1 credit to apply for CPT. The course is oJered in both fall and spring semesters, as well as during the summer.",
        "retrieval_context": [
            "Graduate students: COMP891: Internship Practice. This is a variable credit 1-3 crs course that you will take when you have an external internship. You will need to register in this course for at least 1 credit to apply for CPT. The course is offered in both fall and spring semesters, as well as during the summer."
        ]
    },
    {
        "question": "Tell me more about COMP892",
        "expected_answer": "COMP 892: Applied Research Internship is for graduate students. This is a variable credit 1-3 crs course for students who are currently working full time or part time in the tech fields. The course is offered in both fall and spring semesters, as well as during the summer.",
        "retrieval_context": [
            "Graduate students: COMP 892: Applied Research Internship This is a variable credit 1-3 crs course for students who are currently working full time or part time in the tech fields. The course is offered in both fall and spring semesters, as well as during the summer."
        ]
    },
    {
        "question": "Tell me more about COMP893",
        "expected_answer": "COMP 893: Team Project Internship is for graduate students who are in their last semester of study and need to fulfill the internship requirements. You will work with other students on a collaborative project to gain practical skills and insights into the field of computing. The course is offered in fall and spring semesters.",
        "retrieval_context": [
            "Graduate students: COMP 893: Team Project Internship The course is for students who are in their last semester of study and need to fulfill the internship requirements. The COMP893 Team Project Internship course is designed for students who want to gain practical skills and insights into the field of computing by working on collaborative projects with external stakeholders. The course is offered in fall and spring semesters."
        ]
    },
    {
        "question": "When may I take the internship course COMP690?",
        "expected_answer": "You may take the internship experience course COMP690 any time you have an internship. Additionally, you can take the COMP690 applied research option if you have a part-time or full-time tech job. If you can't find an internship by the last semester of your program, you are allowed to take COMP690 with the group project option.",
        "retrieval_context": [
            "For undergraduate students, you may take the internship course COMP690 any time you have an internship. Additionally, you can take the COMP690 applied research option if you have a part-time or full-time tech job. If you can't find an internship by the last semester of your program, you are allowed to take COMP690 with the group project option."
        ]
    },
    {
        "question": "When may I take COMP890",
        "expected_answer": "You may take COMP890 Internship and Career Planning after you finish your first semester of study.",
        "retrieval_context": [
            "Graduate students: COMP890: Internship and Career Planning. This is a 1 cr course you need to take after the first semester to help you plan for the internship search process. The course is offered in fall and spring semesters."
        ]
    },
    {
        "question": "When may I take COMP891",
        "expected_answer": "You may take COMP891 Internship Practice when you have found an external internship. The course is offered in all semesters year around.",
        "retrieval_context": [
            "Graduate students: COMP891: Internship Practice. This is a variable credit 1-3 crs course that you will take when you have an external internship. You will need to register in this course for at least 1 credit to apply for CPT. The course is offered in both fall and spring semesters, as well as during the summer."
        ]
    },
    {
        "question": "When may I take COMP892",
        "expected_answer": "You may take COMP892 Applied Research Internship if you are currently working full time or part time in the tech fields. The course is offered in all semesters year around.",
        "retrieval_context": [
            "Graduate students: COMP 892: Applied Research Internship This is a variable credit 1-3 crs course for students who are currently working full time or part time in the tech fields. The course is offered in both fall and spring semesters, as well as during the summer."
        ]
    },
    {
        "question": "When may I take COMP893",
        "expected_answer": "You may take COMP893 Team Project Internship in your last semester of study and need to fulfill the internship requirement. The course is offered in fall and spring semesters.",
        "retrieval_context": [
            "Graduate students: COMP 893: Team Project Internship The course is for students who are in their last semester of study and need to fulfill the internship requirements. The COMP893 Team Project Internship course is designed for students who want to gain practical skills and insights into the field of computing by working on collaborative projects with external stakeholders. The course is offered in fall and spring semesters."
        ]
    },
    {
        "question": "What is the course name of COMP893?",
        "expected_answer": " The course name of COMP893 is 'Team Project Internship.'",
        "retrieval_context": [
            "The course information section of the Fall 2024 semester course syllabus states that the name of COMP 893 is Team Project Internship."
        ]
    },
    {
        "question": "What is the course name of COMP690?",
        "expected_answer": "The course name of COMP690 is 'Internship Experience'.",
        "retrieval_context": [
            "The course information section of the Fall 2024 semester course syllabus states that the name of COMP 690 is Internship Experience."
        ]
    },
    {
        "question": "What room is COMP893 in?",
        "expected_answer": "The classroom of COMP893 is in Room P142 in Fall 2024 semester",
        "retrieval_context": [
            " In Fall 2024 semester, COMP893 is located in Room P142"
        ]
    },
    {
        "question": "What room is COMP690 in?",
        "expected_answer": "The classroom of COMP690 is in Room P142 in Fall 2024 semester",
        "retrieval_context": [
            " In Fall 2024 semester, COMP690 is located in Room P142"
        ]
    },
    {
        "question": "What time is COMP893?",
        "expected_answer": " COMP893 has two sections in Fall 2024: M1 Section meets on Wednesday 9:10am-12pm and M2 Section meets on Wednesday 1:10-4pm",
        "retrieval_context": [
            " COMP893 has two sections: M1 Section meets on Wednesday 9:10am-12pm and M2 Section meets on Wednesday 1:10-4pm"
        ]
    },
    {
        "question": "What time is COMP690?",
        "expected_answer": " COMP690 has two sections in Fall 2024: M2 Section meets on Wednesday 9:10am-12pm and M2 Section meets on Wednesday 1:10-4pm",
        "retrieval_context": [
            "COMP690 has two sections: M2 Section meets on Wednesday 9:10am-12pm and M3 Section meets on Wednesday 1:10-4pm"
        ]
    },
    {
        "question": "Who is the instructor of COMP893?",
        "expected_answer": " The instructor of COMP893 Team Project Internship is Professor Karen Jin",
        "retrieval_context": [
            " Professor Karen Jin is from the Department of Applied Engineering and Sciences. She is the instructor for COMP893 and COMP690. She is also the computing program internship coordinator"
        ]
    },
    {
        "question": "Who is the instructor of COMP690?",
        "expected_answer": "The instructor of COMP690 Internship Experience is Professor Karen Jin",
        "retrieval_context": [
            " Professor Karen Jin is from the Department of Applied Engineering and Sciences. She is the instructor for COMP893 and COMP690. She is also the computing program internship coordinator"
        ]
    },
    {
        "question": "What is Karen Jin's role?",
        "expected_answer": " Karen Jin teaches COMP893 and COMP690. She is also the internship coordinator for the computing programs.",
        "retrieval_context": [
            " Professor Karen Jin is from the Department of Applied Engineering and Sciences. She is the instructor for COMP893 and COMP690. She is also the computing program internship coordinator"
        ]
    },
    {
        "question": "How to contact Karen Jin?",
        "expected_answer": "You can contact her by email Karen.Jin@unh.edu. Her office is located in Rm139, Pandora Mill building.",
        "retrieval_context": [
            " Prof. Karen Jin's office is located in Rm139, Pandora Mill building. Her email is Karen.Jin@unh.edu"
        ]
    },
    {
        "question": "What are the instructor's office hours?",
        "expected_answer": " Karen Jin's office hours are Monday 1-4pm and Friday 9-noon. You can also make an appointment with her to meet in person or over zoom.",
        "retrieval_context": [
            " Karen Jin's office hours are Monday 1-4pm and Friday 9-noon. She is available in person or over Zoom, and appointments can be made via email"
        ]
    },
    {
        "question": "How do you make appointments with Karen Jin?",
        "expected_answer": "You should email her directly and arrange these meetings in advance. Her email is Karen.Jin@unh.edu.",
        "retrieval_context": [
            " Email directly the instructor or internship coordinator Karen Jin to make an appointment. It's important to arrange these meetings in advance and provide a clear reason for the meeting. She is available in person or over Zoom, and appointments can be made via email. Her email is Karen.Jin@unh.edu"
        ]
    },
    {
        "question": "What is the course description for COMP893?",
        "expected_answer": " The course description for COMP893 Team Project Internship is as follows: 'The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting.'",
        "retrieval_context": [
            " The course description of COMP893 is stated as 'The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting.'"
        ]
    },
    {
        "question": "What is the course description for COMP690?",
        "expected_answer": "The course description for COMP690 Internship Experience is as follows: 'The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting.'",
        "retrieval_context": [
            "The course description for COMP690 Internship Experience is as follows: 'The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting.'"
        ]
    },
    {
        "question": "Student learning outcome for COMP893?",
        "expected_answer": "The student learning outcomes for COMP893 Team Project Internship are as follows: 1. Analyze complex computing problems and identify solutions by applying principles of computing. 2. Design, implement, and evaluate computing solutions that meet IT computing requirements. 3. Communicate effectively in a variety of professional contexts. 4. Function effectively as a member or leader of a team engaged in IT activities. 5. Identify and analyze user needs in the process of developing and operating computing systems.",
        "retrieval_context": [
            "The student learning outcomes for COMP893 Team Project Internship are as follows: 1. Analyze complex computing problems and identify solutions by applying principles of computing. 2. Design, implement, and evaluate computing solutions that meet IT computing requirements. 3. Communicate effectively in a variety of professional contexts. 4. Function effectively as a member or leader of a team engaged in IT activities. 5. Identify and analyze user needs in the process of developing and operating computing systems."
        ]
    },
    {
        "question": "Student learning outcome for COMP690?",
        "expected_answer": "The student learning outcomes for COMP690 Internship Experience are as follows: 1. Analyze complex computing problems and identify solutions by applying principles of computing. 2. Design, implement, and evaluate computing solutions that meet IT computing requirements. 3. Communicate effectively in a variety of professional contexts. 4. Function effectively as a member or leader of a team engaged in IT activities. 5. Identify and analyze user needs in the process of developing and operating computing systems.",
        "retrieval_context": [
            "The student learning outcomes for COMP690 Internship Experience are as follows: 1. Analyze complex computing problems and identify solutions by applying principles of computing. 2. Design, implement, and evaluate computing solutions that meet IT computing requirements. 3. Communicate effectively in a variety of professional contexts. 4. Function effectively as a member or leader of a team engaged in IT activities. 5. Identify and analyze user needs in the process of developing and operating computing systems."
        ]
    },
    {
        "question": "How much of the grade is class attendance in COMP893?",
        "expected_answer": " In Fall 2024 semester, 10% of the grade is based on class attendance",
        "retrieval_context": [
            " The Fall 2024 semester course syllabus states that 10% of the grade is based on class attendance"
        ]
    },
    {
        "question": "What components does the final grade consist of in COMP893?",
        "expected_answer": " The final grade in Fall 2024 semester consists of four components: 10% Class Attendance of all required meetings. 60% Sprint Grade. 10% Homework and 20% Final Project Report",
        "retrieval_context": [
            " The Fall 2024 semester course syllabus states that the final grade consists of four components: 10% Class Attendance of all required meetings. 60% Sprint Grade. 10% Homework and 20% Final Project Report"
        ]
    },
    {
        "question": "How is sprint grade calculated?",
        "expected_answer": " The sprint grade in Fall 2024 semester is calculated as the Teamwork Grade multiplied by the Sprint Grade. The Teamwork Grade is based on peer evaluation for each of the three sprints, and the Sprint Grade is based on the technical aspect of the product and team project management",
        "retrieval_context": [
            " The Fall 2024 semester course syllabus states that The sprint grade in Fall 2024 semester is calculated as the Teamwork Grade multiplied by the sprint Grade. The Teamwork Grade is based on peer evaluation for each of the three sprints, and the Sprint Grade is based on the technical aspect of the product and team project management"
        ]
    },
    {
        "question": "What is the Credit Hour Workload Estimate?",
        "expected_answer": " The Credit Hour Workload Estimate for COMP893 and COMP690 is a minimum of 45 hours of student academic work per credit per term",
        "retrieval_context": [
            " The Credit Hour Workload Estimate for COMP893 and COMP690 is a minimum of 45 hours of student academic work per credit per term"
        ]
    },
    {
        "question": "What are the attendance policies in COMP893 and COMP690?",
        "expected_answer": " The attendance policy for COMP893 and COMP690 states that students are responsible for attending scheduled meetings and are expected to abide by the University Policy on Attendance. If a student cannot attend a scheduled meeting, they must email the instructor about the circumstances and request to be excused BEFORE the class meeting. Additionally, students need to arrange a meeting with the instructor individually to update their internship progress",
        "retrieval_context": [
            " The attendance policy for COMP893 and COMP690 states that students are responsible for attending scheduled meetings and are expected to abide by the University Policy on Attendance. If a student cannot attend a scheduled meeting, they must email the instructor about the circumstances and request to be excused BEFORE the class meeting. Additionally, students need to arrange a meeting with the instructor individually to update their internship progress"
        ]
    },
    {
        "question": "What do you do if you think you'll miss a meeting?",
        "expected_answer": " If you anticipate missing a meeting, you should email the instructor about the circumstances and request to be excused for the meeting BEFORE the class meeting. It is important to communicate in advance and provide a valid reason for your absence",
        "retrieval_context": [
            " The course syllabus of COMP893 and COMP690 states that a student needs to email the instructor about the circumstances and request to be excused for the meeting BEFORE the class meeting. It is important to communicate in advance and provide a valid reason for any missing meetings."
        ]
    },
    {
        "question": "What is the policy on late submissions?",
        "expected_answer": " A late submission may be granted only if you email prior to the deadline and explains and provides evidence for the circumstances that prevent you from meeting the submission requirement",
        "retrieval_context": [
            " The policy for late submissions in COMP893 and COMP690 is very strict and applies only in exceptional cases of student illness, accidents, or emergencies that are properly documented. A late submission may be granted only if the student emails prior to the deadline and explains and provides evidence for the circumstances that have prevented them from meeting the class requirement. Failing to comply with these rules may result in no credit for the assignment."
        ]
    },
    {
        "question": "Do I still need to take the course if I am currently working?",
        "expected_answer": "Yes, you do. Even if you are currently working or have worked in the past, you will still need to take the Internship Experience course as a degree requirement. However, you don't need to take another internship position if you are currently working in the field. You may use the applied research option of COMP690, or COMP892 to fulfill the internship requirements.",
        "retrieval_context": [
            "Yes, you do. Even if you are currently working or have worked in the past, you will still need to take the Internship Experience course as a degree requirement. However, you don't need to take another internship position if you are currently working in the field. You may use the applied research option of COMP690 for undergrad students, or COMP892 for graduate students to fulfill the internship requirements."
        ]
    },
    {
        "question": "I did an internship last summer. Can I use that to cover the internship requirements?",
        "expected_answer": "No, a past internship completed cannot be used to fulfill the requirements for the course. The internship position and required hours must be completed while you are registered in the Internship Experience course.",
        "retrieval_context": [
            "No, a past internship completed cannot be used to fulfill the requirements for the course. The internship position and required hours must be completed while you are registered in the Internship Experience course."
        ]
    },
    {
        "question": "How to register for Internship Courses?",
        "expected_answer": "You need to get permission from the internship course instructor. For more details, students may contact Prof. Karen Jin, the internship coordinator.",
        "retrieval_context": [
            "Students can register for internship courses by obtaining the instructor's permission. They need to email the faculty internship coordinator or the course instructor to register for the course. For more details, students may contact Prof. Karen Jin the internship coordinator."
        ]
    },
    {
        "question": "What requirements do you need to fulfill to earn the credit?",
        "expected_answer": "To earn credit for the internship course, you need to fulfill the following requirements: - Attend every scheduled class meeting - Submit weekly logs - Complete a final internship report - Give progress presentations during the class Additionally, you must meet the specific course syllabus requirements and complete the necessary hours at your internship based on your enrolled credit hours.",
        "retrieval_context": [
            "To earn credit for the internship course, you need to fulfill the following requirements: - Attend every scheduled class meeting - Submit weekly logs - Complete a final internship report - Give progress presentations during the class Additionally, you must meet the specific course syllabus requirements and complete the necessary hours at your internship based on your enrolled credit hours."
        ]
    },
    {
        "question": "What are the steps for registering your internship experience on handshake?",
        "expected_answer": "To register your internship experience on Handshake, follow these steps: 1. Login to Handshake. 2. On the top right corner, under Career Center, go to Experiences. 3. Click on 'Request an Experience' and fill out the form. 4. Ensure that your internship experience is approved by both your site supervisor and your course instructor.",
        "retrieval_context": [
            " To register your internship experience on Handshake, follow these steps: 1. Login to Handshake. 2. On the top right corner, under Career Center, go to Experiences. 3. Click on 'Request an Experience' and fill out the form. 4. Ensure that your internship experience is approved by both your site supervisor and your course instructor."
        ]
    },
    {
        "question": "Do I need to write weekly logs every week?",
        "expected_answer": "Yes, you need to write weekly logs every week during your internship until you complete the required hours for the credit. For undergraduate students, it's 150 hours for 4 credits of COMP690. For graduate students, the credit hour is roughly equal to 40 hours of internship work based on the number of credit hours you are enrolled in. After reaching the total hours required, it's recommended to continue with the weekly logs, but you don't need to submit logs for weeks you have not worked, like during a break.",
        "retrieval_context": [
            "Yes, you need to write weekly logs every week during your internship until you complete the required hours for the credit. For undergraduate students, it's 150 hours for 4 credits of COMP690. For graduate students, the credit hour is roughly equal to 40 hours of internship work based on the number of credit hours you are enrolled in. After reaching the total hours required, it's recommended to continue with the weekly logs, but you don't need to submit logs for weeks you have not worked, like during a break."
        ]
    },
    {
        "question": "How many hours do I need to log?",
        "expected_answer": "You need to log 150 hours if you are taking COMP690. For graduate students, a credit hour is equal to 40 hours of internship work. For example, if you are enrolled in 3 credit hours of the Internship Experience class, then you must complete 120 hours of internship work.",
        "retrieval_context": [
            "For undergraduate students, you need to complete 150 hours for 4 credits of COMP690. For graduate students, a credit hour is roughly to 40 hours of internship work. For example, if you are enrolled in 3 credit hours of the Internship Experience class, then you must complete 120 hours of internship work."
        ]
    },
    {
        "question": "Can I start my internship position before the Internship Experience course starts?",
        "expected_answer": "Yes, it is permissible to start your internship position before the Internship Experience course starts. However, you can only count up to 20% of the total internship hours required towards the course if you complete the remaining hours during the same semester.",
        "retrieval_context": [
            "It is permissible to start your internship position before the Internship Experience course starts. However, you can only count up to 20% of the total internship hours required towards the course if you complete the remaining hours during the same semester."
        ]
    },
    {
        "question": "How many hours I can count if I start my internship work before starting the course?",
        "expected_answer": "you can only count up to 20% of the total internship hours required towards the course if you complete the remaining hours during the same semester.",
        "retrieval_context": [
            "It is permissible to start your internship position before the Internship Experience course starts. However, you can only count up to 20% of the total internship hours required towards the course if you complete the remaining hours during the same semester."
        ]
    },
    {
        "question": "I just got an internship offer but the semester has already started, what should I do?",
        "expected_answer": "If you receive an internship offer after the semester has already started, you should contact the faculty internship coordinator, Professor Karen Jin, and inform her of the situation. Depending on the timing of your offer, you may be allowed to late add into the internship course or arrange with the employer for a later start date.",
        "retrieval_context": [
            "If you receive an internship offer after the semester has already started, you should contact the faculty internship coordinator, Professor Karen Jin, and inform her of the situation. Depending on the timing of your offer, you may be allowed to late add into the internship course or arrange with the employer for a later start date."
        ]
    },
    {
        "question": "How do I contact CaPS office.",
        "expected_answer": "CaPS office has a website is https://manchester.unh.edu/careers/career-professional-success. You can also reach them by email unhm.career@unh.edu and by phone (603) 641-4394",
        "retrieval_context": [
            "The website for the CaPS office is https://manchester.unh.edu/careers/career-professional-success. Phone: (603) 641-4394 Email: unhm.career@unh.edu"
        ]
    },
    {
        "question": "what is the oiss' website?",
        "expected_answer": "The website for the Office of International Students and Scholars (OISS) is https://www.unh.edu/global/international-students-scholars. You may also email them at oiss@unh.edu",
        "retrieval_context": [
            "The website for the Office of International Students and Scholars (OISS) is https://www.unh.edu/global/international-students-scholars Their email is oiss@unh.edu"
        ]
    }
    
]
# Function to send a question to the chatbot and retrieve its response
def get_chatbot_response(question):
    """
    Sends a question to the chatbot and retrieves the response.
    Args:
        question (str): The question to ask the chatbot.
    Returns:
        str: The chatbot's response or None if an error occurs.
    """
    response = requests.post('http://127.0.0.1:80/ask', json={"message": question})
    if response.status_code == 200:
        return response.json().get("response", "")
    return None

# Function to evaluate a single test case
def evaluate_test_case(test_case):
    """
    Evaluates a chatbot response against a test case using relevancy and faithfulness metrics.
    Args:
        test_case (dict): A dictionary containing the question, expected answer, and retrieval context.
    Returns:
        dict: Evaluation results including relevancy and faithfulness scores.
    """
    actual_output = get_chatbot_response(test_case["question"])
    if not actual_output:
        print(f"Error fetching chatbot response for question: {test_case['question']}")
        return None

    # Create a test case object for evaluation
    llm_test_case = LLMTestCase(
        input=test_case["question"],
        actual_output=actual_output,
        expected_output=test_case["expected_answer"],
        retrieval_context=test_case["retrieval_context"]
    )

    # Evaluate relevancy and faithfulness of the chatbot response
    relevancy_metric = AnswerRelevancyMetric(threshold=0.9, model="gpt-4")
    faithfulness_metric = FaithfulnessMetric(threshold=0.9, model="gpt-4")

    relevancy_metric.measure(llm_test_case)
    faithfulness_metric.measure(llm_test_case)

    # Compile results
    result = {
        "question": test_case["question"],
        "actual_answer": actual_output,
        "expected_answer": test_case["expected_answer"],
        "relevancy_score": relevancy_metric.score,
        "faithfulness_score": faithfulness_metric.score,
        "relevancy_reason": relevancy_metric.reason,
        "faithfulness_reason": faithfulness_metric.reason,
        "retrieval_context": test_case["retrieval_context"]
    }

    # Update the test case with the actual chatbot output
    test_case["actual_answer"] = actual_output
    test_case["relevancy_reason"] = relevancy_metric.reason
    test_case["faithfulness_reason"] = faithfulness_metric.reason
    return result

# Function to evaluate all test cases and save results
def run_tests():
    """
    Runs all test cases, evaluates chatbot performance, and saves results to JSON files.
    """
    results = []
    updated_test_cases = []

    # Evaluate each test case
    for test_case in test_cases:
        result = evaluate_test_case(test_case)
        if result:
            results.append(result)
            updated_test_cases.append(test_case)

    # Save evaluation results to a JSON file
    with open("evaluation_results.json", "w") as results_file:
        json.dump(results, results_file, indent=4)

    # Save updated test cases (with actual outputs) to another JSON file
    with open("updated_test_cases.json", "w") as test_cases_file:
        json.dump(updated_test_cases, test_cases_file, indent=4)

    print("\nEvaluation completed. Results saved to 'evaluation_results.json'.")
    print("Updated test cases saved to 'updated_test_cases.json'.")

# Entry point for the script
if __name__ == "__main__":
    run_tests()