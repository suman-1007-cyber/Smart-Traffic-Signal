# Smart Traffic Signal Timing Controller

## Academic Internal Assessment Project

---

## 1. Project Title

**Smart Traffic Signal Timing Controller Using Artificial Intelligence and Fuzzy Logic**

---

## 2. Abstract

The Smart Traffic Signal Timing Controller is an AI-assisted traffic management system designed to calculate an appropriate green-light duration based on traffic conditions.

The system accepts a traffic scenario written in natural language, such as:

> "There are many vehicles waiting at the signal for several minutes."

An AI language model analyzes the scenario and extracts important traffic parameters such as vehicle density and waiting time.

These parameters are then provided to a Fuzzy Logic Controller. The fuzzy inference system processes the traffic conditions using predefined membership functions and fuzzy rules and calculates the final green-light duration.

The system combines Natural Language Processing, Artificial Intelligence, and Fuzzy Logic to provide an understandable and flexible traffic signal timing solution.

The final result is displayed through an interactive Streamlit web interface.

---

# 3. Introduction

Traffic congestion is a common problem in urban areas. Traditional traffic signals generally operate using fixed timings. Fixed timings do not always respond effectively to changing traffic conditions.

For example, a signal may provide the same green-light duration whether there are only a few vehicles or a large number of vehicles waiting.

The proposed system provides a smarter approach by analyzing traffic conditions and dynamically calculating an appropriate green-light duration.

The system uses two major intelligent components:

1. **AI Language Model** – understands the traffic scenario written in natural language.
2. **Fuzzy Logic Controller** – calculates the green-light duration based on traffic conditions.

This separation makes the system easier to understand, test, and modify.

---

# 4. Problem Statement

Traditional traffic signal systems often use predefined fixed timings.

Such systems may not efficiently handle situations where:

- Traffic density changes rapidly.
- Vehicles are waiting for a long time.
- Traffic conditions are different at different times.
- Manual adjustment of signal timing is required.

Therefore, there is a need for a system that can analyze traffic conditions and dynamically calculate an appropriate green-light duration.

---

# 5. Aim

To develop an AI-assisted Smart Traffic Signal Timing Controller that analyzes traffic conditions and calculates an appropriate green-light duration using Fuzzy Logic.

---

# 6. Objectives

The main objectives of the project are:

- Accept traffic conditions in natural language.
- Use an AI model to understand the traffic scenario.
- Extract vehicle density and waiting time.
- Convert traffic values into fuzzy membership values.
- Apply predefined fuzzy rules.
- Calculate green-light duration using fuzzy inference.
- Provide manual override functionality.
- Display results through an interactive web interface.
- Provide an understandable explanation of the calculated result.

---

# 7. Scope of the Project

The project can be used as a prototype for intelligent traffic signal management.

The current system focuses on:

- Vehicle density.
- Waiting time.
- Dynamic green-light duration.
- Natural-language traffic descriptions.
- Fuzzy logic based decision making.
- AI-assisted traffic analysis.

The system can later be extended to work with real-world traffic sensors, cameras, IoT devices, and live traffic data.

---

# 8. Technologies Used

## Programming Language

**Python**

Python is used to implement the complete application and intelligent processing logic.

## User Interface

**Streamlit**

Streamlit is used to create the interactive web-based user interface.

## Artificial Intelligence

**Groq API with LangChain**

The Groq API provides the language model used to understand natural-language traffic scenarios.

The model used in the project is:

`openai/gpt-oss-120b`

## Fuzzy Logic

**scikit-fuzzy**

The scikit-fuzzy library is used to implement fuzzy membership functions, fuzzy rules, inference, and defuzzification.

## Numerical Processing

**NumPy**

NumPy is used for numerical calculations and generation of fuzzy universes.

## Environment Configuration

**python-dotenv**

python-dotenv is used to load the Groq API key from the `.env` file.

---

# 9. System Architecture

The system follows the following processing pipeline:

```text
User Traffic Scenario
        |
        v
Natural Language Input
        |
        v
AI / Groq Language Model
        |
        v
Traffic Parameter Extraction
        |
        +----------------------+
        |                      |
        v                      v
Vehicle Density          Waiting Time
        |                      |
        +----------+-----------+
                   |
                   v
          Fuzzy Logic Controller
                   |
                   v
             Fuzzification
                   |
                   v
              Fuzzy Rules
                   |
                   v
          Fuzzy Inference
                   |
                   v
            Defuzzification
                   |
                   v
       Green-Light Duration
                   |
                   v
            Streamlit UI

10. System Workflow

The system works in the following steps.

Step 1: User Input

The user enters a traffic scenario in natural language.

Example:

"There are many vehicles at the signal and they have been waiting for a long time."

Step 2: AI Analysis

The AI model analyzes the natural-language scenario.

It extracts:

Vehicle density.
Waiting time.

Both values are converted into numerical values ranging from 0 to 10.

Step 3: Fuzzification

The numerical values are converted into fuzzy membership values.

Vehicle density is classified into:

Low
Medium
High

Waiting time is classified into:

Short
Medium
Long
Step 4: Rule Evaluation

The fuzzy rule base evaluates the traffic conditions.

For example:

IF vehicle density is HIGH
AND waiting time is LONG
THEN green duration is LONG
Step 5: Fuzzy Inference

The system combines the activated fuzzy rules to determine the output fuzzy set.

Step 6: Defuzzification

The fuzzy output is converted into a single numerical value using the centroid method.

Step 7: Final Result

The system displays the recommended green-light duration in seconds.

The current system limits the result between:

10 and 90 seconds

11. Fuzzy Logic Controller

Fuzzy Logic is used because traffic conditions are not always strictly defined.

For example, a traffic density of 6 vehicles-per-defined-scale may not simply be classified as "high". It can have partial membership in both medium and high categories.

Fuzzy logic allows the system to represent such gradual transitions.

12. Input Variables

The system uses two input variables.

12.1 Vehicle Density

Range:

0–10

Membership categories:

Low
Medium
High
Low
[0, 0, 5]
Medium
[2, 5, 8]
High
[5, 10, 10]
12.2 Waiting Time

Range:

0–10 minutes

Membership categories:

Short
Medium
Long
Short
[0, 0, 4]
Medium
[2, 5, 8]
Long
[5, 10, 10]
13. Output Variable

The output variable is:

Green-Light Duration

Range:

10–90 seconds

Membership categories:

Short
Medium
Long
Short
[10, 10, 40]
Medium
[25, 50, 75]
Long
[60, 90, 90]
14. Fuzzy Rule Base

The system uses nine fuzzy rules.

Rule	Vehicle Density	Waiting Time	Green Duration
R1	Low	Short	Short
R2	Low	Medium	Medium
R3	Low	Long	Long
R4	Medium	Short	Medium
R5	Medium	Medium	Medium
R6	Medium	Long	Long
R7	High	Short	Long
R8	High	Medium	Long
R9	High	Long	Long

These rules represent the decision-making logic of the traffic controller.

15. Mamdani Fuzzy Inference

The project uses Mamdani-style fuzzy inference.

For each rule, the degree of activation is calculated using the minimum operator for the AND condition.

For example:

IF density is HIGH
AND waiting time is LONG
THEN green duration is LONG

The rule activation can be represented as:

Activation = min(Density Membership, Waiting Membership)

The activated output fuzzy sets are then combined to produce the final fuzzy output.

16. Defuzzification

After fuzzy inference, the system needs to convert the fuzzy output into a single numerical value.

The project uses the:

Centroid Defuzzification Method

The centroid method calculates the center of gravity of the resulting fuzzy output.

Conceptually:

             Σ(x × μ(x))
Output = --------------------
                Σ μ(x)

Where:

x = possible green-light duration.
μ(x) = membership value at that duration.

The result is then limited to the valid range of:

10–90 seconds

17. Role of Artificial Intelligence

The AI model is responsible for understanding natural-language traffic descriptions.

For example, a user can enter:

Heavy traffic with vehicles waiting for many minutes.

Instead of requiring the user to manually enter numerical values, the AI model extracts the relevant parameters.

Example output:

Vehicle Density: 9
Waiting Time: 8

The extracted values are then passed to the fuzzy controller.

18. Role of Fuzzy Logic

The fuzzy logic system is responsible for the actual traffic timing decision.

The AI model does not directly decide the green-light duration.

The responsibilities are separated as follows:

AI Model
   ↓
Understands Natural Language
   ↓
Extracts Traffic Parameters
   ↓
Fuzzy Logic Controller
   ↓
Calculates Green-Light Duration

This architecture makes the decision process more structured and explainable.

19. Manual Override

The application also provides a manual override option.

The user can manually provide:

Vehicle density.
Waiting time.

This is useful when the user already knows the numerical traffic conditions or wants to test the fuzzy controller independently from the AI model.

20. User Interface

The project uses Streamlit to provide a simple interactive interface.

The interface contains:

Project title.
Traffic scenario input.
Manual override controls.
Calculate button.
Vehicle density result.
Waiting time result.
Green-light duration result.
Traffic intensity visualization.
AI-generated explanation.
Fuzzy membership analysis.
Fuzzy rule information.
Technology stack information.
21. Example Input

Example traffic scenario:

There are heavy vehicles at the intersection and most vehicles have been waiting for a long time.

The AI model may extract values such as:

Vehicle Density = 9
Waiting Time = 8

The fuzzy controller then processes these values and generates an appropriate green-light duration.

22. Example Output

The application displays:

Vehicle Density
9 / 10

Waiting Time
8 minutes

Green-Light Duration
Approximately XX seconds

The exact green-light duration depends on the fuzzy inference calculation.

23. Error Handling

The application includes handling for common errors such as:

Missing API key.
Invalid AI response.
Invalid JSON extraction.
Incorrect numerical values.
Values outside the accepted range.
AI/API request errors.

Traffic parameters are also clamped to the supported range.

24. API Key Configuration

The Groq API key is stored in a .env file.

Example:

GROQ_API_KEY=your_api_key_here

The application loads the key using python-dotenv.

The API key should never be directly written into the Python source code.

25. Security Considerations

The following security practices should be followed:

Do not share the .env file publicly.
Do not upload API keys to GitHub.
Add .env to .gitignore.
Do not expose API keys in screenshots or project reports.
Use environment variables for secret credentials.
26. Advantages

The proposed system provides several advantages:

Accepts natural-language traffic scenarios.
Uses dynamic traffic conditions.
Uses fuzzy logic for gradual decision making.
Provides an interactive user interface.
Supports manual testing.
Provides explainable fuzzy rules.
Can be extended with real-time traffic data.
27. Limitations

The current prototype has some limitations:

It does not directly use real-time traffic cameras.
Vehicle density is estimated rather than measured using physical sensors.
Waiting time is obtained from the user scenario or manual input.
The AI model requires an internet connection and API access.
The current fuzzy rule base is designed as an academic prototype.
The system does not directly control real traffic signals.
28. Future Scope

The project can be extended in several ways.

Real-Time Camera Integration

Traffic cameras can be used to detect vehicles automatically.

Computer Vision

Computer vision techniques can estimate:

Number of vehicles.
Vehicle type.
Traffic density.
Lane occupancy.
IoT Integration

Traffic sensors can send real-time traffic information to the controller.

Multiple Intersections

The system can be extended to coordinate several traffic signals.

Real-Time Traffic Data

Live traffic information can be integrated into the system.

Emergency Vehicle Priority

Emergency vehicles could receive signal priority.

Historical Analysis

Historical traffic data could be stored and analyzed to improve traffic management.

29. Project Directory Structure

The project structure is:

Smart-Traffic-Signal/
│
├── app.py
├── PROJECT_DOCUMENTATION.md
├── requirements.txt
├── .env
├── .gitignore
│
└── .venv/
30. Installation

Create and activate the virtual environment:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

Install the required libraries:

pip install streamlit scikit-fuzzy scipy networkx numpy python-dotenv langchain-groq
31. Environment Setup

Create a file named:

.env

Add:

GROQ_API_KEY=your_groq_api_key

Replace your_groq_api_key with the actual Groq API key.

32. Running the Application

Activate the virtual environment:

.venv\Scripts\Activate.ps1

Run Streamlit:

python -m streamlit run app.py

The application will normally be available at:

http://localhost:8501
33. Algorithm

The complete algorithm is:

START

1. Load environment variables.

2. Load Groq API key.

3. Initialize the AI language model.

4. Create fuzzy input variables:
      Vehicle Density
      Waiting Time

5. Create fuzzy output variable:
      Green-Light Duration

6. Define membership functions.

7. Define fuzzy rules.

8. Accept traffic scenario from the user.

9. Send traffic scenario to the AI model.

10. Extract:
       Vehicle Density
       Waiting Time

11. Clamp values to valid ranges.

12. Perform fuzzification.

13. Evaluate fuzzy rules.

14. Perform fuzzy aggregation.

15. Perform centroid defuzzification.

16. Limit result to 10–90 seconds.

17. Display final green-light duration.

18. Generate an explanation for the result.

END
34. Pseudocode
START

Read traffic scenario

IF manual override is enabled:
    Read vehicle density
    Read waiting time
ELSE:
    Send scenario to AI model
    Extract vehicle density
    Extract waiting time

Validate traffic values

Fuzzify vehicle density

Fuzzify waiting time

FOR each fuzzy rule:
    Calculate rule activation
    Apply rule to output membership

Combine activated fuzzy outputs

Perform centroid defuzzification

Limit green duration between 10 and 90 seconds

Display:
    Vehicle density
    Waiting time
    Green-light duration

Generate explanation

END
35. Testing Strategy

The system should be tested using different traffic conditions.

Test Case	Vehicle Density	Waiting Time	Expected Behavior
Low traffic	2	1	Short green duration
Moderate traffic	5	5	Medium green duration
Heavy traffic	9	8	Long green duration
Low density, long wait	2	9	Long green duration
High density, short wait	9	2	Long green duration

The exact output value depends on fuzzy membership and centroid defuzzification.

36. Sample Scenarios
Scenario 1

Input:

Only a few vehicles are present and nobody has been waiting for long.

Expected traffic condition:

Low Density
Short Waiting Time

The fuzzy system should produce a relatively shorter green duration.

Scenario 2

Input:

There is moderate traffic and vehicles have been waiting for several minutes.

Expected traffic condition:

Medium Density
Medium/Long Waiting Time

The fuzzy system should increase the green duration accordingly.

Scenario 3

Input:

There is very heavy traffic and vehicles have been waiting for a long time.

Expected traffic condition:

High Density
Long Waiting Time

The fuzzy system should produce a longer green duration.

37. Why Fuzzy Logic Is Used

Fuzzy Logic is suitable for this project because traffic conditions are not always binary.

For example:

Low Traffic
Medium Traffic
High Traffic

are not strict boundaries in real-world situations.

A traffic condition can partially belong to multiple categories.

Fuzzy Logic provides a mathematical method to represent this uncertainty and gradual transition.

38. Why AI and Fuzzy Logic Are Combined

AI and fuzzy logic perform different tasks.

AI is useful for understanding human language.

Fuzzy logic is useful for rule-based decision making.

Therefore:

Natural Language
       ↓
      AI
       ↓
Traffic Parameters
       ↓
 Fuzzy Logic
       ↓
Final Timing

This combination provides a flexible input method while maintaining a structured decision-making mechanism.

39. Key Features

The major features of the project are:

Natural-language traffic input.
AI-based traffic parameter extraction.
Fuzzy Logic Controller.
Mamdani inference.
Centroid defuzzification.
Dynamic green-light calculation.
Manual override.
Interactive Streamlit interface.
AI-generated explanation.
Fuzzy rule visualization.
Input validation.
API error handling.
40. Project Outcome

The developed system successfully demonstrates how Artificial Intelligence and Fuzzy Logic can be combined to create an intelligent traffic signal timing prototype.

The system converts a natural-language traffic scenario into numerical traffic parameters and uses fuzzy inference to calculate a green-light duration.

The project demonstrates the practical application of:

Python
Artificial Intelligence
Natural Language Processing
Fuzzy Logic
Streamlit
API integration
41. Conclusion

The Smart Traffic Signal Timing Controller demonstrates an intelligent approach to traffic signal timing.

Instead of relying only on fixed signal timings, the system considers traffic density and waiting time when determining the green-light duration.

The AI model provides natural-language understanding, while the Fuzzy Logic Controller performs the structured decision-making process.

The project is currently an academic prototype, but its architecture can be extended with real-time sensors, cameras, computer vision, IoT devices, and multiple traffic intersections.

Therefore, the project provides a foundation for developing more adaptive and intelligent traffic management systems.

42. Important Project Statement

The core idea of the project can be summarized as:

"AI understands the traffic situation, while Fuzzy Logic decides the appropriate green-light duration."

43. Viva Questions and Answers
Q1. What is the main purpose of this project?

The purpose is to dynamically calculate an appropriate traffic signal green-light duration based on vehicle density and waiting time.

Q2. Why is AI used?

AI is used to understand traffic scenarios written in natural language and extract useful traffic parameters.

Q3. Why is Fuzzy Logic used?

Fuzzy Logic is used to handle gradual and uncertain traffic conditions and convert them into a practical signal timing decision.

Q4. What are the inputs to the fuzzy controller?

The inputs are:

Vehicle Density
Waiting Time
Q5. What is the output?

The output is:

Green-Light Duration in seconds.

Q6. What is the range of vehicle density?

The vehicle density range is:

0–10

Q7. What is the range of waiting time?

The waiting time range is:

0–10 minutes

Q8. What is the range of green-light duration?

The green-light duration is limited to:

10–90 seconds

Q9. What type of fuzzy inference is used?

The system uses Mamdani-style fuzzy inference.

Q10. What is defuzzification?

Defuzzification converts the fuzzy output into a single numerical value.

The project uses the centroid method.

Q11. How many fuzzy rules are used?

The current system uses 9 fuzzy rules.

Q12. What are the membership categories for vehicle density?

They are:

Low
Medium
High
Q13. What are the membership categories for waiting time?

They are:

Short
Medium
Long
Q14. What are the membership categories for green duration?

They are:

Short
Medium
Long
Q15. What is Streamlit?

Streamlit is a Python framework used to create interactive web applications for data science and machine learning projects.

Q16. What is scikit-fuzzy?

scikit-fuzzy is a Python library used to implement fuzzy logic systems.

Q17. What is Groq?

Groq provides an API that allows applications to access language models for AI-based processing.

Q18. Does the AI directly calculate the green-light duration?

No.

The AI primarily extracts traffic parameters from natural language. The fuzzy controller calculates the green-light duration.

Q19. Can the project work without natural-language input?

Yes.

The manual override allows numerical vehicle density and waiting time to be provided directly.

Q20. Can this project be used directly with real traffic signals?

The current version is an academic prototype and does not directly control physical traffic signals.

Additional hardware, safety mechanisms, real-time traffic sensors, and traffic-control integration would be required.

44. Technology Stack Summary
Programming Language
        ↓
      Python
        ↓
     Streamlit
        ↓
   Groq + LangChain
        ↓
    scikit-fuzzy
        ↓
      NumPy
        ↓
   python-dotenv
45. Final Project Summary

Project: Smart Traffic Signal Timing Controller

Input: Natural-language traffic scenario

AI Component: Groq Language Model

Extracted Parameters:

Vehicle Density
Waiting Time

Decision Component: Fuzzy Logic Controller

Inference: Mamdani-style

Defuzzification: Centroid

Output: Green-Light Duration

Output Range: 10–90 seconds

Interface: Streamlit

Programming Language: Python

End of Documentation
