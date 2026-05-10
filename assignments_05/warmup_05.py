from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

#------Part 1: Warmup Exercises--------------

#----------The Chat Completions API---------
#----------API Question 1------------------

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(response.choices[0].message.content)
print(f"Model: {response.model}; tokens used: {response.usage.total_tokens}")

#--------API Question 2------------

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response2 = client.chat.completions.create(
         model="gpt-4o-mini",
         messages=[{"role": "user", "content": prompt}],
         n=1,
         temperature=temp
    )

    print(f"Temperature: {temp}. {response2.choices[0].message.content}")

# Temperatures of 0.7 and 1.5 provide more options and/or explanation. Temp 0 provides the 
# most consistent and reproducbile result.

#-----------API Question 3-----------

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i in range(3):
    print(f"Fun fact {i+1}: {response.choices[i].message.content}")

#--------API Question 4-------

max_tokens=15

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=max_tokens
)

print(response.choices[0].message.content)

# The length of the message was limited to 15 tokens. I'd use it in a real application
# to avoid lengthy responses.

#-----------System Messages and Personas--------------

#----------System Question 1--------------------

messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print(f"Encouraging and friendly tutor: {response.choices[0].message.content}")

messages = [
    {"role": "system", "content": "You are a mentor in a Python 200 course. You're used to work with advanced students and you get impatient and critical when someone doesn't know basics."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print(f"Demanding and critical mentor: {response.choices[0].message.content}")

# The tone changes to more abrupt and impatient - as described in the system role.

#-----------System Question 2----------------

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print(f"Q2 response: {response.choices[0].message.content}")

# The model knows the name because the name is mentioned in the messages that are
# passed to it.

#--------------Prompt Engineering--------------

def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

#-------------Prompt Question 1 — Zero-Shot--------

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

task = "Label each review as positive, negative, or mixed: "
reviews_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews))
result = get_completion(task + reviews_text)

print(result)

#------------------Prompt Question 2 — One-Shot------------

example = "Example: " \
"\n Review: Fast shipping but the item arrived damaged. " \
"\n Sentiment: mixed"

result = get_completion(task + example + reviews_text)

print(result)

# With the example, the format changed but the consistency remained the same.

#---------Prompt Question 3 — Few-Shot-------------

example1 = "Example: " \
"\n Review: Great quality, reasonable price. " \
"\n Sentiment: positive"

example2 = "Example: " \
"\n Review: Arrived late, didn't last long. " \
"\n Sentiment: negative"

result = get_completion(task + example + example1 + example2 + reviews_text)

print(result)

# todo: add comment

#-------------Prompt Question 4 — Chain of Thought-----

problem = """
A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""
task2 = """Please solve the following problem, show its reasoning step by step
before giving a final answer. Do not use markdown formatting.
"""
result = get_completion(task2 + problem)

print(result)