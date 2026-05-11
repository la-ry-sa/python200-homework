#----------------Task 1: Setup and System Prompt---------------------

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are a friendly and supportive job application coach. Your audience is bootcamp 
graduates who are looking for entry level jobs. You should be helping users to 
learn and apply job search strategies, create resumes and write cover letters.
While coaching, please stay focused on job application materials. Always remind the 
user to review and edit your output before submitting anywhere.
Acknowledge that you may not know the user's specific industry norms, and encourage 
the user to use their own judgment.
"""

# Listed concrete tasks the coach should do to avoid being too vague.

#--------------Task 2: Bullet Point Rewriter-----------------------

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON array, starting with [ and ending with ]. No wrapping object, no 
    markdown, no backticks, no explanation.Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result
    final_result = []
    try:
        raw_result = get_completion(messages)
        result = json.loads(raw_result)
        
        for item in result:
            print(f"Original: {item['original']}")
            print(f"Improved: {item['improved']}\n")
        return result

    except json.JSONDecodeError:
        print(f"Error parsing the results: {raw_result}")
    except Exception as e:
        print(f"Error: {e}")

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# Original bullets were weak because they used simplistic vocabulary and
# focused on a process, not mentioning the results at all.
# Updated version makes each bullet look like an achievement, rather than a task
# and is more detailed.

#--------------Task 3: Cover Letter Generator--------------------------

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés,
    like 'innovative solutions' or 'unique perspective'. Make it sound as if written 
    by a person that is not copying generic cover letters but rather creates their 
    own text.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    result = get_completion(messages)
    print(result)
    return result

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

background2 = "Ten years as a literary translator; college degree in linguistics;" \
"completed a few data and Python bootcamps."

generate_cover_letter(job_title, background)
generate_cover_letter(job_title, background2)

# The examples help the model to set the correct tone and avoid generic cliches,
# recognize patterns and make results more consistent. It also helps to make sure
# the output is formatted correctly.

#-------------------------Task 4: Moderation Check-------------------

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged 
    if(flagged):
        print("Your input was flagged. Please rephrase and try again.")
        print(result.results[0].categories)
        return False
    else:
        # print("Message is safe.")
        return True
        
input1 = "Hey there, can you help me with my job search?"
input2 = "Hey, need your help! i'm so mad i could punch someone"

is_safe(input1)
is_safe(input2)

#-------------Task 5: The Chatbot Loop-----------------

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            rewrite_bullets(raw_bullets)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            generate_cover_letter(job_title, background)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})    


if __name__ == "__main__":
    run_chatbot()

#-----------------Task 6: Ethics Reflection--------------------

# Your bot was trained on text written by and about certain kinds of people. 
# How might this produce biased advice? Could it favor certain communication styles, 
# industries, or cultural backgrounds?

# The bot could make recommendations that wouldn't really work for an underepresented 
# group. It could be oriented to the IT industry if it was trained mostly by IT 
# professional.

# What could go wrong if a job-seeker submitted the bot's output directly — without 
# reviewing it — to a real employer?

# Copy-pasting the output without reviewing can create embarrassing situations.
# Many bots use introduction or conclusion phrases, like "Here's a cover letter you 
# could use" or "Let me know if you need to adjust anything". It can also
# contain factual mistakes.