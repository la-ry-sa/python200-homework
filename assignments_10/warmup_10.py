#-------------LLMs as Transform----------------

#------------LLMs as Transform Question 1----------------

# Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".

# Deterministic code. No decisions or interpretations needed, just straightforward logic.

# Classify a customer support ticket -- "my card was charged twice" -- into one of: 
# billing, technical, or general.

# LLM. Language processing required.

# Calculate the average of a list of numbers.

# Code. Simple math formula.

# Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp 
# (contract)".

# LLM. Free form means language processing requires, impossible to parse by simple code.

# Determine whether a product review is more than 100 words long.

# Code. Programmable and reproducible.

#----------------LLMs as Transform Question 2---------------------

# The prompt is too vague and doesn't have specific instructions on what exactly
# the summary has to contain. Output will be inconsistent and not possible to 
# integrate into a pipepline.

# Improved prompt:

# system = "Categorize a review: positive, negative or mixed. Use one word only.
# If you're unable to categorize, use 'unknown'."

#---------------LLMs as Transform Question 3------------------------

# 1. 50,000 seconds or 13.8 hours.
# 2. Using BatchApi allows to handle it more efficiently. 

#----------------Azure OpenAI-------------------------------------

#----------------Azure OpenAI Question 1--------------------------

# Azure OpenAi advantages:
# 1. Data stays within the company's infrastructure and is not sent to OpenAi servers.
# 2. Support and billing all go through Microsoft, so less vendor contracts is needed.

#---------------Azure OpenAI Question 2---------------------------

# 1. azure_endpoint - a URL of a particular Azure resource.
# 2. azure-api-key - a key to connect to API within Azure. 
# 3. api_version - API version 

#--------------Azure OpenAI Question 3----------------------------

# The model takes a deployment name. The deployment is created and configured by the 
# company admin. The name is found in Azure AI Foundry.