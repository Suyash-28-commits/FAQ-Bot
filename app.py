#Steps of building a FAQ - Bot
#Step 1 : Import the libraries
#Step 2 : Make a user query
#Step 3 : Maintain a list of answers and a list of questions
#Step 4 : Convert answers and questions to embeddings
#Step 5 : Check the cosine similarity of user_query and already asked questions
#Step 6 : Get the index of the highest answering question
#Step 7 : Perform cosine similarity again between question and answer
#Step 8 : Output : Answer with highest score

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

freq_asked_questions=[
    "What does the company do?",
    "Who founded the company?",
    "When was the company founded?",
    "What are the main AI products of the company?",
    "Does the company offer free AI tools?",
    "How can I contact the company?",
    "Where is the company located?",
    "Does the company do AI research?",
    "Can I use the company's APIs in my projects?",
    "What programming languages are supported by the company's tools?"
]

answers = [
    "The company develops advanced artificial intelligence models and tools for various applications, including natural language processing, image generation, and code generation",
    "The company was founded by Elon Musk, Sam Altman, Greg Brockman, Ilya Sutskever, Wojciech Zaremba, and John Schulman",
    "The company was founded in December 2015",
    "Some of the main AI products are GPT (ChatGPT), DALL·E, Codex, and Whisper",
    "Yes, the company offers limited free access to some of its AI tools, while more advanced features are paid",
    "You can contact the company via their official website or support email provided on their website",
    "The company is headquartered in San Francisco, California, USA",
    "Yes, the company conducts extensive AI research in areas like deep learning, reinforcement learning, and natural language processing",
    "Yes, the company provides APIs that developers can use to integrate AI capabilities into their applications",
    "The company's APIs and tools mainly support Python, but some tools can also be accessed via HTTP requests from other programming languages"
]

mapping = {key:value for key ,value in zip(freq_asked_questions,answers)}
#mapping : key -> question , value -> answer
user_query = "Who founded the company?"

#384 Dense Dimensional Vector
user_query_embedding = embedding.embed_query(user_query)
# print("User query embedding \n")
# print(user_query_embedding)
# print("\n")
# print("\n")
# print("\n")
#Question embeddings
# print("Frequently asked questions embedding \n")

#matrix of embeddings of each vector
question_embedding = embedding.embed_documents(freq_asked_questions)

#matrix of cosine similarity scores
score_list = cosine_similarity([user_query_embedding],question_embedding)
print("List of scores \n")
print(score_list)
#Extracting first or zeroth index of matrix -> index of first vector
print("Extracting 0th vector from list of scores \n")
score = score_list[0]
print(score)
#enumerated list
enum_list = enumerate(score)

# print("Sorted cosine similarity score \n")
highest_scoring = sorted(enum_list,key= lambda x : float(x[1]))
# print(highest_scoring)

index , highest_score = highest_scoring[-1]
# print(index)

#Steps to find the answer of question user is asking
#Step 1 : Get index of highest similarity question
#Step 2 : Initialize a count variable 
#Step 3 : if count == index , store the answer into variable ans
#Step 4 : print ans
count = 0
ans =""
for i in mapping.keys():
    if(count == index):
        ans = mapping.get(i)
        break
    else:
        count+=1

print("User Query: ",user_query)
print("Answer: ",ans)




