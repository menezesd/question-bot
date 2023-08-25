Imagine you are given a problem:
```
Implmenet an application with conversational interface that can analyze sequence of user's messages, understand what user is asking for and find an answer in 
the Q&A database ( https://huggingface.co/datasets/web_questions ). 
If app identifies that user's question exactly matches semantics of one of the questions from Q&A and thus it can be fully answered by answer from a singe Q&A entry,
app should give user the answer from database, without any rephrasing. Otherwise it should respond "No information found".
```

The actual task is to build methodology/framework to assess the quality of the implementation of the task above.
- How to say if one version of the application is better or worse? 
- Is there regressions in a newer version? 
- What to do if Q&A database is changed?
And so on.
 
As deliverable we expect:
1. Simple implementation of Q&A application. Its quality and form is not important. Anything reasonable. You may chose any input/output format/interface which is convinient for quality assesment.
2. Implementation of the quality assessment framework, that can be used for automatic assessment. **The assesment framework and methodology are the most important pieces that would be evaluated by our team.**

We suggest to use pure python (with publically available libs) for both deliverables, but you may use different set of technologies as long as it easy to run and not platform dependant.

In order to simplify access to the models we suggest you to use credentials and models from this sample.
```
import openai
import logging

logging.basicConfig(level=logging.DEBUG)

openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

openai.api_key = "de4d5adbc0af45bca22499dc3847b134"
openai.api_base = "https://ai-proxy.epam-rail.com"

deployment_name = "gpt-35-turbo"

message = "how are you?"

print(openai.ChatCompletion.create(
    engine=deployment_name,

    temperature=0,
    messages=[
        {
            "role": "user",
            "content": message
        }
    ],
))

print(openai.Embedding.create(
    engine='text-embedding-ada-002',
    input="how are you?"
))
```
