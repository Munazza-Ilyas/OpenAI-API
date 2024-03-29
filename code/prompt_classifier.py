import openai
from joblib import Memory
import backoff

import os
import dotenv

CACHE_DIR = ".cachedir"

#PROMPT = "Please classify the following text as 'ham' or 'spam'. Just say 'ham' or 'spam' and nothing else:"


memory = Memory(CACHE_DIR, verbose=0)

openai.api_requestor.TIMEOUT_SECS = 4

def _backoff_hdlr(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries\n {details['exception']}"
    )


@backoff.on_exception(
    backoff.expo,
    (
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
        openai.error.APIError,
    ),
    on_backoff=_backoff_hdlr,
)

@memory.cache
def classify_text_message(text, api_key):
    """
    Takes a text message and an OpenAI API Key and classifies it as "ham" or "spam".
    If GPT fails to return a valid response ("ham" or "spam") returns "ham".
    """
    try:
        openai.api_key = api_key
        PROMPT = f"Please classify the following text as 'ham' or 'spam'.{text}, Just say 'ham' or 'spam' and nothing else:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=PROMPT,
            max_tokens=5,
            timeout=10
        )
        predicted_label = response.choices[0].text.strip().lower()

        if predicted_label in ('ham', 'spam'):
            return predicted_label
        else:
            return 'ham'  
    except Exception as e:
        print("An error occurred:", e)
        return 'ham'  

if __name__ == "__main__":
    import os
    import dotenv

    dotenv.load_dotenv()

    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    text_messages_labels = [
        ("Cool, I'll text you when I'm on the way", "ham"),
        (
            "Wan2 win a Meet+Greet with Westlife 4 U or a m8? They are currently on what tour? 1)Unbreakable, 2)Untamed, 3)Unkempt. Text 1,2 or 3 to 83049. Cost 50p +std text",
            "spam",
        ),
        ("No plm i will come da. On the way.", "ham"),
    ]

    for text_message, label in text_messages_labels:
        print(f"{text_message=}")
        predicted_label = classify_text_message(
            text=text_message, api_key=OPENAI_API_KEY
        )
        print(f"Expected: {label=}")
        print(f"Observed: {predicted_label=}")
        print()
