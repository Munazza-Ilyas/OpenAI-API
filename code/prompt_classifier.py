import openai
from joblib import Memory

CACHE_DIR = ".cachedir"

PROMPT = """
Write this.
"""

memory = Memory(CACHE_DIR, verbose=0)


@memory.cache
def classify_text_message(text, api_key):
    """
    Takes a text message and an OpenAI API Key and classifies it as "ham" or "spam".
    If GPT fails to return a valid response ("ham" or "spam") returns "ham".
    """

    # write this


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
