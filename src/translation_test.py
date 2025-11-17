import json
import os
import random

from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_translation_openai(text_to_translate):
    response = openai_client.responses.create(
        model="gpt-5-nano",
        instructions=(
            "You are a professional English-to-Polish translator. Translate all user messages from English to Polish. "
            "Only output the translated text, no explanations, formatting, or quotes."
        ),
        input=text_to_translate,
    )

    return response.output_text


def get_translation_gemini(text_to_translate):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a professional English-to-Polish translator. Translate all user messages from English to Polish. "
                "Only output the translated text, no explanations, formatting, or quotes."
            ),
        ),
        contents=text_to_translate,
    )

    return response.text


def get_random_texts(num):
    dataset = load_dataset("lavita/MedREQAL", split="train")
    texts = []

    for col in ["question", "objective"]:
        texts.extend([x for x in dataset[col] if x])

    samples = random.sample(texts, num)

    return samples


def test_translation():
    texts_to_translate = get_random_texts(10)

    results = []

    for text in tqdm(texts_to_translate, desc="Translating texts..."):
        openai_translation = get_translation_openai(text)
        gemini_translation = get_translation_gemini(text)

        results.append(
            {
                "text": text,
                "openai_translation": openai_translation,
                "gemini_translation": gemini_translation,
                # placeholders:
                "google_translate": "",
                "deepl_translate": "",
            }
        )

    with open("translations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main():
    test_translation()


if __name__ == "__main__":
    main()
