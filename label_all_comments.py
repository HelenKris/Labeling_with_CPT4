import json
import openai
import re
import time

openai.api_key = "<your_api_key>"
input_file_path = "comments.txt"
output_file_path = "output_responses.json"

def get_emotion_labels(input_text):
    prompt = f"Classify the emotion labels (admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral) for the following prompt:\n{input_text}\nEmotion labels:"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.1,
        max_tokens=50
    )

    response_text = response.choices[0].text.strip()
    emotion_labels = re.findall(r'\b(?:admiration|amusement|anger|annoyance|approval|caring|confusion|curiosity|desire|disappointment|disapproval|disgust|embarrassment|excitement|fear|gratitude|grief|joy|love|nervousness|optimism|pride|realization|relief|remorse|sadness|surprise|neutral)\b', response_text, flags=re.I)

    # Returning the input text along with the extracted emotion labels
    return {"text": input_text, "emotion_labels": emotion_labels}


def convert_ls_format(input_dict):
    """
    Convert sentiment analysis output from a simple format to Label Studio's prediction format.

    Args:
        input_dict (dict): A dictionary containing text and emotion_labels keys.
            Example: {'text': 'I had a terrible time at the party last night!',
                       'emotion_labels': ['disappointment', 'sadness', 'annoyance', 'fear', 'disapproval']}

    Returns:
        dict: A dictionary in Label Studio's prediction format.
    """
    score_value = 1.00  # We don't know the model confidence
    output_dict = {
        "data": {
            "text": input_dict["text"]
        },
        "predictions": []
    }

    for emotion_label in input_dict["emotion_labels"]:
        prediction = {
            "result": [
                {
                    "value": {
                        "choices": [
                            emotion_label.capitalize()  # Capitalize each emotion label
                        ]
                    },
                    "from_name": "sentiment",
                    "to_name": "text",
                    "type": "choices"
                }
            ],
            "score": score_value,
            "model_version": "gpt-3.5-turbo"
        }
        output_dict["predictions"].append(prediction)

    return output_dict


with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    examples = []
    for line in input_file:
        text = line.strip()
        if text:
            examples.append(convert_ls_format(get_emotion_labels(text)))
    output_file.write(json.dumps(examples))
    # time.sleep(60)
