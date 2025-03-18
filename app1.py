from flask import Flask, request, jsonify, render_template
from groq import Groq
import pyttsx3
import threading
import markdown
import re
from textblob import TextBlob
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Create Groq client with API key from environment variable
client = Groq(
    api_key=os.getenv('GROQ_API_KEY')
)

# Emotional context mapping
EMOTION_CONTEXT = {
    'positive': {
        'tone': 'enthusiastic',
        'rate': 150,
        'volume': 1.0
    },
    'negative': {
        'tone': 'empathetic',
        'rate': 120,
        'volume': 0.8
    },
    'neutral': {
        'tone': 'calm',
        'rate': 130,
        'volume': 0.9
    }
}

def analyze_sentiment(text):
    """
    Analyze the sentiment of the input text using TextBlob.
    Returns a dictionary with polarity and subjectivity scores.
    """
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def get_emotional_context(sentiment_score):
    """
    Determine emotional context based on sentiment score.
    """
    if sentiment_score > 0.2:
        return 'positive'
    elif sentiment_score < -0.2:
        return 'negative'
    else:
        return 'neutral'

def adjust_voice_parameters(emotion):
    """
    Adjust voice parameters based on emotional context.
    """
    context = EMOTION_CONTEXT.get(emotion, EMOTION_CONTEXT['neutral'])
    engine.setProperty('rate', context['rate'])
    engine.setProperty('volume', context['volume'])

def clean_response(raw_response):
    """
    Cleans the raw response to remove unnecessary symbols like *, _, etc.
    Also ensures structured formatting.
    """
    # Remove unwanted characters (*, _, etc.) while keeping useful content
    cleaned_response = re.sub(r"[*_~`]", "", raw_response)

    # Replace newlines and multiple spaces with a single space
    cleaned_response = re.sub(r"\s+", " ", cleaned_response)

    # Convert the cleaned text to Markdown for proper formatting
    # Use Markdown to HTML, then strip HTML tags before passing it
    formatted_response = markdown.markdown(cleaned_response)
    
    # Strip out HTML tags from the markdown conversion
    plain_text = re.sub(r"<[^>]*>", "", formatted_response)

    # Return the cleaned and formatted response
    return plain_text

def format_response_for_speech(text):
    """
    Clean up the formatted response for text-to-speech, ensuring it sounds natural.
    """
    # Remove HTML tags and extra spaces
    clean_text = re.sub(r"<[^>]*>", "", text)
    return re.sub(r"\n", " ", clean_text).strip()

def get_llama_response(prompt, emotion_context):
    """Fetch a response from ChatGroq's Llama model and format it with emotional awareness."""
    try:
        # Create an emotionally aware system message
        system_message = f"""You are an emotionally intelligent AI assistant. The user's message appears to be {emotion_context['tone']}. 
        Please respond in a way that acknowledges their emotional state and provides appropriate support or enthusiasm.
        Maintain a {emotion_context['tone']} tone in your response while being helpful and informative."""

        # Call the Groq API's chat completion endpoint with emotional context
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",  # Updated to use a supported model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        # Extract and clean the response
        raw_response = completion.choices[0].message.content
        cleaned_response = clean_response(raw_response)

        return cleaned_response
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

def speak_response(text):
    """Function to handle text-to-speech asynchronously."""
    # Clean the response before passing it to the text-to-speech engine
    clean_text = format_response_for_speech(text)

    # Pass the cleaned text to the TTS engine
    engine.say(clean_text)
    engine.runAndWait()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()

    if user_message.lower() == "stop":
        engine.stop()  # Stop any ongoing speech synthesis
        return jsonify({"reply": "Speaking functionality has been stopped."})

    # Analyze user's emotional state
    sentiment = analyze_sentiment(user_message)
    emotion_context = get_emotional_context(sentiment['polarity'])
    
    # Adjust voice parameters based on emotional context
    adjust_voice_parameters(emotion_context)

    # Fetch response from Llama via ChatGroq with emotional context
    bot_reply = get_llama_response(user_message, EMOTION_CONTEXT[emotion_context])

    # Create a thread to handle the speech synthesis
    threading.Thread(target=speak_response, args=(bot_reply,)).start()

    return jsonify({
        "reply": bot_reply,
        "emotion": {
            "context": emotion_context,
            "polarity": sentiment['polarity'],
            "subjectivity": sentiment['subjectivity']
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
