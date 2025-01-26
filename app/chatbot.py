from app.utils import detect_sentiment, log_error, log_info
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Hugging Face model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(user_input):
    """
    Generates a response from the AI therapy chatbot, tailored to sentiment.
    """
    try:
        # Detect sentiment using the utility function
        sentiment = detect_sentiment(user_input)
        log_info(f"Detected sentiment: {sentiment}")

        # Customize prompt based on sentiment
        if sentiment == "positive":
            prompt = (
                "You are a supportive and empathetic AI therapist. Respond to the user's positive feelings "
                "with encouragement and reinforcement. Keep your response under 3 sentences.\n\n"
                f"User: {user_input}\nAI:"
            )
        elif sentiment == "negative":
            prompt = (
                "You are a compassionate AI therapist. Respond empathetically to the user's struggles, "
                "providing support and encouragement. Keep your response under 3 sentences.\n\n"
                f"User: {user_input}\nAI:"
            )
        else:  # Neutral
            prompt = (
                "You are a helpful and professional AI therapist. Respond to the user's message in a supportive way, "
                "offering general advice and encouragement. Keep your response under 3 sentences.\n\n"
                f"User: {user_input}\nAI:"
            )

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Generate response
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )

        # Decode and return the generated text
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split("AI:")[-1].strip()  

        # Remove repeated or redundant sentences
        unique_sentences = []
        for line in response.split("\n"):
            if line.strip() and line not in unique_sentences:
                unique_sentences.append(line.strip())
        response = " ".join(unique_sentences)

        log_info(f"Generated response: {response}")
        return response

    except Exception as e:
        log_error(f"Error generating AI response: {e}")
        return "I'm sorry, I couldn't process that. Please try again."
