import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from io import BytesIO
from PIL import Image
from classify import classifier
import torch
from CNN import CNN


# Initialize the model before loading weights
model = CNN()  # Ensure MyModel is defined
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))  # Load state_dict into the model
class_names = {0:'NORMAL', 1:'PNEUMONIA'}

# Telegram Bot Token (replace with your own)
TOKEN = "7228489625:AAEqlagO9ohMVBH29Fcnd3luDvfJ0iESNn0"

# Classification Function
def classify_image(image):
    class_name, confidence_score = classifier(image, model, class_names)
    return class_name, confidence_score

# Start Command
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "🫁 Welcome to the Chest X-ray Classification Bot! 📸\n"
        "Send me a chest X-ray image, and I'll analyze it to determine whether it shows signs of **PNEUMONIA** or is **NORMAL**. 🧪"
    )


# Help Command
async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "🫁 This bot can classify chest X-ray images to detect **PNEUMONIA** or confirm if the lungs are **NORMAL**. 📸\n"
        "Simply send a chest X-ray image, and the bot will analyze it and provide the classification result.\n\n"
        "Commands:\n"
        "/start - Welcome message\n"
        "/help - Information about how to use the bot\n"
        "/about - Details about the model and how it works"
    )
    await update.message.reply_text(help_text)


# About Command
async def about_command(update: Update, context: CallbackContext):
    about_text = (
        "🫁 **Chest X-ray Classification**: This bot classifies chest X-ray images as either **PNEUMONIA** or **NORMAL** using a Convolutional Neural Network (CNN).\n\n"
        "🔍 **Preprocessing & Data Augmentation**:\n"
        "   - Normalizes images to a 0-1 scale.\n"
        "   - Applies transformations like rotation, shifting, shearing, zooming, and horizontal flipping for better model generalization.\n\n"
        "🚀 **CNN Architecture**:\n"
        "   - Five convolutional layers with **batch normalization** and **max pooling** for robust feature extraction.\n"
        "   - Fully connected layers to classify X-ray images with high accuracy.\n\n"
        "🎯 **Training Strategy**:\n"
        "   - Uses **Categorical Crossentropy** as the loss function.\n"
        "   - Optimized with the **Adam optimizer**.\n"
        "   - Implements **cosine annealing learning rate scheduling** for adaptive learning rates.\n"
        "   - Saves the best model checkpoint based on validation accuracy.\n\n"
        "📊 **Evaluation & Visualization**:\n"
        "   - Plots training & validation **loss** and **accuracy** curves for performance monitoring.\n"
        "   - Uses test images to predict class and displays confidence scores.\n\n"
        "⚡ **Hardware Acceleration**: Uses **GPU acceleration** via TensorFlow for faster training and inference.\n"
    )
    await update.message.reply_text(about_text)



# Handle Photo Messages
async def handle_photo(update: Update, context: CallbackContext):
        photo = update.message.photo[-1:][0]  # Process the highest resolution photo
        file = await photo.get_file()
        image_bytes = BytesIO(await file.download_as_bytearray())
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert NumPy array to PIL image
        image = Image.fromarray(image)
        
        label, confidence = classify_image(image)

        # Send original image with classification result
        image_bytes.seek(0)  # Reset file pointer before sending    
        await update.message.reply_photo(photo=image_bytes, caption=f"Original Image: {label} ({confidence}%)")

# Main Function
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
