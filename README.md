🐾 Dog Disease Diagnosis Chatbot using Machine Learning

This project is a machine learning-powered chatbot that helps identify potential diseases in dogs based on user input such as the affected body part, symptoms, and severity.
It uses a CSV dataset (projectdataset.csv) containing diseases, symptoms, and recommended treatments.
Built using Python, scikit-learn, and Streamlit for an interactive UI.

📂 Project Structure
├── app.py                  # Main Streamlit app
├── projectdataset.csv      # Dataset with symptoms, diseases, and treatments
├── README.md    

🔍 Features
Interactive web interface using Streamlit

Dropdowns for selecting affected body part and severity

Text input for custom symptoms

Disease prediction using Naive Bayes classification

Treatment recommendation based on diagnosis

Real-time response with friendly UI

🧪 Sample Input
Affected Body Part: skin

Severity: medium

Symptoms: itching, redness, rashes

Output:

Predicted Disease: Dermatitis

Recommended Treatment: Use medicated shampoo, topical ointments

📊 Dataset Format (projectdataset.csv)
The dataset should contain the following columns (example):

disease	part	severity	Symptoms1	Symptoms2	...	treatment
Dermatitis	skin	medium	itching	redness	...	Use medicated shampoo...

🤖 Model Details
Vectorization: Bag of Words using CountVectorizer

Model: MultinomialNB (Naive Bayes)

Filtering: Based on part and severity before model training

📌 Future Improvements
Add symptom autocomplete suggestions

Use deep learning or ensemble models

Store and analyze user queries for improvement

Add voice-based input with speech-to-text

🙋‍♀️ Author
Lavanya

CSE Student passionate about AI, ML, and building real-world tools for pets and people 🐶

