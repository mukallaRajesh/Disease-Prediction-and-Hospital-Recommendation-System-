import os
from pathlib import Path

from flask import Flask, request, jsonify, render_template
import pandas as pd
from collections import Counter
import joblib

BASE_DIR = Path(__file__).resolve().parent

# Flask app initialization
app = Flask(__name__)

# Load the pre-trained models and data
dis_sym_data_v1 = pd.read_csv(BASE_DIR / "Processed_Dataset1.csv")
doc_data = pd.read_csv(BASE_DIR / "Doctor_Versus_Disease.csv", encoding='latin1', names=['Disease', 'Specialist'])
des_data = pd.read_csv(BASE_DIR / "Disease_Description.csv")
algorithms = joblib.load(BASE_DIR / "trained_algorithms.pkl")  # Save algorithms dict after training
le = joblib.load(BASE_DIR / "label_encoder.pkl")  # Save the label encoder after training

# Column names excluding 'Disease'
dis_sym_data_v1 = dis_sym_data_v1.loc[:, ~dis_sym_data_v1.columns.str.contains('^Unnamed')]

test_col = [col for col in dis_sym_data_v1.columns if col != 'Disease']

@app.route('/')
def home():
    return render_template('index2.html')

# API to get symptom list
@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({'symptoms': test_col})

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     symptoms = data.get('symptoms', [])

#     if not symptoms:
#         return jsonify({"error": "No symptoms provided."}), 400

#     # Validate that all symptoms are in the dataset
#     invalid_symptoms = [symptom for symptom in symptoms if symptom not in test_col]
#     if invalid_symptoms:
#         return jsonify({"error": f"Invalid symptoms provided: {', '.join(invalid_symptoms)}"}), 400

#     # Prepare test data for prediction
#     test_data = {col: 1 if col in symptoms else 0 for col in test_col}
#     test_df = pd.DataFrame(test_data, index=[0])

#     # Predict using multiple algorithms
#     predicted = []
#     for model_name, values in algorithms.items():
#         predict_disease = values["model"].predict(test_df)
#         predict_disease = le.inverse_transform(predict_disease)
#         predicted.extend(predict_disease)

#     # Calculate chances of each predicted disease
#     disease_counts = Counter(predicted)
#     percentage_per_disease = {disease: (count / len(algorithms)) * 100 for disease, count in disease_counts.items()}

#     # Diet recommendations dictionary
#     diet_recommendations = {
#         "Malaria": "Focus on high-calorie foods to combat fatigue. Include fruits like bananas and oranges, leafy greens, and protein sources like chicken and fish.",
#         "Allergy": "Avoid allergens (e.g., nuts, dairy, gluten) based on individual sensitivities. Incorporate anti-inflammatory foods such as fatty fish, berries, and leafy greens.",
#         "Hypothyroidism": "Increase iodine intake with iodized salt and seaweed. Include selenium-rich foods like Brazil nuts and zinc-rich foods such as pumpkin seeds.",
#         "Psoriasis": "Follow an anti-inflammatory diet rich in omega-3 fatty acids (e.g., fish, flaxseeds), fruits, vegetables, and whole grains while avoiding processed foods.",
#         "GERD": "Eat smaller meals and avoid trigger foods like spicy dishes, chocolate, caffeine, and acidic fruits. Focus on lean proteins and whole grains.",
#         "Chronic cholestasis": "Limit fat intake, especially saturated fats. Include fiber-rich foods such as fruits, vegetables, and whole grains to help digestion.",
#         "hepatitis A": "Stay hydrated and consume a balanced diet rich in fruits, vegetables, lean proteins, and whole grains while avoiding alcohol.",
#         "Osteoarthristis": "Focus on anti-inflammatory foods such as fatty fish, nuts, olive oil, and plenty of fruits and vegetables while limiting processed sugars.",
#         "(vertigo) Paroymsal  Positional Vertigo": "Stay hydrated; limit caffeine and alcohol which can exacerbate symptoms. Incorporate vitamin D-rich foods like salmon or fortified dairy.",
#         "Hypoglycemia": "Eat small frequent meals with complex carbohydrates (whole grains), protein (nuts, legumes), and healthy fats to stabilize blood sugar levels.",
#         "Acne": "Reduce dairy intake; include foods rich in antioxidants (berries) and omega-3 fatty acids (fish). Stay hydrated.",
#         "Diabetes": "Focus on low glycemic index foods such as whole grains, legumes, vegetables, lean proteins while monitoring carbohydrate intake.",
#         "Impetigo": "Include nutrient-dense foods to support healing; focus on protein-rich foods (lean meats, eggs) and plenty of fruits and vegetables.",
#         "Hypertension": "Adopt a DASH diet rich in fru3its, vegetables, whole grains, lean protein while reducing sodium intake.",
#         "Peptic ulcer diseae": "Avoid spicy foods; include high-fiber foods like oats and bananas while consuming smaller meals throughout the day.",
#         "Dimorphic hemorrhoids(piles)": "Increase fiber intake through fruits, vegetables, and whole grains to prevent constipation; stay hydrated.",
#         "Common Cold": "Stay hydrated; consume vitamin C-rich foods (citrus fruits) and zinc-rich foods (nuts) to support immune function.",
#         "Chicken pox": "Maintain hydration; eat soft foods if lesions are present in the mouth; include plenty of fruits and vegetables for vitamins.",
#         "Cervical spondylosis": "Focus on anti-inflammatory foods such as fatty fish and nuts; maintain a balanced diet with adequate calcium for bone health.",
#         "Hyperthyroidism": "Limit iodine-rich foods (seafood); focus on a balanced diet with adequate protein from lean sources.",
#         "Urinary tract infection": "Drink plenty of water; incorporate cranberry juice which may help prevent UTIs; avoid irritants like caffeine and alcohol.",
#         "Varicose veins": "Increase fiber intake to prevent constipation; include flavonoid-rich foods such as berries to improve circulation.",
#         "AIDS": "Focus on nutrient-dense foods to support immune function; include lean proteins, whole grains, fruits, and vegetables while avoiding processed sugars.",
#         "Paralysis (brain hemorrhage)": "Consume a balanced diet rich in antioxidants (fruits/vegetables), omega-3 fatty acids (fish), and adequate hydration for recovery support.",
#         "Typhoid": "Focus on easily digestible foods like rice or bananas; avoid raw fruits/vegetables until recovery is complete; stay hydrated.",
#         "Hepatitis B": "Avoid alcohol; focus on a balanced diet rich in vitamins A, C, E from fruits/vegetables to support liver health.",
#         "Fungal infection": "Reduce sugar intake as it can promote fungal growth; include garlic which has antifungal properties along with probiotics from yogurt.",
#         "Hepatitis C": "Focus on a liver-friendly diet rich in antioxidants from fruits/vegetables; avoid alcohol completely for liver health.",
#         "Migraine": "Identify trigger foods (e.g., aged cheese); maintain hydration; consume magnesium-rich foods like spinach or nuts that may help reduce frequency.",
#         "Bronchial Asthma": "Avoid food allergens that trigger asthma attacks; include omega-3 fatty acids from fish which may help reduce inflammation.",
#         "Alcoholic Hepatitis": "Abstain from alcohol completely; focus on a balanced diet rich in nutrients to support liver recovery including proteins from lean meats or legumes.",
#         "Jaundice": "Avoid fatty or fried foods; increase fluid intake; consume fresh fruits/vegetables to support liver function.",
#         "Hepatitis E": "Stay hydrated; focus on a balanced diet rich in nutrients while avoiding alcohol until recovery is complete.", 
#         "Dengue": "Stay hydrated with fluids like coconut water; consume nutrient-dense foods like papaya leaves which may help increase platelet count.", 
#         "Pneumonia": "Stay hydrated with fluids; focus on nutrient-dense foods that support immune function such as chicken soup or broth-based meals during recovery.", 
#         "Hepatitis D": "Similar to other hepatitis diets—avoid alcohol; eat a balanced diet rich in vitamins to support liver health.", 
#         "Heart Attack": "Adopt a heart-healthy diet low in saturated fats and cholesterol—focus on whole grains, lean proteins (fish/chicken), fruits/vegetables.", 
#         "Arthritis": "Follow an anti-inflammatory diet rich in omega-3 fatty acids from fish/nuts along with plenty of fruits/vegetables while avoiding processed sugars.", 
#         "Gastroenteritis": "Stay hydrated with clear fluids; consume bland foods like bananas or rice until symptoms improve before reintroducing regular diet slowly.", 
#         "Tuberculosis":"Focus on high-calorie nutrient-dense foods to combat weight loss associated with TB—include proteins from meat/dairy along with plenty of fruits/vegetables for vitamins/minerals."
# }
#     # Create response
#     result_df = pd.DataFrame({"Disease": list(percentage_per_disease.keys()),
#                               "Chances": list(percentage_per_disease.values())})
#     result_df = result_df.merge(doc_data, on='Disease', how='left')
#     result_df = result_df.merge(des_data, on='Disease', how='left')

#     results = []
#     for _, row in result_df.iterrows():
#         disease_name = row['Disease']
#         diet_recommendation = diet_recommendations.get(disease_name, 'Diet recommendation not available')
#         results.append({
#             "Disease": row['Disease'],
#             "Chances": row['Chances'],
#             "Specialist": row.get('Specialist', 'Specialist info not available'),
#             "Description": row.get('Description', 'Description not available'),
#             "Diet Recommendation": diet_recommendation
#         })

#     return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    symptoms = data.get('symptoms', [])
    allergies = data.get('allergies', [])

    if not isinstance(symptoms, list) or any(not isinstance(symptom, str) for symptom in symptoms):
        return jsonify({"error": "Symptoms must be a list of symptom names."}), 400
    if not symptoms:
        return jsonify({"error": "No symptoms provided."}), 400

    if not isinstance(allergies, list) or any(not isinstance(allergy, str) for allergy in allergies):
        return jsonify({"error": "Allergies must be a list of names."}), 400

    invalid_symptoms = [symptom for symptom in symptoms if symptom not in test_col]
    if invalid_symptoms:
        return jsonify({"error": f"Invalid symptoms provided: {', '.join(invalid_symptoms)}"}), 400

    test_data = {col: 1 if col in symptoms else 0 for col in test_col}
    test_df = pd.DataFrame(test_data, index=[0])

    predicted = []
    for values in algorithms.values():
        predict_disease = values["model"].predict(test_df)
        predict_disease = le.inverse_transform(predict_disease)
        predicted.extend(predict_disease)

    disease_counts = Counter(predicted)
    percentage_per_disease = {disease: (count / len(algorithms)) * 100 for disease, count in disease_counts.items()}

    diet_recommendations = {
            "Malaria": "Focus on high-calorie foods to combat fatigue. Include fruits like bananas and oranges, leafy greens, and protein sources like chicken and fish.",
            "Allergy": "Avoid allergens (e.g., nuts, dairy, gluten) based on individual sensitivities. Incorporate anti-inflammatory foods such as fatty fish, berries, and leafy greens.",
            "Hypothyroidism": "Increase iodine intake with iodized salt and seaweed. Include selenium-rich foods like Brazil nuts and zinc-rich foods such as pumpkin seeds.",
            "Psoriasis": "Follow an anti-inflammatory diet rich in omega-3 fatty acids (e.g., fish, flaxseeds), fruits, vegetables, and whole grains while avoiding processed foods.",
            "GERD": "Eat smaller meals and avoid trigger foods like spicy dishes, chocolate, caffeine, and acidic fruits. Focus on lean proteins and whole grains.",
            "Chronic cholestasis": "Limit fat intake, especially saturated fats. Include fiber-rich foods such as fruits, vegetables, and whole grains to help digestion.",
            "hepatitis A": "Stay hydrated and consume a balanced diet rich in fruits, vegetables, lean proteins, and whole grains while avoiding alcohol.",
            "Osteoarthristis": "Focus on anti-inflammatory foods such as fatty fish, nuts, olive oil, and plenty of fruits and vegetables while limiting processed sugars.",
            "(vertigo) Paroymsal  Positional Vertigo": "Stay hydrated; limit caffeine and alcohol which can exacerbate symptoms. Incorporate vitamin D-rich foods like salmon or fortified dairy.",
            "Hypoglycemia": "Eat small frequent meals with complex carbohydrates (whole grains), protein (nuts, legumes), and healthy fats to stabilize blood sugar levels.",
            "Acne": "Reduce dairy intake; include foods rich in antioxidants (berries) and omega-3 fatty acids (fish). Stay hydrated.",
            "Diabetes": "Focus on low glycemic index foods such as whole grains, legumes, vegetables, lean proteins while monitoring carbohydrate intake.",
            "Impetigo": "Include nutrient-dense foods to support healing; focus on protein-rich foods (lean meats, eggs) and plenty of fruits and vegetables.",
            "Hypertension": "Adopt a DASH diet rich in fru3its, vegetables, whole grains, lean protein while reducing sodium intake.",
            "Peptic ulcer diseae": "Avoid spicy foods; include high-fiber foods like oats and bananas while consuming smaller meals throughout the day.",
            "Dimorphic hemorrhoids(piles)": "Increase fiber intake through fruits, vegetables, and whole grains to prevent constipation; stay hydrated.",
            "Common Cold": "Stay hydrated; consume vitamin C-rich foods (citrus fruits) and zinc-rich foods (nuts) to support immune function.",
            "Chicken pox": "Maintain hydration; eat soft foods if lesions are present in the mouth; include plenty of fruits and vegetables for vitamins.",
            "Cervical spondylosis": "Focus on anti-inflammatory foods such as fatty fish and nuts; maintain a balanced diet with adequate calcium for bone health.",
            "Hyperthyroidism": "Limit iodine-rich foods (seafood); focus on a balanced diet with adequate protein from lean sources.",
            "Urinary tract infection": "Drink plenty of water; incorporate cranberry juice which may help prevent UTIs; avoid irritants like caffeine and alcohol.",
            "Varicose veins": "Increase fiber intake to prevent constipation; include flavonoid-rich foods such as berries to improve circulation.",
            "AIDS": "Focus on nutrient-dense foods to support immune function; include lean proteins, whole grains, fruits, and vegetables while avoiding processed sugars.",
            "Paralysis (brain hemorrhage)": "Consume a balanced diet rich in antioxidants (fruits/vegetables), omega-3 fatty acids (fish), and adequate hydration for recovery support.",
            "Typhoid": "Focus on easily digestible foods like rice or bananas; avoid raw fruits/vegetables until recovery is complete; stay hydrated.",
            "Hepatitis B": "Avoid alcohol; focus on a balanced diet rich in vitamins A, C, E from fruits/vegetables to support liver health.",
            "Fungal infection": "Reduce sugar intake as it can promote fungal growth; include garlic which has antifungal properties along with probiotics from yogurt.",
            "Hepatitis C": "Focus on a liver-friendly diet rich in antioxidants from fruits/vegetables; avoid alcohol completely for liver health.",
            "Migraine": "Identify trigger foods (e.g., aged cheese); maintain hydration; consume magnesium-rich foods like spinach or nuts that may help reduce frequency.",
            "Bronchial Asthma": "Avoid food allergens that trigger asthma attacks; include omega-3 fatty acids from fish which may help reduce inflammation.",
            "Alcoholic Hepatitis": "Abstain from alcohol completely; focus on a balanced diet rich in nutrients to support liver recovery including proteins from lean meats or legumes.",
            "Jaundice": "Avoid fatty or fried foods; increase fluid intake; consume fresh fruits/vegetables to support liver function.",
            "Hepatitis E": "Stay hydrated; focus on a balanced diet rich in nutrients while avoiding alcohol until recovery is complete.", 
            "Dengue": "Stay hydrated with fluids like coconut water; consume nutrient-dense foods like papaya leaves which may help increase platelet count.", 
            "Pneumonia": "Stay hydrated with fluids; focus on nutrient-dense foods that support immune function such as chicken soup or broth-based meals during recovery.", 
            "Hepatitis D": "Similar to other hepatitis diets—avoid alcohol; eat a balanced diet rich in vitamins to support liver health.", 
            "Heart Attack": "Adopt a heart-healthy diet low in saturated fats and cholesterol—focus on whole grains, lean proteins (fish/chicken), fruits/vegetables.", 
            "Arthritis": "Follow an anti-inflammatory diet rich in omega-3 fatty acids from fish/nuts along with plenty of fruits/vegetables while avoiding processed sugars.", 
            "Gastroenteritis": "Stay hydrated with clear fluids; consume bland foods like bananas or rice until symptoms improve before reintroducing regular diet slowly.", 
            "Tuberculosis":"Focus on high-calorie nutrient-dense foods to combat weight loss associated with TB—include proteins from meat/dairy along with plenty of fruits/vegetables for vitamins/minerals."
}

    def remove_allergens(diet_str, allergens):
        for allergen in allergens:
            allergen = allergen.lower()
            if allergen in diet_str.lower():
                diet_str = diet_str.replace(allergen, "Not Recommended due to allergy")
        return diet_str

    result_df = pd.DataFrame({"Disease": list(percentage_per_disease.keys()),
                              "Chances": list(percentage_per_disease.values())})
    result_df = result_df.merge(doc_data, on='Disease', how='left')
    result_df = result_df.merge(des_data, on='Disease', how='left')

    results = []
    for _, row in result_df.iterrows():
        disease_name = row['Disease']
        diet_recommendation = diet_recommendations.get(disease_name, 'Diet recommendation not available')
        if allergies:
            diet_recommendation = remove_allergens(diet_recommendation, allergies)

        results.append({
            "Disease": row['Disease'],
            "Chances": row['Chances'],
            "Specialist": row.get('Specialist', 'Specialist info not available'),
            "Description": row.get('Description', 'Description not available'),
            "Diet Recommendation": diet_recommendation
        })

    return jsonify(results)



if __name__ == '__main__':
    app.run(debug=True)
