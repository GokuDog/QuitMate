import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Step 1: Create the Dataset

# Data has been taken from a survey of 10 students
data = {
    'stress_level': [7, 8, 9, 6, 5, 8, 4, 7, 6, 9], #Stress Level: 1-10
    'trigger':      [0, 0, 0, 0, 0, 0, 1, 1, 2, 2], #Trigger: Peer Pressure - 0, Boredom - 1, Anxiety - 2
    'prev_day_vaped': [1, 1, 1, 0, 0, 1, 1, 0, 1, 0], # 0 = No, 1 = Yes
    'vaped_today':    [1, 1, 1, 0, 0, 1, 0, 0, 1, 1] #0 = No, 1 = Yes
}

df = pd.DataFrame(data)

# Step 2: Train the model

X = df[['stress_level', 'trigger', 'prev_day_vaped']]
y = df['vaped_today']

model = LogisticRegression()
model.fit(X, y)


# Step 3: Motivation messages


def get_motivation(reason):
    if reason == "stress":
        return "You're stronger than your cravings. Take deep breaths and keep pushing forward."
    elif reason == "boredom":
        return "Try a walk, call a friend, or do something creative instead!"
    elif reason == "peer_pressure":
        return "You're in control. Be proud for saying NO."
    elif reason == "habit":
        return "Habits can be broken — and you're already doing it!"
    else:
        return "Whatever the reason, your health and goals matter more!"

# Step 4: Tracking and Prediction

streak = 0

# Simulate one day
def run_quitmate_day():
    global streak
    
    print("\n--- QuitMate Daily Tracker ---")
    
    # 1. Input Reason
    print("Why did you feel like vaping today?")
    print("Options: stress, boredom, peer_pressure, habit, other")
    reason = input("Enter reason: ").strip().lower()

    # 2. Stress Level
    stress = int(input("On a scale of 0 to 10, how stressed were you today? "))

    # 3. Trigger 
    print("What triggered your craving?")
    print("0 = peer, 1 = boredom, 2 = anxiety")
    trigger = int(input("Enter trigger number: "))

    # 4. Previous Day Status
    prev_vaped = int(input("Did you vape yesterday? (1 = Yes, 0 = No): "))

    # 5. Predict Today
    input_data = np.array([[stress, trigger, prev_vaped]])
    prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of vaping
    prediction = model.predict(input_data)[0]  # 0 or 1

    # 6. Update Streak
    if prediction == 1:
        streak = 0
    else:
        streak += 1

    # 7. Show Results
    print("\n--- QuitMate Report ---")
    print("Chance of vaping today: {:.1f}%".format(prediction_prob * 100))
    print("Prediction: {}".format("You might vape today." if prediction == 1 else "You’re staying strong!"))
    print("Current streak: {} clean day(s)".format(streak))
    print("Motivation: " + get_motivation(reason))

# Run run_quitmate_day() to move on to the next day
