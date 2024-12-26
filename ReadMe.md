
# Create virtual Environments
    => python -m venv .venv

# Activate python venv 
    => .venv\Scipts\acticate

# First Before run the project please Type 
    => pip install -r requirements.txt

# After install dependencies type this to Run training Process
# I have prepared four user "alexa" "rasy_huot" "sarat-rorng"  "sokharoth" "sovannmolika"
    python train_and_authenticate.py <user_name>
# Example :
    => python train_and_authenticate.py alexa

# After done training it will save the model to the directory "models"
    model_path = "models/voice_auth_model.h5", "models/owner_centroid.h5"

# when it finished authentication it will save the result into the file excel 
    results_path = results/