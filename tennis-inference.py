# %% [markdown]
# ## Environment Setup

# %%
import dotenv

dotenv.load_dotenv()


# %% [markdown]
# ## Imports Stuff

# %%
import hopsworks
import pandas as pd
from xgboost import XGBClassifier

# %% [markdown]
# ## Connect to Hopsworks and Load Data

# %%
project = hopsworks.login(project="ATP_Tennis_Prediction")
fs = project.get_feature_store()

feature_view = fs.get_feature_view(
    name="tennis_match_prediction",
    version=1,
)

# %%
mr = project.get_model_registry()

retrieved_model = mr.get_model(
    name="tennis_match_predictor",  # ← Your model name
    version=1,
)

saved_model_dir = retrieved_model.download()

model = XGBClassifier()
model.load_model(saved_model_dir + "/model.json")


# %% [markdown]
# ### Holdout
# currently no new matches happening yet until January

# %%
# Get ALL matches from main feature group
tennis_fg = fs.get_feature_group(
    name="tennis_matches",
    version=2,
)

all_matches = tennis_fg.read()

# Filter for "upcoming" (pretend 2025 hasn't happened)
all_matches["date"] = pd.to_datetime(all_matches["date"])
df_new = all_matches[all_matches["date"].dt.year == 2025].copy()


# %% [markdown]
# ### Upcoming Matches

# %%
# upcoming_fg = fs.get_feature_group(
#     name="tennis_upcoming_matches",
#     version=1,
# )

# df_new = upcoming_fg.read()


# %% [markdown]
# ## Update Model

# %%
drop_list = [
    # Metadata
    "date",
    "player_1",
    "player_2",
    "winner",
    "timestamp",
    # Categorical text (have encoded versions)
    "tournament",
    "surface",
    "series",
    "round",
    "court",
    "tournament_clean",
]

# Also drop target (we're predicting it!)
drop_list.append("player_1_won")

# Filter for columns that exist
cols_to_drop = [col for col in drop_list if col in df_new.columns]

# Create features
features = df_new.drop(cols_to_drop, axis=1)

print(f"Features shape: {features.shape}")
print(f"Features columns: {features.columns.tolist()}")

# Verify no missing values
if features.isnull().sum().sum() > 0:
    print("Missing values found!")
    features = features.fillna(0)


# %% [markdown]
# ## Prediction

# %%
# Predict match outcomes
predictions = model.predict(features)

# Get probabilities
prediction_probabilities = model.predict_proba(features)
prob_player1_wins = prediction_probabilities[:, 1]  # P(Player 1 wins)

print(f"Predictions made: {len(predictions)}")
print(f"Player 1 wins predicted: {predictions.sum()} ({predictions.mean():.1%})")


# %%
results = df_new[["date", "player_1", "player_2"]].copy()

# Add predictions
results["predicted_player_1_wins"] = predictions
results["player_1_win_probability"] = prob_player1_wins

results["predicted_winner"] = results.apply(
    lambda row: row["player_1"]
    if row["predicted_player_1_wins"] == 1
    else row["player_2"],
    axis=1,
)

results["confidence"] = [x if x > 0.5 else 1 - x for x in prob_player1_wins]

# Validation
if "player_1_won" in df_new.columns:
    results["actual_winner"] = df_new.apply(
        lambda row: row["player_1"] if row["player_1_won"] == 1 else row["player_2"],
        axis=1,
    )
    results["correct"] = results["predicted_winner"] == results["actual_winner"]

    accuracy = results["correct"].mean()
    print(f"Validation Accuracy: {accuracy:.2%}")

results.head(20)

# %%
results.to_csv("tennis_predictions.csv", index=False, sep=";")
print(f"Saved {len(results)} predictions to tennis_predictions.csv")

# %%
