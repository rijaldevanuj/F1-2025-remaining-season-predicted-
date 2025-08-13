# F1-2025-remaining-season-predicted-
This machine learning model was created during the Formula 1 summer break—after the Hungarian Grand Prix and before the Dutch Grand Prix. It predicts:  Race winners and all point scorers for the remaining races of the 2025 season Final drivers’ and constructors’ championship standings The World Drivers’ Champion and Constructors’ Champion 

# F1 2025 Remaining Season Prediction

This project uses FastF1 and machine learning to predict the outcome of the 2025 Formula 1 season, including race winners, all point scorers, and final championship standings.  
Predictions are based on results up to the Hungarian GP (Race 14) and reflect the real grid and points situation at the summer break.

## Features

- Predicts winners and all point scorers for each remaining race
- Projects final drivers’ and constructors’ standings
- Estimates championship probabilities and margins
- Summarizes season trends and storylines

## Setup & Usage

### 1. Clone the repository

```powershell
git clone https://github.com/yourusername/F1-2025-remaining-season-predicted-.git
cd F1-2025-remaining-season-predicted-
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # For PowerShell
```
Or, if using Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create the FastF1 cache directory

```powershell
mkdir f1_cache
```

### 5. Run the prediction script

```powershell
python 2025f1.py
```

### 6. (Optional) View championship battle visualization

Uncomment the last lines in `2025f1.py` to display a plot of the predicted top 5 drivers.

## Notes

- Make sure you have Python 3.8+ installed.
- The script uses real 2025 results up to the Hungarian GP; predictions for the rest of the season are based on current form, track characteristics, and team development.
- All data is processed locally; no external API keys required.

## License

MIT
