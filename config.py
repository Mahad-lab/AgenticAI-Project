"""
config.py — Central configuration for PSX Trading Agent
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Provider ────────────────────────────────────────────────────────────
# Options: "groq" | "gemini" | "openai_compatible"
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq")

PROVIDER_CONFIGS = {
    "groq": {
        "model":       "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "temperature": 0,
    },
    "gemini": {
        "model":       "gemini-1.5-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "temperature": 0,
    },
    "openai_compatible": {
        "model":       os.getenv("OAI_MODEL", "gpt-4o-mini"),
        "api_key_env": "OPENAI_API_KEY",
        "base_url":    os.getenv("OAI_BASE_URL", "https://api.openai.com/v1"),
        "temperature": 0,
    },
}

# ─── Trading Parameters ───────────────────────────────────────────────────────
DEFAULT_CAPITAL_PKR = 5_000       # Starting capital in PKR
RISK_PER_TRADE_PCT  = 0.02        # Risk 2% of capital per trade
MAX_OPEN_POSITIONS  = 3
COMMISSION_PCT      = 0.001       # 0.1% per side (realistic PSX estimate)

# ─── Data Settings ────────────────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS = 365       # 1 year default backtest window
SWING_WINDOW          = 5         # Bars each side to confirm a swing H/L

# ─── Shariah-Compliant Watchlist ──────────────────────────────────────────────
# Low-priced, high-volume KMI-30 / KMI All-Share stocks (< ~10 PKR)
SHARIAH_WATCHLIST = [
    "WTL", "CNERGY", "UNITY", "KEL", "PACE", "LOTCHEM",
    "MLCF", "PIBTL", "TELE", "PAEL", "SNGP",
    "OGDC", "PPL", "HUBC", "KAPCO", "DGKC",
]

# KMI-30 constituents (update periodically from PSX website)
KMI_30 = [
    "OGDC", "PPL", "HUBC", "ENGRO", "LUCK", "MCB", "HBL",
    "UBL", "MEBL", "BAHL", "AKBL", "NBP", "FCCL", "MLCF",
    "DGKC", "KOHC", "PIOC", "CHCC", "SRVI", "ILP", "NESTLE",
    "UNITY", "CNERGY", "KEL", "KAPCO", "SNGP", "SSGC", "PAEL",
    "LOTCHEM", "PIBTL",
]

LOW_PRICE_THRESHOLD_PKR = 10

# ─── Fibonacci & Technical Levels ────────────────────────────────────────────
FIB_LEVELS = {
    "0.0":   0.000,
    "23.6":  0.236,
    "33.3":  0.333,   # Trisection
    "38.2":  0.382,
    "50.0":  0.500,   # Bisection
    "61.8":  0.618,   # Golden ratio (key buy zone)
    "66.7":  0.667,   # Trisection
    "78.6":  0.786,
    "100.0": 1.000,
}

# Strategy: buy signal zone + stop-loss level
BUY_ZONE_LOW  = 0.500   # 50% retracement
BUY_ZONE_HIGH = 0.618   # 61.8% retracement
STOP_LOSS_FIB = 0.786   # Stop loss below 78.6%
TAKE_PROFIT_FIB = 0.236 # Target: 23.6% retracement (near top)

RSI_PERIOD       = 14
RSI_OVERSOLD     = 40   # Slightly relaxed for PSX volatility
RSI_OVERBOUGHT   = 65
MA_SHORT         = 20
MA_LONG          = 50
VOLUME_MA_PERIOD = 20
