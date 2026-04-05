"""
tools/shariah_tools.py — Shariah compliance screening for PSX stocks
"""
from __future__ import annotations

from langchain_core.tools import tool

from config import KMI_30, SHARIAH_WATCHLIST, LOW_PRICE_THRESHOLD_PKR
from tools.data_tools import fetch_ohlcv, get_current_price


# ─── KMI Index Data ───────────────────────────────────────────────────────────
# Source: PSX KMI-30 and KMI All-Share Index
# Update periodically from: https://www.psx.com.pk/psx/resources/indices

# Sectors known to be non-Shariah (interest-based / prohibited)
NON_SHARIAH_SECTORS = {
    "CONVENTIONAL_BANKING", "INSURANCE", "LEASING", "MODARABA_QUESTIONABLE",
}

# Extended watchlist combining KMI-30 + KMI All-Share low-priced stocks
ALL_SHARIAH_TICKERS = set(KMI_30 + SHARIAH_WATCHLIST)

# Known non-compliant tickers (conventional banks, etc.)
NON_COMPLIANT = {
    "MCB", "HBL", "UBL", "ABL", "BAFL", "FABL", "SILK", "SNBL",
    "JSBL", "BAHL",   # conventional banks
    "JSCL", "AICL",  # conventional insurance
}

# Manual overrides — confirmed KMI All-Share compliant low-price stocks
KMI_ALLSHARE_LOW_PRICE = {
    "WTL", "CNERGY", "UNITY", "KEL", "PACE", "LOTCHEM",
    "MLCF", "PIBTL", "TELE", "PAEL", "SNGP", "SSGC",
    "FCCL", "KOHC", "CHCC", "PIOC", "DGKC", "ACPL",
    "KAPCO", "HUBC", "LPCL", "ENGRO",
}

COMPLIANCE_NOTES = {
    "WTL":    "World Telecom Ltd — KMI All-Share compliant. Telecom sector.",
    "CNERGY": "Cnergyico PK — KMI All-Share compliant. Energy/refining.",
    "UNITY":  "Unity Foods — KMI All-Share compliant. Food sector.",
    "KEL":    "K-Electric — KMI All-Share compliant. Power utility.",
    "PACE":   "PACE Pakistan — KMI All-Share compliant. Real estate.",
    "LOTCHEM":"Lotte Chemical — KMI All-Share compliant. Chemicals.",
    "MLCF":   "Maple Leaf Cement — KMI-30 compliant.",
    "PIBTL":  "Pioneer Cement — KMI All-Share compliant.",
    "SNGP":   "Sui Northern Gas — KMI All-Share compliant. Gas utility.",
    "OGDC":   "OGDC — KMI-30 compliant. Oil & Gas.",
    "PPL":    "PPL — KMI-30 compliant. Oil & Gas.",
    "KAPCO":  "Kapco — KMI-30 compliant. Power.",
    "HUBC":   "Hub Power — KMI-30 compliant. Power.",
}


# ─── Core Screening Function ─────────────────────────────────────────────────

def check_compliance(ticker: str) -> dict:
    """
    Return a compliance assessment for a PSX ticker.

    Screening logic (in order):
    1. If ticker is in NON_COMPLIANT set → NOT COMPLIANT
    2. If ticker is in KMI_ALLSHARE_LOW_PRICE or KMI_30 → COMPLIANT
    3. Otherwise → UNVERIFIED (user should check PSX KMI All-Share list manually)
    """
    ticker = ticker.upper().strip()

    if ticker in NON_COMPLIANT:
        return {
            "ticker":    ticker,
            "status":    "NOT_COMPLIANT",
            "reason":    "Conventional bank, insurance, or other interest-based entity.",
            "in_kmi30":  ticker in KMI_30,
            "note":      "Avoid for Shariah-compliant trading.",
            "price_pkr": None,
        }

    if ticker in KMI_ALLSHARE_LOW_PRICE or ticker in KMI_30:
        try:
            price = get_current_price(ticker)
        except Exception:
            price = None

        return {
            "ticker":    ticker,
            "status":    "COMPLIANT",
            "reason":    "Listed in KMI All-Share or KMI-30 Shariah-compliant index.",
            "in_kmi30":  ticker in KMI_30,
            "note":      COMPLIANCE_NOTES.get(ticker, "Verify latest KMI list on psx.com.pk"),
            "price_pkr": round(price, 2) if price else None,
            "low_price": price is not None and price <= LOW_PRICE_THRESHOLD_PKR,
        }

    # Unknown ticker
    return {
        "ticker":  ticker,
        "status":  "UNVERIFIED",
        "reason":  "Not found in local KMI screening database.",
        "in_kmi30": False,
        "note":    ("Manually verify at https://www.psx.com.pk → "
                    "Indices → KMI All-Share. "
                    "If listed there, it is Shariah-compliant."),
        "price_pkr": None,
    }


def get_compliant_watchlist(max_price: float = LOW_PRICE_THRESHOLD_PKR) -> list[dict]:
    """Return all confirmed compliant tickers with optional price filter."""
    results = []
    for ticker in sorted(KMI_ALLSHARE_LOW_PRICE):
        info = check_compliance(ticker)
        if info["status"] == "COMPLIANT":
            # Only include if price data is available and within threshold
            price = info.get("price_pkr")
            if price is None or price <= max_price:
                results.append(info)
    return results


# ─── LangChain Tool ───────────────────────────────────────────────────────────

@tool
def check_shariah(ticker: str) -> str:
    """
    Check whether a PSX stock ticker is Shariah-compliant (KMI-screened).

    Input: ticker symbol, e.g. "WTL" or "KEL"

    Returns compliance status, KMI index membership, current price,
    and guidance on verification.
    """
    info = check_compliance(ticker.strip().upper())

    icon = {"COMPLIANT": "✅", "NOT_COMPLIANT": "❌", "UNVERIFIED": "⚠️"}.get(
        info["status"], "❓"
    )

    lines = [
        f"{icon}  Shariah Compliance — {info['ticker']}",
        f"  Status   : {info['status']}",
        f"  In KMI-30: {'Yes' if info['in_kmi30'] else 'No'}",
        f"  Reason   : {info['reason']}",
        f"  Note     : {info['note']}",
    ]
    if info.get("price_pkr"):
        tag = "✅ LOW-PRICE" if info.get("low_price") else "ABOVE 10 PKR"
        lines.append(f"  Price    : {info['price_pkr']} PKR  ({tag})")

    return "\n".join(lines)


@tool
def list_shariah_watchlist(_: str = "") -> str:
    """
    List all Shariah-compliant, low-priced (<10 PKR) PSX stocks
    in the project watchlist.

    No input required (pass empty string or any text).
    """
    items = get_compliant_watchlist(max_price=LOW_PRICE_THRESHOLD_PKR)

    if not items:
        return "No low-priced compliant stocks found in local database."

    lines = ["📋  Shariah-Compliant Low-Price Watchlist  (≤10 PKR, KMI All-Share)"]
    for item in items:
        kmi = "KMI-30" if item["in_kmi30"] else "KMI-All"
        price_str = f"{item['price_pkr']} PKR" if item["price_pkr"] else "price N/A"
        lines.append(f"  • {item['ticker']:<10} [{kmi}]  {price_str}")

    lines += [
        "",
        "Tip: Verify the latest KMI All-Share list at psx.com.pk → Indices.",
    ]
    return "\n".join(lines)
