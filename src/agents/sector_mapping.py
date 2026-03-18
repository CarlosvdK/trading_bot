"""Sector and sub-industry mapping for agent universe filtering."""

from typing import Dict, List, Optional


# GICS Sector -> Ticker mapping (major US equities)
GICS_SECTORS: Dict[str, List[str]] = {
    "technology": [
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "TSM", "AVGO", "ORCL",
        "CSCO", "ADBE", "CRM", "AMD", "INTC", "TXN", "QCOM", "NOW", "SNOW",
        "SHOP", "SQ", "PLTR", "NET", "CRWD", "ZS", "PANW", "FTNT", "DDOG",
        "MDB", "TEAM", "HUBS", "WDAY", "VEEV", "ANSS", "CDNS", "SNPS", "KLAC",
        "LRCX", "AMAT", "MRVL", "ON", "NXPI", "MCHP", "ADI", "MPWR", "SWKS",
        "MU", "STX", "WDC", "HPQ", "DELL", "IBM", "ACN", "INFY", "EPAM",
    ],
    "healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "MRNA", "BIIB", "ILMN", "ISRG", "SYK",
        "BSX", "MDT", "ZBH", "EW", "DXCM", "ALGN", "IDXX", "VEEV",
        "CI", "HUM", "CVS", "MCK", "CAH",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "SCHW",
        "BLK", "SPGI", "ICE", "CME", "MCO", "MSCI", "CBOE", "NDAQ",
        "AXP", "V", "MA", "COF", "DFS", "SYF",
        "MET", "PRU", "AIG", "ALL", "TRV", "PGR", "CB", "AFL", "HIG",
        "BRK.B", "MMC", "AON", "WTW", "AJG",
    ],
    "energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD", "DVN",
        "OXY", "HAL", "BKR", "FANG", "HES", "MRO", "APA", "CTRA",
        "KMI", "WMB", "OKE", "ET", "EPD", "TRGP", "LNG",
        "ENPH", "SEDG", "FSLR", "RUN", "NEE",
    ],
    "consumer_discretionary": [
        "AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TJX", "LOW", "TGT",
        "ROST", "CMG", "DHI", "LEN", "PHM", "NVR",
        "GM", "F", "RIVN", "LCID",
        "MAR", "HLT", "WYNN", "LVS", "MGM",
        "DPZ", "YUM", "DARDEN", "BKNG", "ABNB", "EXPE",
        "LULU", "GPS", "ANF", "DECK",
    ],
    "consumer_staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
        "GIS", "K", "SJM", "HSY", "HRL", "CAG", "CPB", "TSN", "BG",
        "STZ", "TAP", "SAM", "DEO",
        "EL", "CHD", "CLX", "SPC",
    ],
    "industrials": [
        "HON", "UPS", "UNP", "BA", "CAT", "DE", "GE", "MMM", "RTX", "LMT",
        "NOC", "GD", "HII", "LHX", "TDG", "HWM",
        "EMR", "ROK", "ETN", "IR", "AME", "GNRC",
        "FDX", "CSX", "NSC", "JBHT", "CHRW",
        "WM", "RSG", "VRSK", "PAYC", "DAL", "UAL", "LUV", "AAL",
        "VMC", "MLM", "JCI", "CARR", "OTIS",
    ],
    "materials": [
        "LIN", "APD", "SHW", "ECL", "DD", "DOW", "PPG", "NEM", "FCX",
        "GOLD", "AEM", "NUE", "STLD", "CLF", "X", "AA", "TECK",
        "FMC", "MOS", "CF", "ALB", "CTVA", "IFF", "CE",
    ],
    "real_estate": [
        "PLD", "AMT", "CCI", "EQIX", "PSA", "DLR", "WELL", "AVB",
        "EQR", "SPG", "O", "VICI", "VTR", "ARE", "MAA",
        "UDR", "ESS", "CPT", "INVH", "SUI", "ELS",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC",
        "ES", "AEE", "CMS", "DTE", "FE", "PPL", "AWK", "ATO",
    ],
    "communication_services": [
        "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
        "CHTR", "EA", "ATVI", "TTWO", "RBLX", "U",
        "WBD", "PARA", "FOX", "FOXA", "LYV", "IACI", "MTCH", "SNAP", "PINS",
    ],
}

# Sub-industry mappings for specialist agents
SUB_INDUSTRIES: Dict[str, List[str]] = {
    "semiconductors": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "TXN", "QCOM", "MRVL", "ON", "NXPI", "MCHP", "ADI", "MPWR", "SWKS", "MU", "KLAC", "LRCX", "AMAT"],
    "cloud_saas": ["CRM", "NOW", "SNOW", "SHOP", "DDOG", "NET", "MDB", "TEAM", "HUBS", "WDAY", "VEEV", "ZS", "CRWD", "PANW", "FTNT"],
    "biotech": ["AMGN", "GILD", "VRTX", "REGN", "MRNA", "BIIB", "ILMN", "SGEN"],
    "pharma": ["JNJ", "PFE", "ABBV", "MRK", "LLY", "BMY", "AZN", "NVS", "GSK", "SNY"],
    "banks": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "SCHW"],
    "insurance": ["MET", "PRU", "AIG", "ALL", "TRV", "PGR", "CB", "AFL", "HIG", "BRK.B"],
    "oil_gas": ["XOM", "CVX", "COP", "EOG", "PXD", "DVN", "OXY", "MRO", "APA", "CTRA", "FANG", "HES"],
    "renewables": ["ENPH", "SEDG", "FSLR", "RUN", "NEE"],
    "defense": ["RTX", "LMT", "NOC", "GD", "HII", "LHX", "TDG"],
    "aerospace": ["BA", "GE", "HWM", "TDG", "HEI", "SPR", "ERJ"],
    "mining": ["NEM", "FCX", "GOLD", "AEM", "NUE", "STLD", "CLF", "X", "AA", "TECK"],
    "reits": ["PLD", "AMT", "CCI", "EQIX", "PSA", "DLR", "WELL", "SPG", "O", "VICI"],
    "retail": ["AMZN", "HD", "TJX", "LOW", "TGT", "ROST", "COST", "WMT", "LULU", "GPS", "ANF", "DECK"],
    "food_beverage": ["KO", "PEP", "MDLZ", "GIS", "K", "SJM", "HSY", "HRL", "CAG", "CPB", "STZ", "TAP", "SAM", "TSN"],
    "media": ["DIS", "CMCSA", "WBD", "PARA", "FOX", "FOXA", "LYV", "NFLX"],
    "telecom": ["T", "VZ", "TMUS", "CHTR"],
    "gaming": ["EA", "ATVI", "TTWO", "RBLX", "U", "WYNN", "LVS", "MGM"],
    "auto": ["TSLA", "GM", "F", "RIVN", "LCID"],
    "construction": ["DHI", "LEN", "PHM", "NVR", "VMC", "MLM", "JCI", "CARR", "OTIS"],
    "cyber_security": ["CRWD", "ZS", "PANW", "FTNT", "NET", "S"],
}

# Reverse lookup cache (built on import)
_TICKER_TO_SECTOR: Dict[str, str] = {}
_TICKER_TO_SUB: Dict[str, str] = {}

for _sector, _tickers in GICS_SECTORS.items():
    for _t in _tickers:
        _TICKER_TO_SECTOR[_t] = _sector

for _sub, _tickers in SUB_INDUSTRIES.items():
    for _t in _tickers:
        _TICKER_TO_SUB[_t] = _sub


def get_sector(symbol: str) -> Optional[str]:
    """Get GICS sector for a ticker symbol."""
    return _TICKER_TO_SECTOR.get(symbol)


def get_sub_industry(symbol: str) -> Optional[str]:
    """Get sub-industry for a ticker symbol."""
    return _TICKER_TO_SUB.get(symbol)


def get_symbols_for_sector(sector: str) -> List[str]:
    """Get all symbols in a given sector."""
    return GICS_SECTORS.get(sector, [])


def get_symbols_for_sub_industry(sub_industry: str) -> List[str]:
    """Get all symbols in a given sub-industry."""
    return SUB_INDUSTRIES.get(sub_industry, [])


def get_agent_universe(
    primary_sectors: List[str],
    secondary_sectors: List[str],
) -> Dict[str, float]:
    """
    Build a weighted symbol universe for an agent.
    Returns {symbol: weight} where primary=1.0, secondary=0.5.
    """
    universe: Dict[str, float] = {}
    for sector in primary_sectors:
        for sym in get_symbols_for_sector(sector):
            universe[sym] = 1.0
        # Also check sub-industries
        for sym in get_symbols_for_sub_industry(sector):
            universe[sym] = 1.0
    for sector in secondary_sectors:
        for sym in get_symbols_for_sector(sector):
            if sym not in universe:
                universe[sym] = 0.5
        for sym in get_symbols_for_sub_industry(sector):
            if sym not in universe:
                universe[sym] = 0.5
    return universe
