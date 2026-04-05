from tools.data_tools      import get_stock_summary
from tools.technical_tools import analyze_technical
from tools.backtest_tools  import run_backtest
from tools.shariah_tools   import check_shariah, list_shariah_watchlist

ALL_TOOLS = [
    get_stock_summary,
    analyze_technical,
    run_backtest,
    check_shariah,
    list_shariah_watchlist,
]
