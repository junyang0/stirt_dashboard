import re
import time
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from io import StringIO
from bs4 import BeautifulSoup
from calendar import monthrange
from urllib.parse import unquote
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.188 Safari/537.36"
}
BARCHART_URL = "https://www.barchart.com/proxies/core-api/v1/historical/get?"
MAX_WORKERS = 6

# Mapping dictionaries
UNDERLYING_MAP = {
    "FED_FUNDS": "USA",
    "ESTR": "Eurozone",
    "CORRA": "Canada",
    "SFE_BA": "Australia",
    "SONIA": "United Kingdom",
    "SARON": "Switzerland",
    "TONA": "Japan"
}

POLICY_RATE_MAP = {
    "FED_FUNDS": "Fed Funds",
    "ESTR": "ECB Deposit Rate",
    "SONIA": "Bank of England Base Rate",
    "CORRA": "Bank of Canada Overnight Rate",
    "SARON": "Swiss National Bank Policy Rate",
    "SFE_BA": "Reserve Bank of Australia Cash Rate",
    "TONA": "Bank of Japan Policy Rate"
}

CB_MAP = {
    "FED_FUNDS": "FED",
    "ESTR": "ECB",
    "SONIA": "BOE",
    "CORRA": "BOC",
    "SARON": "SNB",
    "SFE_BA": "RBA",
    "TONA": "BOJ"
}

# --- Data Fetching Functions ---

def get_upcoming_fomc_dates(url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"):
    """Scrape upcoming FOMC meeting dates from the Federal Reserve website."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        meetings = []
        today = datetime.today()

        for panel in soup.select(".panel.panel-default"):
            year_heading = panel.select_one(".panel-heading h4")
            if not year_heading:
                continue
            year = year_heading.text.strip().split(":")[-1].strip().split("FOMC")[0].strip()

            for row in panel.select(".fomc-meeting"):
                month_str = row.select_one(".fomc-meeting__month strong").text.strip().split("/")[0].strip()
                day_str = row.select_one(".fomc-meeting__date").text.strip().split("*")[0].split('-')[-1].strip()

                for fmt in ("%b %d %Y", "%B %d %Y"):
                    try:
                        date = datetime.strptime(f"{month_str} {day_str} {year}", fmt)
                        if date >= today:
                            meetings.append(date.strftime("%Y-%m-%d"))
                        break
                    except ValueError:
                        continue
                else:
                    print(f"Unrecognized date format: {month_str} {day_str} {year}")
        return pd.Series(meetings, name="FOMC_Date")
    except Exception as e:
        print(f"Error fetching FOMC dates: {e}")
        return pd.Series([], name="FOMC_Date")

def get_upcoming_ecb_monetary_dates(url="https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"):
    """Scrape upcoming ECB monetary policy meeting dates."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        meetings = []
        today = datetime.today()

        for dt, dd in zip(soup.select("div.definition-list dl dt"), soup.select("div.definition-list dl dd")):
            date_str = dt.text.strip()
            desc = dd.text.strip()
            if "monetary policy meeting" in desc.lower() and "day 2" in desc.lower():
                try:
                    date = datetime.strptime(date_str, "%d/%m/%Y")
                    if date >= today:
                        meetings.append(date.strftime("%Y-%m-%d"))
                except ValueError:
                    print(f"Unrecognized date format: {date_str}")
        return pd.Series(meetings, name="ECB_Date")
    except Exception as e:
        print(f"Error fetching ECB dates: {e}")
        return pd.Series([], name="ECB_Date")

def get_upcoming_mpc_dates(url="https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates"):
    """Scrape upcoming Bank of England MPC meeting dates."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        meetings = []
        today = datetime.today()

        for table in soup.select("table"):
            year_heading = table.find_previous(["h2", "h3", "h4"])
            year = int(re.search(r"\d{4}", year_heading.text).group()) if year_heading and re.search(r"\d{4}", year_heading.text) else today.year

            for row in table.select("tbody tr"):
                tds = row.find_all("td")
                if len(tds) != 2:
                    continue
                date_str = tds[0].text.strip()
                try:
                    date = datetime.strptime(f"{date_str} {year}", "%A %d %B %Y")
                    if date >= today:
                        meetings.append(date.strftime("%Y-%m-%d"))
                except ValueError:
                    print(f"Unrecognized date format: {date_str} with year {year}")
        return pd.Series(meetings, name="MPC_Date")
    except Exception as e:
        print(f"Error fetching MPC dates: {e}")
        return pd.Series([], name="MPC_Date")

def get_upcoming_boc_dates(pages=3, url="https://www.bankofcanada.ca/press/upcoming-events/"):
    """Scrape upcoming Bank of Canada interest rate and monetary policy events."""
    try:
        events = []
        today = datetime.today()

        for page in range(1, pages + 1):
            page_url = url if page == 1 else f"{url}?mt_page={page}"
            response = requests.get(page_url, headers=DEFAULT_HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            for media in soup.select(".media-body"):
                date_span = media.select_one(".media-date")
                h3 = media.select_one("h3.media-heading a")
                if not date_span or not h3:
                    continue
                date_str = date_span.text.strip()
                desc = h3.text.strip()
                if "monetary policy" in desc.lower() or "interest rate" in desc.lower():
                    try:
                        date = datetime.strptime(date_str, "%B %d, %Y")
                        if date >= today:
                            events.append(date.strftime("%Y-%m-%d"))
                    except ValueError:
                        print(f"Unrecognized date format: {date_str}")
        return pd.Series(events, name="BOC_Date")
    except Exception as e:
        print(f"Error fetching BOC dates: {e}")
        return pd.Series([], name="BOC_Date")

def get_upcoming_snb_dates(url="https://www.snb.ch/en/services-events/digital-services/event-schedule"):
    """Scrape upcoming Swiss National Bank monetary policy events."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        events = []
        today = datetime.today()

        for li in soup.select("li.col-12.link-list-item"):
            title_tag = li.select_one(".publication-title h3")
            date_tag = li.select_one(".publication-date")
            if not title_tag or not date_tag or "monetary policy" not in title_tag.text.lower():
                continue
            date_str = date_tag.text.strip()
            try:
                date = datetime.strptime(date_str, "%d.%m.%Y")
                if date >= today:
                    events.append(date.strftime("%Y-%m-%d"))
            except ValueError:
                print(f"Unrecognized date format: {date_str}")
        return pd.Series(events, name="SNB_Date")
    except Exception as e:
        print(f"Error fetching SNB dates: {e}")
        return pd.Series([], name="SNB_Date")

def get_rba_board_meeting_dates(url="https://www.rba.gov.au/schedules-events/board-meeting-schedules.html"):
    """Scrape upcoming Reserve Bank of Australia board meeting dates."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table", class_="table-linear")
        current_year = datetime.today().year
        meetings = []
        month_map = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June",
                                                 "July", "August", "September", "October", "November", "December"], 1)}

        for table in tables:
            caption = table.find("caption")
            if not caption:
                continue
            year_match = re.search(r"(\d{4})", caption.text)
            if not year_match or int(year_match.group(1)) not in [current_year, current_year + 1]:
                continue
            year = int(year_match.group(1))

            for row in table.tbody.find_all("tr"):
                cols = row.find_all(["th", "td"])
                if len(cols) < 2:
                    continue
                month_name = cols[0].text.strip()
                month_num = month_map.get(month_name)
                if not month_num:
                    continue
                dates_text = re.sub(r'\s*\(.*?\)', '', cols[1].text.strip())
                match_range = re.findall(r'(\d+)[–-](\d+)\s*(\w+)?', dates_text)
                if match_range:
                    for start, end, month_override in match_range:
                        month_final = month_map.get(month_override, month_num) if month_override else month_num
                        date = datetime.strptime(f"{end} {month_final} {year}", "%d %m %Y")
                        if date >= datetime.today():
                            meetings.append(date)
                else:
                    day_match = re.search(r'(\d+)\s*(\w+)?', dates_text)
                    if day_match:
                        day = day_match.group(1)
                        month_final = month_map.get(day_match.group(2), month_num) if day_match.group(2) else month_num
                        date = datetime.strptime(f"{day} {month_final} {year}", "%d %m %Y")
                        if date >= datetime.today():
                            meetings.append(date)
        df = pd.DataFrame(sorted(meetings), columns=["Date"]).dt.strftime("%Y-%m-%d")
        return pd.Series(df["Date"], name="RBA_Date")
    except Exception as e:
        print(f"Error fetching RBA dates: {e}")
        return pd.Series([], name="RBA_Date")

def get_boj_meeting_dates(url="https://www.mnimarkets.com/calendars/bank-of-japan-meeting-calendar"):
    """Scrape upcoming Bank of Japan meeting dates."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        meetings = []

        for list_div in soup.select("div.list"):
            year = int(list_div.find("h2").text.strip().split()[-1])
            for li in list_div.select("ul li"):
                text = li.get_text(strip=True)
                if "Meeting:" not in text:
                    continue
                date_str = text.replace("Meeting:", "").strip()
                if '-' in date_str:
                    start_part, end_part = date_str.split('-')
                    start_part, end_part = start_part.strip(), end_part.strip()
                    try:
                        start_date = datetime.strptime(f"{start_part.split(',')[0]} {year}", "%b %d %Y")
                        end_date = datetime.strptime(f"{end_part} {year}", "%b %d %Y") if any(c.isalpha() for c in end_part) else start_date.replace(day=int(end_part))
                    except ValueError:
                        continue
                else:
                    try:
                        start_date = end_date = datetime.strptime(f"{date_str} {year}", "%b %d %Y")
                    except ValueError:
                        continue
                if end_date >= datetime.today():
                    meetings.append(start_date.strftime("%Y-%m-%d"))
        return pd.Series(meetings, name="BOJ_Date")
    except Exception as e:
        print(f"Error fetching BOJ dates: {e}")
        return pd.Series([], name="BOJ_Date")

def get_central_bank_rates(url="https://www.cbrates.com/"):
    """Fetch current central bank policy rates."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        data = []

        for row in soup.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 6:
                rate = cols[1].get_text(strip=True).replace('%', '')
                change = cols[2].get_text(strip=True)
                country = cols[4].get_text(strip=True).split("|")[0].strip()
                change_date = cols[5].get_text(strip=True)
                data.append([country, rate, change, change_date])
        df = pd.DataFrame(data, columns=["Country", "Rate", "Change", "Date"]).set_index("Country")
        return df
    except Exception as e:
        print(f"Error fetching central bank rates: {e}")
        return pd.DataFrame(columns=["Rate", "Change", "Date"]).set_index("Country")

def get_barchart_ticker_name(interest_rate, tenor, month, year):
    """Generate Barchart ticker name for a given interest rate, tenor, month, and year."""
    month_codes = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    ticker_prefix = {
        "FED_FUNDS": {"1M": "ZQ"},
        "ESTR": {"1M": "EG"},
        "TONA": {"3M": "T0"},
        "SONIA": {"1M": "JU"},
        "CORRA": {"1M": "RI", "3M": "RG"},
        "SARON": {"3M": "J2"},
        "SFE_BA": {"1M": "IQ"}
    }
    if interest_rate not in ticker_prefix or tenor not in ticker_prefix[interest_rate]:
        raise ValueError(f"Unsupported interest rate or tenor: {interest_rate}, {tenor}")
    if month not in month_codes or not (2000 <= year <= 2099):
        raise ValueError("Invalid month or year")
    return f"{ticker_prefix[interest_rate][tenor]}{month_codes[month]}{str(year)[-2:]}"

def get_implied_rate(price):
    """Calculate implied rate from futures price."""
    return 100 - price

def get_data(ticker, url=BARCHART_URL, max_retries=3, sleep_sec=5):
    """Fetch historical data for a given ticker from Barchart API."""
    for attempt in range(1, max_retries + 1):
        try:
            with requests.Session() as req:
                req.headers.update(DEFAULT_HEADERS)
                r = req.get(url[:25])
                xsrf_token = r.cookies.get('XSRF-TOKEN')
                if not xsrf_token:
                    raise ValueError("No XSRF-TOKEN found")
                req.headers.update({'X-XSRF-TOKEN': unquote(xsrf_token)})
                params = {
                    'symbol': ticker,
                    'fields': 'tradeTime.format(m/d/Y),openPrice,highPrice,lowPrice,lastPrice,priceChange,percentChange,volume,openInterest,symbolCode,symbolType',
                    'type': 'eod',
                    'orderBy': 'tradeTime',
                    'orderDir': 'desc',
                    'limit': '65',
                    'meta': 'field.shortName,field.type,field.description',
                    'raw': '1'
                }
                r = req.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                df = pd.DataFrame(data['data']).iloc[:, :-1]
                df.index = pd.to_datetime(df['tradeTime'])
                return df
        except (requests.RequestException, ValueError, KeyError, requests.exceptions.JSONDecodeError) as e:
            print(f"[Attempt {attempt}] Failed to fetch {ticker}: {e}")
            if attempt < max_retries:
                time.sleep(sleep_sec)
            else:
                print(f"All {max_retries} attempts failed for {ticker}")
                return pd.DataFrame()

def fetch_single(ticker, underlying, tenor, month, year):
    """Fetch and format data for a single ticker."""
    try:
        print(f"Fetching data for {ticker}")
        df = get_data(ticker)
        if df.empty:
            return df
        df = df[['lastPrice']].copy()
        df['underlying'] = underlying
        df['tenor'] = tenor
        df['contract_name'] = ticker
        df['monthYear'] = ticker[2:]
        df['month'] = month
        df['year'] = year
        df['period'] = datetime(year, month, 1)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_database(underlyings, start_month, year, n_contracts, max_workers=MAX_WORKERS):
    """Fetch futures data for multiple underlyings and tenors."""
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for underlying, tenor in underlyings:
            step = 3 if tenor == "3M" else 1
            n_contracts_adj = 12 if tenor == "3M" else n_contracts
            adj_start_month = start_month + (3 - start_month % 3) if tenor == "3M" and start_month % 3 != 1 else start_month
            for i in range(0, n_contracts_adj, step):
                abs_month = adj_start_month + i
                year_offset, month = divmod(abs_month - 1, 12)
                month += 1
                current_year = year + year_offset
                ticker = get_barchart_ticker_name(underlying, tenor, month, current_year)
                tasks.append(executor.submit(fetch_single, ticker, underlying, tenor, month, current_year))
        results = [future.result() for future in as_completed(tasks)]
    if not results:
        return pd.DataFrame()
    df = pd.concat(results, axis=0)
    df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce')
    df['implied_rate'] = df['lastPrice'].apply(get_implied_rate)
    return df.sort_values(by=['year', 'month'], ascending=True)

def get_latest_data(db, month, year):
    """Get the most recent data for a specific month and year."""
    return db[(db['month'] == month) & (db['year'] == year)].sort_index().iloc[-1]

def get_matrix(db, underlying, tenor, start_month_and_year, n_months, price_or_rate='rate'):
    """Generate a difference matrix for implied rates or prices."""
    df = db[(db['underlying'] == underlying) & (db['tenor'] == tenor)].copy()
    matrix = pd.DataFrame()
    current_month, current_year = start_month_and_year
    for i in range(n_months):
        month = current_month + i
        year = current_year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        try:
            latest = get_latest_data(df, month, year)
            matrix = pd.concat([matrix, latest.to_frame().T], axis=0)
        except Exception as e:
            print(f"Error for {underlying} {tenor} {month} {year}: {e}")
    if matrix.empty:
        return pd.DataFrame()
    matrix.index = matrix['contract_name']
    values = matrix['implied_rate'] if price_or_rate == 'rate' else matrix['lastPrice']
    diff_matrix = pd.DataFrame(values.values.reshape(-1, 1) - values.values.reshape(1, -1), index=matrix.index, columns=matrix.index)
    return diff_matrix

def get_latest(db, underlying, tenor, start_month_and_year, n_months, price_or_rate='rate'):
    """Generate a DataFrame of latest contract values."""
    df = db[(db['underlying'] == underlying) & (db['tenor'] == tenor)].copy()
    matrix = pd.DataFrame()
    current_month, current_year = start_month_and_year
    for i in range(n_months):
        month = current_month + i
        year = current_year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        try:
            latest = get_latest_data(df, month, year)
            matrix = pd.concat([matrix, latest.to_frame().T], axis=0)
        except Exception as e:
            print(f"Error for {underlying} {tenor} {month} {year}: {e}")
    if matrix.empty:
        return pd.DataFrame()
    matrix['expiry'] = matrix.apply(lambda row: f"{int(row['month']):02d}/{row['year']}", axis=1)
    matrix.index = pd.to_datetime(matrix.index).strftime('%Y-%m-%d')
    return matrix[['contract_name', 'expiry', 'lastPrice', 'implied_rate']]

def plot_df(df, title="Time Series", ylabel="Value", xlabel="Date"):
    """Plot time series data using Altair."""
    df.index = pd.to_datetime(df.index)
    df = df.reset_index(names=[xlabel])
    df_long = df.melt(id_vars=[xlabel], var_name="Contract", value_name=ylabel)
    legend_selection = alt.selection_point(fields=["Contract"], bind="legend")
    chart = (
        alt.Chart(df_long)
        .mark_line()
        .encode(
            x=alt.X(xlabel, title=xlabel),
            y=alt.Y(ylabel, title=ylabel, scale=alt.Scale(zero=False)),
            color=alt.Color("Contract", sort=list(df_long["Contract"].unique()), legend=alt.Legend(title="Contract")),
            tooltip=[xlabel, "Contract", ylabel],
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.1))
        )
        .properties(title=title, width=700, height=400)
        .add_params(legend_selection)
    )
    st.altair_chart(chart, use_container_width=True)

def fetch_fed_probabilities(url="https://www.investing.com/central-banks/fed-rate-monitor"):
    """Fetch implied rate change probabilities for FOMC meetings."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        data = []
        for meeting in soup.find_all('div', class_='cardWrapper'):
            date_div = meeting.find('div', class_='fedRateDate')
            meeting_date = date_div.text.strip() if date_div else "Unknown"
            for p in meeting.find_all('div', class_='percfedRateItem'):
                spans = p.find_all('span')
                if len(spans) >= 2:
                    data.append({
                        'Meeting Date': pd.to_datetime(meeting_date),
                        'Rate': spans[0].text.strip(),
                        'Probability': spans[-1].text.strip().replace('%', ''),
                        'Central Bank': 'FED',
                        'Source': url
                    })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching FED probabilities: {e}")
        return pd.DataFrame()

def fetch_ecb_probabilities(url="https://ecb-watch.eu/probabilities"):
    """Fetch implied rate change probabilities for ECB meetings."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['abs_data'])
        records = []
        for meeting_date, row in df.T.iterrows():
            for rate, prob in row.items():
                for offset in [0, 1]:
                    records.append({
                        'Meeting Date': pd.to_datetime(meeting_date) + pd.DateOffset(days=offset),
                        'Rate': rate,
                        'Probability': prob,
                        'Central Bank': 'ECB',
                        'Source': url
                    })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error fetching ECB probabilities: {e}")
        return pd.DataFrame()

# --- Streamlit Dashboard ---

st.set_page_config(layout="wide", page_title="STIRT — Interest Rate Dashboard")
st.title("STIRT Dashboard")

# Cache meeting dates
@st.cache_data(ttl=3600 * 12)
def get_all_meeting_dates():
    """Cache all central bank meeting dates."""
    return {
        "FED": get_upcoming_fomc_dates().tolist(),
        "ECB": get_upcoming_ecb_monetary_dates().tolist(),
        "BOE": get_upcoming_mpc_dates().tolist(),
        "BOC": get_upcoming_boc_dates().tolist(),
        "SNB": get_upcoming_snb_dates().tolist(),
        "RBA": get_rba_board_meeting_dates().tolist(),
        "BOJ": get_boj_meeting_dates().tolist()
    }

# Initialize session state
if "db" not in st.session_state:
    underlyings = [
        ('FED_FUNDS', '1M'), ('ESTR', '1M'), ('TONA', '3M'), ('SONIA', '1M'),
        ('SARON', '3M'), ('CORRA', '1M'), ('SFE_BA', '1M')
    ]
    start_date = datetime.now()
    st.session_state.db = get_database(underlyings, start_date.month, start_date.year, n_contracts=6, max_workers=MAX_WORKERS)
if "meeting_dates" not in st.session_state:
    st.session_state.meeting_dates = get_all_meeting_dates()

db = st.session_state.db
meeting_dates = st.session_state.meeting_dates
df_rates = get_central_bank_rates()

# Sidebar
with st.sidebar:
    underlying_rate = st.selectbox("Choose Underlying Rate", db['underlying'].unique())
    price_or_rate = st.selectbox("Plot Price or Yield", ["Price", "Yield"], index=1)
    mat_tenor = st.selectbox("1M/3M", db[db['underlying'] == underlying_rate]['tenor'].unique())
    with st.expander("Data Sources", expanded=False):
        st.markdown("""
        - [Barchart](https://www.barchart.com/futures/quotes/ZQ*0/futures-prices)
        - [CBRates](https://www.cbrates.com/)
        - [Investing.com FED Rate Monitor](https://www.investing.com/central-banks/fed-rate-monitor)
        - [ECB Watch](https://ecb-watch.eu/probabilities)
        - [FOMC Dates](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
        - [ECB Meeting Dates](https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html)
        - [BOE MPC Dates](https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates)
        - [BOC Upcoming Events](https://www.bankofcanada.ca/press/upcoming-events/)
        - [SNB Event Schedule](https://www.snb.ch/en/services-events/digital-services/event-schedule)
        - [RBA Board Meeting Schedules](https://www.rba.gov.au/schedules-events/board-meeting-schedules.html)
        - [BOJ Meeting Calendar](https://www.mnimarkets.com/calendars/bank-of-japan-meeting-calendar)
        """)

# Main Dashboard with Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Time Series", "Contract Values", "Difference Matrix", "Meetings & Probabilities"])

with tab1:
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #1e3a8a;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] label[data-testid="stMetricLabel"],
    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: white !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    country = UNDERLYING_MAP.get(underlying_rate, "")
    if country and country in df_rates.index:
        policy_info = df_rates.loc[country]
        country_display = "EU" if country == "Eurozone" else "US" if country == "USA" else country
        st.subheader(f"Current Policy Rate: {POLICY_RATE_MAP.get(underlying_rate, underlying_rate)}")
        col1, col2, col3 = st.columns(3, gap="small")
        col1.metric(label="Country", value=country_display)
        col2.metric(label="Rate (%)", value=policy_info['Rate'])
        col3.metric(label="Last Change", value=policy_info['Change'])
        st.caption(f"Last Change Date: {policy_info['Date'][:-4] + ' ' + policy_info['Date'][-4:]}")

with tab2:
    series_choice = "implied_rate" if price_or_rate == "Yield" else "lastPrice"
    contract_df = db[db['underlying'] == underlying_rate]
    to_plot = [contract_df[contract_df['contract_name'] == i][series_choice].rename(i) for i in contract_df['contract_name'].unique()]
    if to_plot:
        to_plot_df = pd.concat(to_plot, axis=1)
        plot_df(to_plot_df, title=f"{underlying_rate} ({price_or_rate})", ylabel=price_or_rate, xlabel="Date")

with tab3:
    st.subheader(f"Latest Contract Values as of {datetime.now().strftime('%Y-%m-%d')}")
    n_contracts_adj = 12 if mat_tenor == '3M' else 6
    latest_df = get_latest(db, underlying_rate, mat_tenor, (datetime.now().month, datetime.now().year), n_contracts_adj, price_or_rate.lower())
    if not latest_df.empty:
        st.dataframe(latest_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Difference Matrix")
    adj_start_month = datetime.now().month + (3 - datetime.now().month % 3) if mat_tenor == '3M' and datetime.now().month % 3 != 1 else datetime.now().month
    diff_df = get_matrix(db, underlying_rate, mat_tenor, (adj_start_month, datetime.now().year), n_contracts_adj, price_or_rate.lower())
    if not diff_df.empty:
        def color_pos_neg(val):
            if pd.isna(val):
                return ''
            color = 'green' if val > 0 else 'red' if val < 0 else ''
            return f'color: {color}'
        styled_df = diff_df.replace(0, np.nan).style.applymap(color_pos_neg).format("{:.2f}")
        st.dataframe(styled_df, use_container_width=True)

with tab5:
    st.subheader("Upcoming Meetings")
    meetings = meeting_dates.get(CB_MAP.get(underlying_rate, ""), [])
    if meetings:
        st.dataframe(pd.DataFrame(meetings.head(5), columns=["Meeting Date"]), use_container_width=True, hide_index=True)

    if underlying_rate in CB_MAP and CB_MAP[underlying_rate] in ["FED", "ECB"]:
        combined_df = pd.concat([fetch_fed_probabilities(), fetch_ecb_probabilities()], ignore_index=True)
        combined_df['Probability'] = combined_df['Probability'].astype(float) / 100.0
        combined_df['MeetingStr'] = combined_df['Meeting Date'].dt.strftime('%Y-%m-%d')
        combined_df['Underlying'] = combined_df['Central Bank'].map({v: k for k, v in CB_MAP.items()})

        st.subheader("Implied Rate Change Probabilities")
        meeting = st.selectbox("Select Meeting Date", meetings)
        filtered_df = combined_df[(combined_df['Central Bank'] == CB_MAP.get(underlying_rate)) & (combined_df['MeetingStr'] == meeting)]
        if not filtered_df.empty:
            chart = (
                alt.Chart(filtered_df)
                .mark_bar()
                .encode(
                    x=alt.X("Rate:N", title="Rate Change (%)"),
                    y=alt.Y("Probability:Q", title="Implied Probability", axis=alt.Axis(format='%')),
                    color="Rate:N",
                    tooltip=["Central Bank", "Underlying", "MeetingStr", alt.Tooltip("Probability", format=".0%")]
                )
                .properties(width=300, height=400)
            )
            st.altair_chart(chart, use_container_width=True)