import re
import time 
import requests
import functools
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

DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
}
BARCHART_URL = "https://www.barchart.com/proxies/core-api/v1/historical/get?"
MAX_WORKERS = 6

def get_upcoming_fomc_dates(url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"):
    """
    Scrape the Federal Reserve FOMC calendar page and return a Pandas Series
    of upcoming FOMC meeting dates (YYYY-MM-DD).
    """
    response = requests.get(url)
    response.raise_for_status()  # raise error if request fails
    soup = BeautifulSoup(response.content, "html.parser")

    meetings = []
    today = datetime.today()

    for panel in soup.select(".panel.panel-default"):
        year_heading = panel.select_one(".panel-heading h4")
        if year_heading:
            year_text = year_heading.text.strip()
            year = year_text.split(":")[-1].strip().split("FOMC")[0].strip()
            
            for row in panel.select(".fomc-meeting"):
                month_str = row.select_one(".fomc-meeting__month strong").text.strip()
                if "/" in month_str:
                    month_str = month_str.split("/")[0].strip()
                day_str = row.select_one(".fomc-meeting__date").text.strip().split("*")[0]
                if '-' in day_str:
                    day_str = day_str.split('-')[1].strip()
                
                # Try parsing month in abbreviated and full form
                for fmt in ("%b %d %Y", "%B %d %Y"):
                    try:
                        date = datetime.strptime(f"{month_str} {day_str} {year}", fmt)
                        break
                    except ValueError:
                        continue
                else:
                    print(f"Unrecognized date format: {month_str} {day_str} {year}")
                    continue
                print(f"Found meeting date: {date.strftime('%Y-%m-%d')}")
                if date >= today:
                    meetings.append(date)

    # Convert to Pandas Series with YYYY-MM-DD format
    return pd.Series([d.strftime("%Y-%m-%d") for d in meetings], name="FOMC_Date")

def get_upcoming_ecb_monetary_dates(url="https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"):
    """
    Scrape the ECB calendar page and return a Pandas Series
    of upcoming monetary policy meeting dates (YYYY-MM-DD).
    """
    response = requests.get(url)
    response.raise_for_status()  # raise error if request fails
    soup = BeautifulSoup(response.content, "html.parser")

    meetings = []
    today = datetime.today()

    # Iterate over all <dt>/<dd> pairs in the definition list
    dts = soup.select("div.definition-list dl dt")
    dds = soup.select("div.definition-list dl dd")

    for dt, dd in zip(dts, dds):
        date_str = dt.text.strip()  # e.g., "04/02/2026"
        desc = dd.text.strip()

        # Only include monetary policy meetings
        if "monetary policy meeting" in desc.lower() and "day 2" in desc.lower():
            try:
                date = datetime.strptime(date_str, "%d/%m/%Y")
            except ValueError:
                print(f"Unrecognized date format: {date_str}")
                continue

            if date >= today:
                meetings.append({
                    "Date": date.strftime("%Y-%m-%d"),
                })
                print(f"Found ECB monetary policy meeting: {date.strftime('%Y-%m-%d')} - {desc}")

    df = pd.DataFrame(meetings)
    return pd.Series(df['Date'], name="ECB_Date")

def get_upcoming_mpc_dates(url="https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates"):
    """
    Scrape the Bank of England MPC calendar page for all tables (multiple years) and return
    a DataFrame of upcoming MPC meeting dates (YYYY-MM-DD) with descriptions and links.
    """
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/116.0.5845.188 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    meetings = []
    today = datetime.today()

    # Iterate over all tables (assuming each table corresponds to a year)
    for table in soup.select("table"):
        # Optional: detect year from preceding heading
        year_heading = table.find_previous(["h2", "h3", "h4"])
        if year_heading and any(str(y) in year_heading.text for y in range(2025, 2030)):
            year = int([y for y in range(2025, 2030) if str(y) in year_heading.text][0])
        else:
            year = today.year  # fallback

        for row in table.select("tbody tr"):
            tds = row.find_all("td")
            if len(tds) != 2:
                continue
            date_str = tds[0].text.strip()  # e.g., "Thursday 6 February"
            desc_span = tds[1].find("span")
            desc = desc_span.text.strip() if desc_span else ""
            
            # Parse date with detected year
            try:
                date = datetime.strptime(f"{date_str} {year}", "%A %d %B %Y")
            except ValueError:
                print(f"Unrecognized date format: {date_str} with year {year}")
                continue

            if date >= today:
                links = [a['href'] for a in tds[1].find_all("a")]
                meetings.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Description": desc,
                    "Links": links
                })
                print(f"Found MPC meeting: {date.strftime('%Y-%m-%d')} - {desc}")

    df = pd.DataFrame(meetings)
    return pd.Series(df['Date'], name="MPC_Date")

def get_upcoming_boc_dates(pages=3):
    """
    Scrape the Bank of Canada upcoming events pages for interest rate and monetary policy events.
    Returns a DataFrame with Date, Description, and Link.
    """
    base_url = "https://www.bankofcanada.ca/press/upcoming-events/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0.5845.188 Safari/537.36"
    }

    events = []
    today = datetime.today()

    for page in range(1, pages + 1):
        url = base_url if page == 1 else f"{base_url}?mt_page={page}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        for media in soup.select(".media-body"):
            date_span = media.select_one(".media-date")
            h3 = media.select_one("h3.media-heading a")
            if not date_span or not h3:
                continue

            date_str = date_span.text.strip()  # e.g., "October 29, 2025"
            desc = h3.text.strip()
            link = h3['href']

            # Only include monetary policy / interest rate announcements
            if "monetary policy" in desc.lower() or "interest rate" in desc.lower():
                try:
                    date = datetime.strptime(date_str, "%B %d, %Y")
                except ValueError:
                    print(f"Unrecognized date format: {date_str}")
                    continue

                if date >= today:
                    events.append({
                        "Date": date.strftime("%Y-%m-%d")
                    })
                    print(f"Found BOC event: {date.strftime('%Y-%m-%d')} - {desc}")

    df = pd.DataFrame(events)
    return pd.Series(df['Date'], name="BOC_Date")

def get_upcoming_snb_dates(url="https://www.snb.ch/en/services-events/digital-services/event-schedule"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0.5845.188 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    events = []
    today = datetime.today()

    for li in soup.select("li.col-12.link-list-item"):
        a_tag = li.find("a", class_="link-list-item__link")
        if not a_tag:
            continue
        title_tag = li.select_one(".publication-title h3")
        date_tag = li.select_one(".publication-date")
        time_tag = li.select_one(".publication-type")

        if not title_tag or not date_tag:
            continue

        title = title_tag.text.strip()
        # Filter only monetary policy assessments
        if "monetary policy" not in title.lower():
            continue

        date_str = date_tag.text.strip()  # e.g., "25.09.2025"
        time_str = time_tag.text.strip() if time_tag else ""

        try:
            date = datetime.strptime(date_str, "%d.%m.%Y")
        except ValueError:
            print(f"Unrecognized date format: {date_str}")
            continue

        if date >= today:
            events.append({
                "Date": date.strftime("%Y-%m-%d")
            })
            print(f"Found SNB event: {date.strftime('%Y-%m-%d')} - {title}")

    return pd.Series([e['Date'] for e in events], name="SNB_Date")

def get_rba_board_meeting_dates(url="https://www.rba.gov.au/schedules-events/board-meeting-schedules.html"):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    tables = soup.find_all("table", class_="table-linear")
    current_year = datetime.today().year
    next_year = current_year + 1

    meetings = []

    month_map = {m: i for i, m in enumerate([
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"], 1)}

    for table in tables:
        caption = table.find("caption")
        if not caption:
            continue
        year_match = re.search(r"(\d{4})", caption.text)
        if not year_match:
            continue
        year = int(year_match.group(1))
        if year not in [current_year, next_year]:
            continue

        for row in table.tbody.find_all("tr"):
            cols = row.find_all(["th", "td"])
            if len(cols) < 2:
                continue
            month_name = cols[0].text.strip()
            month_num = month_map.get(month_name, None)
            if not month_num:
                continue
            dates_text = cols[1].text.strip()
            if not dates_text:
                continue

            # Remove footnotes
            dates_text = re.sub(r'\s*\(.*?\)', '', dates_text)
            # Handle date ranges
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
                    month_override = day_match.group(2)
                    month_final = month_map.get(month_override, month_num) if month_override else month_num
                    date = datetime.strptime(f"{day} {month_final} {year}", "%d %m %Y")
                    if date >= datetime.today():
                        meetings.append(date)

    df = pd.DataFrame(sorted(meetings), columns=["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return pd.Series(df['Date'], name="RBA_Date")

def get_barchart_ticker_name(interest_rate, tenor, month, year):
    """
    Returns a ticker name for a bar chart based on the interest rate, month, and year.

    Parameters:
    interest_rate str: Name of underlying interest rate (e.g., 'LIBOR', 'SOFR').
    month (int): Month as an integer (1-12).
    year (int): Year as a four-digit integer (e.g., 2023).
    tenor (str): Tenor of the interest rate (e.g., '1M', '3M').
    Returns:
    str: Formatted ticker name (e.g., ZQU25 for Fed Funds, December 2025).
    """
    month_codes = {
        1: 'F',  # January
        2: 'G',  # February
        3: 'H',  # March
        4: 'J',  # April
        5: 'K',  # May
        6: 'M',  # June
        7: 'N',  # July
        8: 'Q',  # August
        9: 'U',  # September
        10: 'V', # October
        11: 'X', # November
        12: 'Z'  # December
    }
    ticker_prefix = { 
        "FED_FUNDS": {
            "1M": "ZQ"
        },
        "ESTR": {
            "1M": "EG"
        }, 
        "TONA": {
            "3M": "T0"
        }, 
        "SONIA": { 
            "1M": "JU", 
            #"3M": "J8",
        },
        "CORRA": { 
            "1M": "RI", 
            "3M": "RG"
        }, 
        "SARON": { 
            "3M": "J2"
        }, 
        "SFE_BA":{ 
            "1M": "IQ",
            #"3M": "IR"
        }
    }
    if interest_rate not in ticker_prefix:
        raise ValueError(f"Unsupported interest rate: {interest_rate}")
    if tenor not in ticker_prefix[interest_rate]:
        raise ValueError(f"Unsupported tenor: {tenor} for interest rate: {interest_rate}")
    if month not in month_codes:
        raise ValueError("Month must be an integer between 1 and 12.")
    if year < 2000 or year > 2099:
        raise ValueError("Year must be a four-digit integer between 2000 and 2099.")
    month_code = month_codes[month]
    year_code = str(year)[-2:]
    return f"{ticker_prefix[interest_rate][tenor]}{month_code}{year_code}"

def get_implied_rate(price):
    return 100 - price

def get_data(ticker, url):
    with requests.Session() as req:
        req.headers.update(DEFAULT_HEADERS)
        r = req.get(url[:25])
        req.headers.update(
            {'X-XSRF-TOKEN': unquote(r.cookies.get_dict()['XSRF-TOKEN'])})
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
        r = req.get(url, params=params).json()
        df= pd.DataFrame(r['data']).iloc[:, :-1]
        return df

def fetch_single(ticker, u, tenor, month, year):
    try:
        print(f"Fetching data for {ticker}")
        df = get_data(ticker, BARCHART_URL)
        df.index = pd.to_datetime(df['tradeTime'])
        df = df[['lastPrice']].copy()
        df['underlying'] = u
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

def convert_column_to_float(df, column_names):
    for col in column_names:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_central_bank_rates():
    url = "https://www.cbrates.com/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    rows = soup.find_all("tr")
    data = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 6:
            rate = cols[1].get_text(strip=True).replace('%','')
            change = cols[2].get_text(strip=True)
            country = cols[4].get_text(strip=True).split("|")[0].strip() if len(cols) > 4 else ""
            change_date = cols[5].get_text(strip=True) if len(cols) > 5 else ""
            data.append([country, rate, change, change_date])

    df = pd.DataFrame(data, columns=["Country", "Rate", "Change", "Date"])

    # Set index to Country
    df.set_index("Country", inplace=True)

    return df

@functools.lru_cache(maxsize=32)
def get_database(details, start_month, year, n_contracts, max_workers=5):
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for u, tenor in details:
            current_year = year  # prevent modifying outer `year`
            for i in range(n_contracts + 1):
                month = start_month + i
                if month > 12:
                    month -= 12
                    current_year += 1
                ticker = get_barchart_ticker_name(u, tenor, month, current_year)
                tasks.append(executor.submit(fetch_single, ticker, u, tenor, month, current_year))   
        res = []
        for future in as_completed(tasks):
            df = future.result()
            res.append(df)

    res = pd.concat(res, axis=0)
    res = convert_column_to_float(res, ['lastPrice'])
    res['implied_rate'] = res['lastPrice'].apply(get_implied_rate)
    return res 

def get_latest_data(db : pd.DataFrame, month: int, year: int):
    return db[(db['month'] == month) & (db['year'] == year)].sort_index().iloc[-1]

def get_matrix(db : pd.DataFrame , underlying, tenor, start_month_and_year : list|tuple, n_months : int, price_or_rate = 'rate'):
    df = db[(db['underlying'] == underlying) & (db['tenor'] == tenor)].copy()
    matrix = pd.DataFrame()
    current_month, current_year = start_month_and_year
    for i in range(n_months):
        month = current_month + i
        year = current_year
        if month > 12:
            month -= 12
            year += 1
        try:
            latest = get_latest_data(df, month, year)
            matrix = pd.concat([matrix, latest.to_frame().T], axis=0)
        except:
            print(f"Error for {underlying} {tenor} {month} {year}")
        
    matrix.index = matrix['contract_name']
    if price_or_rate == 'rate':
        values = matrix['implied_rate']
    else:
        values = matrix['lastPrice']
        
    # Create n×n difference matrix
    diff_matrix = values.values.reshape(-1, 1) - values.values.reshape(1, -1)
    # make it 2dp 
    diff_df = pd.DataFrame(diff_matrix, index=matrix.index, columns=matrix.index)
    # Clean up labels
    diff_df.index.name = None
    diff_df.columns.name = None

    return diff_df


def plot_df(df: pd.DataFrame, title: str = "Time Series",
            ylabel: str = "Value", xlabel: str = "Date"):
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    df = df.reset_index(names=[xlabel])
    
    # Melt to long format for multiple lines
    df_long = df.melt(id_vars=[xlabel], var_name="Contract", value_name=ylabel)
    
    chart = (
        alt.Chart(df_long)
        .mark_line()
        .encode(
            x=alt.X(xlabel, title=xlabel),
            y=alt.Y(ylabel, title=ylabel, scale=alt.Scale(zero=False)),
            color="Contract",
            tooltip=[xlabel, "Contract", ylabel]
        )
        .properties(title=title, width=700, height=400)
    )
    
    st.altair_chart(chart, use_container_width=True)

# Exammple to get_matrix of spread get_matrix(db, 'FED_FUNDS', '1M', (datetime.now().month, datetime.now().year), 3, 'price')
# app.py


st.set_page_config(layout="wide", page_title="STIRT — Interest Rate Dashboard")

st.title("STIRT Dashboard")
underlyings = [
    ('FED_FUNDS','1M'),
    ('ESTR','1M'),
    ('CORRA','1M'),
    ('SFE_BA','1M'),
    ('SONIA','1M'),
    ('TONA','3M'),
]
df_rates = get_central_bank_rates()

underlying_map = {
    "FED_FUNDS": "USA",
    "ESTR": "Eurozone",
    "CORRA": "Canada",
    "SFE_BA": "Australia",
    "SONIA": "United Kingdom", 
    "SARON": "Switzerland",
}

# ---------- Load data ----------
@st.cache_data(ttl=3600*12)  # cache for 6 hours
def load_db_from_barchart(underlyings, start_month, start_year, n_contracts, max_workers):
    # uses your get_database
    #print that fetching data
    return get_database(tuple(underlyings), start_month, start_year, n_contracts, max_workers=max_workers)

# Sidebar controls
with st.sidebar:
    st.header("Data / Controls")
    n_contracts = st.number_input("Contracts per underlying (n_contracts)", min_value=1, max_value=24, value=6)
    start_month = st.number_input("Start month (1-12)", 1, 12, value=datetime.now().month)
    start_year = st.number_input("Start year", 2024, 2099, value=datetime.now().year)
    # Load DB once, store in session state
    if "db" not in st.session_state:
        st.session_state.db = load_db_from_barchart(underlyings, start_month, start_year, n_contracts, MAX_WORKERS)
    db = st.session_state.db
    underlying_rate = st.selectbox("Choose underlying", db['underlying'].unique()) 

# ---------- Dashboard Main ----------
# Display current policy rate for selected underlying
country = underlying_map.get(underlying_rate)
if country and country in df_rates.index:
    policy_info = df_rates.loc[country]
    
    st.subheader(f"Current Policy Rate: {underlying_rate}")
    
    # Create three columns for Rate, Last Change, Last Change Date
    col1, col2, col3 = st.columns(3)
    
    col1.metric(label="Country", value=country)
    col2.metric(label="Rate (%)", value=policy_info['Rate'])
    col3.metric(label="Last Change", value=policy_info['Change'], delta=None)
    
    # Optionally show the date below as a caption
    st.caption(f"Last Change Date: {policy_info['Date']}")
    
    # Optional: colored highlight based on rate direction
    rate_change = policy_info['Change'].replace('+','').replace('-','')
    try:
        rate_change_val = float(rate_change)
        if '+' in policy_info['Change']:
            st.success(f"Rate increased by {policy_info['Change']}")
        elif '-' in policy_info['Change']:
            st.error(f"Rate decreased by {policy_info['Change']}")
        else:
            st.info("No change in rate")
    except:
        st.info("No change in rate")

else:
    st.warning("Policy rate data not available for this underlying.")# choose contract to plot
price_or_rate = st.selectbox("Plot Price or Yield", ["Price", "Yield"], index=1) 
series_choice = "implied_rate" if price_or_rate == "Yield" else "lastPrice"
to_plot = []
contract_df = db[db['underlying'] == underlying_rate]
for i in contract_df['contract_name'].unique():
    series = contract_df[contract_df['contract_name'] == i][series_choice]
    series.name = i
    to_plot.append(series)
to_plot_df = pd.concat(to_plot, axis=1)
plot_df(to_plot_df)

st.subheader("Difference matrix")
mat_tenor = st.selectbox("Matrix tenor", db[db['underlying'] == underlying_rate]['tenor'].unique())
diff_df = get_matrix(db, underlying_rate, mat_tenor, (start_month, start_year), n_contracts, price_or_rate=price_or_rate)
# Function to color values
def color_pos_neg(val):
    if val == None:
        return ''
    color = ''
    if val > 0: 
        color = 'green'
    elif val < 0: 
        color = 'red'
    return f'color: {color}'

styled_df = diff_df.replace(0, np.nan).style.applymap(color_pos_neg).format("{:.2f}")
st.dataframe(styled_df, use_container_width=True)


st.markdown("---")
st.subheader("Central bank meeting calendar & 25bp-cut probabilities")
st.write("Meeting dates shown for mapped central banks. Probabilities are a simple linear mapping from 'futures implied rate change' to a [0,1] probability for a 25bp move (see docs).")




cb_map = {
    "FED_FUNDS": "FED",
    "ESTR": "ECB",
    "SONIA": "BOE",
    "CORRA": "BOC",
    "SARON": "SNB",
    "SFE_BA": "RBA"
}


@st.cache_data(ttl=3600*12)  # cache for 12 hours
def get_all_meeting_dates():
    return {
        "FED": get_upcoming_fomc_dates().tolist(),
        "ECB": get_upcoming_ecb_monetary_dates().tolist(),
        "BOE": get_upcoming_mpc_dates().tolist(),
        "BOC": get_upcoming_boc_dates().tolist(),
        "SNB": get_upcoming_snb_dates().tolist(),
        "RBA": get_rba_board_meeting_dates().tolist()
    }
if "meeting_dates" not in st.session_state:
    st.session_state.meeting_dates = get_all_meeting_dates()

meeting_dates = st.session_state.meeting_dates
spot_df = pd.DataFrame({
    "underlying": ["FED_FUNDS", "ESTR", "SONIA", "CORRA", "SARON", "SFE_BA"],
    "spot_rate": [float(df_rates.loc[underlying_map[u], 'Rate'].split("-")[-1]) if underlying_map[u] in df_rates.index else np.nan for u in ["FED_FUNDS", "ESTR", "SONIA", "CORRA", "SARON", "SFE_BA"]]
})

def implied_cut_probability_refined(current_rate, futures_rate, cut_size=0.25, days_after_meeting=10, total_days=30):
    """
    Refined implied probability of a rate cut using Fed Funds futures.
    """
    if days_after_meeting <= 0:
        # No post-meeting days in this contract, no information about cut probability
        return 0.0
    
    weight = days_after_meeting / total_days
    prob = (current_rate - futures_rate) / (cut_size * weight)
    return max(0, min(1, prob))

def get_most_recent_data(db, underlying): 
    db = db.copy()
    db = db.sort_index()
    return db[db['underlying'] == underlying].groupby('contract_name').last()

def implied_cut_probability(earlier_price, later_price, step_bp=0.25): 
    prob = max(0, min(1, (earlier_price - later_price) / step_bp)) 
    raw = max(0, (earlier_price - later_price) / step_bp)
    return prob, raw

prob_records = []
for underlying, cb in cb_map.items():
    # Spot rate from separate DataFrame
    spot = spot_df.loc[spot_df['underlying']==underlying, 'spot_rate'].values
    if len(spot) == 0:
        continue
    spot = spot[0]
    # Current month futures rate
    current_month_contract = db[(db['underlying']==underlying) & (db['month']==start_month)]
    if current_month_contract.empty:
        continue
    current_rate = current_month_contract[series_choice].iloc[-1] 
    # Iterate over upcoming meetings
    latest = get_most_recent_data(db, underlying)
    print(f"Processing {underlying} for {cb} with spot {spot} and current rate {current_rate}")
    for meeting in meeting_dates.get(cb, []):
        print(f"Processing meeting {meeting} for {underlying}")
        meeting_date = pd.to_datetime(meeting, format="%Y-%m-%d")
        # Find the nearest contract before meeting
        try:
            rate_at_month_of_meeting = latest[(latest.month == meeting_date.month) & (latest.year == meeting_date.year)]['implied_rate'].values[0] 
            if meeting_date.month!= start_month or meeting_date.year!= start_year: 
                if meeting_date.month == 1: 
                    rate_before_month_of_meeting = latest[(latest.month==12) & (latest.year == meeting_date.year)]['implied_rate'].values[0] 
                else: 
                    rate_before_month_of_meeting = latest[(latest.month==meeting_date.month-1) & (latest.year == meeting_date.year)]['implied_rate'].values[0] 
                prob_25, raw_25 = implied_cut_probability(rate_before_month_of_meeting, rate_at_month_of_meeting, 0.25)
                prob_50 , raw_50  = implied_cut_probability(rate_before_month_of_meeting, rate_at_month_of_meeting, 0.5)
                prob_25 = raw_25 / (raw_25 + raw_50)
                prob_50 = raw_50 / (raw_25 + raw_50)
                cut_or_hike = "cut" if rate_at_month_of_meeting < spot else "hike"
            else:
                d = meeting_date.day
                D = monthrange(meeting_date.year, meeting_date.month)[1]
                print(f"SPOT: {spot}, RATE: {rate_at_month_of_meeting}")
                prob_25 = implied_cut_probability_refined(spot, rate_at_month_of_meeting, 0.25, D-d, D)
                prob_50 = implied_cut_probability_refined(spot, rate_at_month_of_meeting, 0.50, D-d, D)
                sum_ = (prob_25 + prob_50)
                prob_25 = prob_25 / sum_
                prob_50  = prob_50 / sum_
                cut_or_hike = "cut" if rate_at_month_of_meeting < spot else "hike"

            prob_records.append({
                "Central Bank": cb,
                "Underlying": underlying,
                "Meeting": meeting_date,
                "Implied Probability": prob_25, 
                "Change in rates": f"25bps {cut_or_hike}"
            })
            prob_records.append({
                "Central Bank": cb,
                "Underlying": underlying,
                "Meeting": meeting_date,
                "Implied Probability": prob_50, 
                "Change in rates": f"50bps {cut_or_hike}"
            })
        except Exception as e:
            import traceback
            #traceback.print_exc()
            #print(f"Error processing meeting {meeting} for {underlying}: {e}")
            continue

prob_df = pd.DataFrame(prob_records)
prob_df['MeetingStr'] = prob_df['Meeting'].dt.strftime('%Y-%m-%d')
meeting  = st.selectbox(r"Select Meeting Date", meeting_dates.get(cb_map.get(underlying_rate)))
filtered_df = prob_df[(prob_df['MeetingStr'] == meeting) & (prob_df['Underlying'] == underlying_rate)]
# Create bar chart: one bar for 25bps cut, one for 50bps cut
chart = (
    alt.Chart(filtered_df)
    .mark_bar()
    .encode(
        x=alt.X("Change in rates:N", title="Rate Change"),
        y=alt.Y("Implied Probability:Q", title="Implied Probability", axis=alt.Axis(format='%')),
        color="Change in rates:N",
        tooltip=["Central Bank", "Underlying", "MeetingStr", alt.Tooltip("Implied Probability", format=".0%")]
    )
    .properties(width=300, height=400)
)

st.altair_chart(chart, use_container_width=True)