import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta

# PE holidays (fixed + variable) — comprehensive list
# We'll use a function-based approach for known fixed holidays
# and add manually curated variable holidays (Semana Santa, etc.)

def get_pe_holidays(year):
    holidays = set()
    # Fixed holidays
    holidays.add(date(year, 1, 1))   # Año Nuevo
    holidays.add(date(year, 5, 1))   # Día del Trabajo
    holidays.add(date(year, 6, 29))  # San Pedro y San Pablo
    holidays.add(date(year, 7, 23))  # Día de la FAP
    holidays.add(date(year, 7, 28))  # Fiestas Patrias
    holidays.add(date(year, 7, 29))  # Fiestas Patrias
    holidays.add(date(year, 8, 6))   # Batalla de Junín
    holidays.add(date(year, 8, 30))  # Santa Rosa de Lima
    holidays.add(date(year, 10, 8))  # Combate de Angamos
    holidays.add(date(year, 11, 1))  # Todos los Santos
    holidays.add(date(year, 12, 8))  # Inmaculada Concepción
    holidays.add(date(year, 12, 9))  # Batalla de Ayacucho
    holidays.add(date(year, 12, 25)) # Navidad
    # Easter-based (Jueves Santo, Viernes Santo)
    easter = compute_easter(year)
    holidays.add(easter - timedelta(days=3))  # Jueves Santo
    holidays.add(easter - timedelta(days=2))  # Viernes Santo
    return holidays

def compute_easter(year):
    # Anonymous Gregorian algorithm
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

def get_us_holidays(year):
    holidays = set()
    # Fixed
    holidays.add(date(year, 1, 1))   # New Year
    holidays.add(date(year, 6, 19))  # Juneteenth (from 2021, but include anyway)
    holidays.add(date(year, 7, 4))   # Independence Day
    holidays.add(date(year, 11, 11)) # Veterans Day
    holidays.add(date(year, 12, 25)) # Christmas
    # MLK Day: 3rd Monday of January
    holidays.add(nth_weekday(year, 1, 0, 3))
    # Presidents Day: 3rd Monday of February
    holidays.add(nth_weekday(year, 2, 0, 3))
    # Memorial Day: last Monday of May
    holidays.add(last_weekday(year, 5, 0))
    # Labor Day: 1st Monday of September
    holidays.add(nth_weekday(year, 9, 0, 1))
    # Columbus Day: 2nd Monday of October
    holidays.add(nth_weekday(year, 10, 0, 2))
    # Thanksgiving: 4th Thursday of November
    holidays.add(nth_weekday(year, 11, 3, 4))
    return holidays

def nth_weekday(year, month, weekday, n):
    # weekday: 0=Monday, 3=Thursday
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))

def last_weekday(year, month, weekday):
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)

# Build combined holiday set for all years in data
def build_holiday_set(start_year, end_year):
    holidays = set()
    for y in range(start_year, end_year + 2):  # +2 for safety with year-end rolls
        holidays.update(get_pe_holidays(y))
        holidays.update(get_us_holidays(y))
    return holidays

def is_business_day(d, holidays):
    return d.weekday() < 5 and d not in holidays

def add_business_days(d, n, holidays):
    current = d
    added = 0
    while added < n:
        current += timedelta(days=1)
        if is_business_day(current, holidays):
            added += 1
    return current

def modified_following(d, holidays):
    # If d is a business day, return d
    if is_business_day(d, holidays):
        return d
    # Roll forward
    rolled = d
    while not is_business_day(rolled, holidays):
        rolled += timedelta(days=1)
    # If rolled into next month, roll backward from original instead
    if rolled.month != d.month:
        rolled = d
        while not is_business_day(rolled, holidays):
            rolled -= timedelta(days=1)
    return rolled

def tenor_to_relativedelta(cod_plazo):
    cod_plazo = cod_plazo.strip().upper()
    if cod_plazo.endswith('M'):
        months = int(cod_plazo[:-1])
        return relativedelta(months=months)
    elif cod_plazo.endswith('Y'):
        years = int(cod_plazo[:-1])
        return relativedelta(years=years)
    else:
        raise ValueError(f'Unknown tenor: {cod_plazo}')

def compute_value_date(trade_date, cod_plazo, holidays):
    # Spot value date = T+2 (as specified)
    spot_date = add_business_days(trade_date, 2, holidays)
    # NDF value date = spot + tenor, then modified following
    tenor_delta = tenor_to_relativedelta(cod_plazo)
    raw_maturity = spot_date + tenor_delta
    value_date = modified_following(raw_maturity, holidays)
    return spot_date, value_date

# Read data
df = pd.read_excel('/mnt/user-data/uploads/pca_ndf.xlsx')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df["Fecha"].min()} to {df["Fecha"].max()}')
print(f'Tenors: {df["CodPlazo"].unique()}')
print(f'\nFirst 10 rows:')
print(df.head(10))

# Build holiday set
min_year = pd.to_datetime(df['Fecha']).dt.year.min()
max_year = pd.to_datetime(df['Fecha']).dt.year.max()
holidays = build_holiday_set(min_year, max_year)

# Compute value dates and implied PEN rate
df['Fecha'] = pd.to_datetime(df['Fecha'])
spot_dates = []
value_dates = []
actual_days = []
pen_implied = []

for _, row in df.iterrows():
    trade_date = row['Fecha'].date()
    cod_plazo = row['CodPlazo']
    mid_pips = row['Mid']
    spot = row['pen_fixing']
    sofr_pct = row['SOFR']

    # Value date computation
    spot_dt, val_dt = compute_value_date(trade_date, cod_plazo, holidays)
    T = (val_dt - spot_dt).days  # ACT/360 day count

    spot_dates.append(spot_dt)
    value_dates.append(val_dt)
    actual_days.append(T)

    # Forward rate
    fwd_points = mid_pips / 10000.0
    F = spot + fwd_points

    # USD rate as decimal
    r_usd = sofr_pct / 100.0

    # Implied PEN simple rate first
    r_pen_simple = ((F / spot) * (1 + r_usd * T / 360.0) - 1) * 360.0 / T
    # Convert to annual effective rate (TEA): (1 + r_simple * T/360)^(360/T) - 1
    r_pen_tea = (1 + r_pen_simple * T / 360.0) ** (360.0 / T) - 1
    pen_implied.append(r_pen_tea * 100.0)  # store as percentage

df['spot_date'] = spot_dates
df['value_date'] = value_dates
df['actual_days'] = actual_days
df['pen_implied_calc'] = pen_implied

# Compare with existing pen_implied column if it exists
if 'pen_implied' in df.columns:
    # Clean the existing column (might have % signs)
    existing = df['pen_implied'].astype(str).str.replace('%', '').astype(float)
    df['diff_bps'] = (df['pen_implied_calc'] - existing) * 100  # difference in bps
    print(f'\nDifference vs existing pen_implied (bps):')
    print(df['diff_bps'].describe())

print(f'\nSample output:')
print(df[['Fecha', 'CodPlazo', 'actual_days', 'Mid', 'pen_fixing', 'SOFR', 'pen_implied_calc']].head(20))

# Save output
df.to_excel('/home/claude/pca_ndf_output.xlsx', index=False)
print(f'\nSaved to /home/claude/pca_ndf_output.xlsx')
