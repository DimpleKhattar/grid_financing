"""
Author: Dimple
Date created: 16 May 2026

Purpose; Master Data Downloader —
  1. ERCOT LMP  — HB_HOUSTON + HB_WEST hourly settlement-point prices (Feb–today 2026)
  2. Henry Hub  — Daily natural gas spot price from FRED/EIA (Jan 2025–today)
  3. ERCOT Wind — Hourly wind generation by ERCOT region (Feb–today 2026)
  4. ERCOT Solar— Hourly solar generation by ERCOT region (Feb–today 2026)


ASSUMPTIONS & MARKET STANDARDS
-------------------------------
1. ERCOT CDR Report IDs (verified from ERCOT EMIL public directory):
     RTM_SPP_RTID = 13061   NP6-785-ER  Historical Real-Time Market Settlement
                             Point Prices — annual xlsx, all hubs and load zones
     WIND_RTID    = 13028   NP4-742-CD  Wind Production Hourly by Weather Zone
     SOLAR_RTID   = 21809   NP4-745-CD  Solar Production Hourly by Weather Zone
   These IDs are subject to change if ERCOT restructures its CDR; verify at
   https://www.ercot.com/misapp/GetReports.do if downloads fail.

2. LMP Hub Selection: HB_HOUSTON and HB_WEST
   HB_HOUSTON = Houston Ship Channel hub, primary for Gulf Coast gas plants.
   HB_WEST    = West Texas hub (Waha area), primary for Permian Basin resources.
   Both are ERCOT "Hub" settlement points, not load-zone prices.

3. Hourly Aggregation for LMP
   Raw ERCOT RTM SPP data is published at 15-minute intervals (4 per hour).
   This script averages the 15-minute prices to produce hourly means,
   consistent with how gas-plant tolling agreements typically settle (hourly).

4. ERCOT Time Convention: HOUR_ENDING
   ERCOT labels hours 1–24 (hour ending), so "Hour 1" is 00:00–01:00 CST.
   We subtract 1 before constructing datetime to get the start-of-hour timestamp,
   which is the convention used by pandas and the rest of this project.

5. Tolling Cost Formula (Henry Hub file)
   tolling_cost_per_mwh = (henry_hub_price_mmbtu + gas_adder) × heat_rate
   where:
     gas_adder = $3.00/MMBtu  (Houston Ship Channel premium over Henry Hub;
                                includes transport, fuel, and variable O&M)
     heat_rate = 9.5 MMBtu/MWh (simple-cycle gas turbine, ERCOT market standard
                                  for peaker plants; combined-cycle ≈ 7.0)
   This matches the GBM Monte Carlo spark spread calculation in
   gbm_calibration_monte_carlo.py.

6. Henry Hub Data Source: FRED Series DHHNGSP
   Published by the U.S. Energy Information Administration (EIA).
   Filtered to Jan 2025 onward to capture the current price regime for GBM
   volatility calibration. Using earlier data would mix pre/post-2022 regimes
   with structurally different volatility.

7. Wind/Solar File Limit: 100 most-recent documents
   ERCOT publishes one CDR file per day (or sub-day interval) for wind/solar.
   Downloading 100 files covers approximately 3 months of history.
   Rate-limited to 0.15s between requests to avoid hitting ERCOT CDR throttle.

8. All timestamps stored in Central Standard Time (CST), not adjusted for DST.
   This is the ERCOT convention for market settlement.

SETUP:
    pip install pandas requests openpyxl


"""

import os, time, zipfile, requests, warnings
import pandas as pd
from io import StringIO, BytesIO
from datetime import date, timedelta

warnings.filterwarnings("ignore")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
START_DATE  = "2026-02-01"
END_DATE    = date.today().isoformat()

os.makedirs(DATA_FOLDER, exist_ok=True)

# ERCOT CDR report type IDs (verified from ERCOT EMIL directory)
RTM_SPP_RTID   = 13061   # NP6-785-ER  Historical RTM Load Zone & Hub Prices (annual xlsx)
WIND_RTID      = 13028   # NP4-742-CD  Wind Production Hourly by Region
SOLAR_RTID     = 21809   # NP4-745-CD  Solar Production Hourly by Region

CDR_LIST_URL  = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={rtid}"
CDR_DL_URL    = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={did}"

download_log = []

def log(status, name, path, rows=None, note=""):
    download_log.append(dict(
        status=status, dataset=name,
        file=os.path.basename(path) if path else "—",
        rows=f"{rows:,}" if rows else "—",
        note=note,
    ))

# Helpers 

def get_cdr_docs(rtid):
    r = requests.get(CDR_LIST_URL.format(rtid=rtid), timeout=30)
    r.raise_for_status()
    result = r.json().get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
    if not result:
        return []
    return result if isinstance(result, list) else [result]

def dl_bytes(doc_id):
    r = requests.get(CDR_DL_URL.format(did=doc_id), timeout=120)
    r.raise_for_status()
    return r.content

def parse_zip_or_csv(content, doc_name=""):
    """Return a DataFrame"""
    try:
        with zipfile.ZipFile(BytesIO(content)) as z:
            frames = []
            for n in z.namelist():
                nl = n.lower()
                if nl.endswith(".csv"):
                    with z.open(n) as f:
                        frames.append(pd.read_csv(f, low_memory=False))
                elif nl.endswith(".xlsx") or nl.endswith(".xls"):
                    with z.open(n) as f:
                        xl = pd.ExcelFile(BytesIO(f.read()))
                        for sh in xl.sheet_names:
                            try:
                                frames.append(xl.parse(sh))
                            except Exception:
                                pass
                elif nl.endswith(".xml"):
                    with z.open(n) as f:
                        try:
                            frames.append(pd.read_xml(f))
                        except Exception:
                            pass
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except zipfile.BadZipFile:
        pass
    try:
        return pd.read_xml(BytesIO(content))
    except Exception:
        pass
    try:
        return pd.read_csv(BytesIO(content), low_memory=False)
    except Exception:
        return pd.DataFrame()

def add_datetime(df):
    """Build datetime_cst from DeliveryDate+DeliveryHour """
    df.columns = [c.strip() for c in df.columns]
    dc = next((c for c in df.columns if c.lower().replace("_","") == "deliverydate"), None)
    hc = next((c for c in df.columns if c.lower().replace("_","") in ("deliveryhour", "hourending")), None)
    if dc and hc:
        df["datetime_cst"] = pd.to_datetime(
            df[dc].astype(str) + " " +
            (pd.to_numeric(df[hc], errors="coerce").fillna(1).astype(int) - 1).astype(str) + ":00"
        )
        return df
    for c in df.columns:
        if any(k in c.lower() for k in ("time","timestamp","datetime","interval")):
            try:
                df["datetime_cst"] = pd.to_datetime(df[c]).dt.tz_localize(None)
                return df
            except Exception:
                continue
    raise ValueError(f"No datetime col found. Cols: {list(df.columns)[:8]}")

def resample_hourly(df, val_col):
    """Resample a long-format df to hourly mean using integer groupby (no freq string)."""
    df = df.copy()
    df["datetime_cst"] = pd.to_datetime(df["datetime_cst"])
    # Truncate to hour manually — avoids all pandas freq resolution issues
    df["hour_ts"] = df["datetime_cst"].dt.floor("s").apply(
        lambda x: x.replace(minute=0, second=0, microsecond=0)
    )
    return df.groupby("hour_ts")[val_col].mean().reset_index().rename(
        columns={"hour_ts": "datetime_cst"}
    )



#ERCOT LMP  

def download_ercot_lmp():
    name    = "ERCOT LMP (HB_HOUSTON, HB_WEST)"
    outfile = os.path.join(DATA_FOLDER, "ercot_lmp_hb_houston_hb_west.csv")
    locs    = ["HB_HOUSTON", "HB_WEST"]

 

    try:
        print(f"  Fetching document list (reportTypeId={RTM_SPP_RTID})...")
        docs = get_cdr_docs(RTM_SPP_RTID)
        print(f"  Found {len(docs)} document(s)")

        # Find 2026 annual file — it's an xlsx or zip named like *2026*
        target_doc = None
        for dw in docs:
            d    = dw.get("Document", dw)
            name_ = (d.get("FriendlyName","") + d.get("ConstructedName","")).lower()
            ext  = d.get("Extension","").lower()
            if "2026" in name_ and ext in ("xlsx","zip","xls"):
                target_doc = d
                break

        # If not found by name, just take the most recent file
        if not target_doc and docs:
            target_doc = docs[0].get("Document", docs[0])

        if not target_doc:
            raise ValueError("No suitable annual file found in CDR listing")

        doc_id   = target_doc.get("DocID","")
        doc_name = target_doc.get("FriendlyName", target_doc.get("ConstructedName",""))
        ext      = target_doc.get("Extension","").lower()
        print(f"  Downloading: {doc_name} (id={doc_id}, ext={ext})")

        content = dl_bytes(doc_id)
        print(f"  Downloaded {len(content)/1024/1024:.1f} MB")

        # Parse depending on file type
        if ext == "xlsx" or ext == "xls":
            print("  Parsing Excel file (this may take 30–60s for a full year)...")
            xl = pd.ExcelFile(BytesIO(content))
            print(f"  Sheets: {xl.sheet_names}")
            frames = []
            for sh in xl.sheet_names:
                try:
                    df = xl.parse(sh)
                    frames.append(df)
                except Exception:
                    pass
            raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            raw = parse_zip_or_csv(content, doc_name)

        if raw.empty:
            raise ValueError("Parsed file is empty")

        raw.columns = [c.strip() for c in raw.columns]
        print(f"  Raw shape: {raw.shape} | Columns: {list(raw.columns)[:8]}")

        # Find settlement point and price columns (normalize spaces + underscores)
        def norm(s): return s.lower().replace("_","").replace(" ","")
        sp_col = next((c for c in raw.columns
                       if norm(c) in ("settlementpoint", "settlementpointname")), None)
        pr_col = next((c for c in raw.columns
                       if "settlementpointprice" in norm(c)), None)
        dc_col = next((c for c in raw.columns
                       if norm(c) == "deliverydate"), None)
        hr_col = next((c for c in raw.columns
                       if norm(c) in ("deliveryhour", "hourending")), None)

        print(f"  Key columns: sp={sp_col}, price={pr_col}, date={dc_col}, hour={hr_col}")

        if not all([sp_col, pr_col, dc_col, hr_col]):
            raise ValueError(f"Missing required columns. Available: {list(raw.columns)}")

        # Filter to our two hubs
        raw = raw[raw[sp_col].isin(locs)].copy()
        print(f"  After hub filter: {len(raw):,} rows")

        if raw.empty:
            raise ValueError(f"No rows found for {locs}. Available locations: {raw[sp_col].unique()[:10].tolist()}")

        # Build datetime
        raw["datetime_cst"] = pd.to_datetime(
            raw[dc_col].astype(str) + " " +
            (pd.to_numeric(raw[hr_col], errors="coerce").fillna(1).astype(int) - 1).astype(str) + ":00"
        )
        raw["lmp"] = pd.to_numeric(raw[pr_col], errors="coerce")
        raw = raw.rename(columns={sp_col: "settlement_point"})
        raw = raw[["datetime_cst","settlement_point","lmp"]].dropna()

        # Filter to project window
        raw = raw[
            (raw["datetime_cst"] >= pd.Timestamp(START_DATE)) &
            (raw["datetime_cst"] <= pd.Timestamp(END_DATE) + timedelta(days=1))
        ]
        print(f"  After date filter: {len(raw):,} rows")

        # Hourly mean per hub — using floor instead of Grouper to avoid freq issues
        raw["hour_ts"] = raw["datetime_cst"].apply(
            lambda x: x.replace(minute=0, second=0, microsecond=0)
        )
        hourly = (
            raw.groupby(["settlement_point","hour_ts"])["lmp"]
            .mean().reset_index()
            .rename(columns={"hour_ts":"datetime_cst"})
        )

        # Pivot wide
        wide = hourly.pivot_table(
            index="datetime_cst", columns="settlement_point",
            values="lmp", aggfunc="mean"
        ).reset_index()
        wide.columns.name = None

        wide["hour_of_day"] = wide["datetime_cst"].dt.hour
        wide["day_of_week"] = wide["datetime_cst"].dt.day_name()
        wide["month"]       = wide["datetime_cst"].dt.to_period("M").astype(str)
        wide["is_weekend"]  = wide["datetime_cst"].dt.dayofweek >= 5
        if "HB_HOUSTON" in wide.columns and "HB_WEST" in wide.columns:
            wide["spread_houston_minus_west"] = wide["HB_HOUSTON"] - wide["HB_WEST"]

        wide = wide.sort_values("datetime_cst").reset_index(drop=True)
        wide.to_csv(outfile, index=False)

        note = f"{wide['datetime_cst'].min().date()} → {wide['datetime_cst'].max().date()}"
        log("OK", name, outfile, len(wide), note)
        print(f"Rows: {len(wide):,} | {note}")
        if "HB_HOUSTON" in wide.columns:
            print(f"     HB_HOUSTON: mean=${wide['HB_HOUSTON'].mean():.2f} | min=${wide['HB_HOUSTON'].min():.2f} | max=${wide['HB_HOUSTON'].max():.2f}")
        if "HB_WEST" in wide.columns:
            print(f"     HB_WEST:    mean=${wide['HB_WEST'].mean():.2f} | min=${wide['HB_WEST'].min():.2f} | max=${wide['HB_WEST'].max():.2f}")
        

    except Exception as e:
        log("FAIL", name, None, note=str(e)[:150])
        print(f" ERROR: {e}")
        import traceback; traceback.print_exc()



#  Henry Hub 

def download_henry_hub():
    name    = "Henry Hub Natural Gas Daily Spot Price"
    outfile = os.path.join(DATA_FOLDER, "henry_hub_daily.csv")

   
    try:
        print("  Downloading DHHNGSP from FRED...")
        r = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP",
            timeout=30
        )
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date","henry_hub_price_mmbtu"]
        df["date"] = pd.to_datetime(df["date"])
        df["henry_hub_price_mmbtu"] = pd.to_numeric(df["henry_hub_price_mmbtu"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)

        # Keep from Jan 2025 for GBM volatility calibration
        df = df[df["date"] >= pd.Timestamp("2025-01-01")]

        # Tolling cost = (HH + $3 premium) × 9,500 BTU/kWh ÷ 1,000
        df["tolling_cost_per_mwh"] = (df["henry_hub_price_mmbtu"] + 3) * 9.5

        df.to_csv(outfile, index=False)

        in_win = df[df["date"] >= pd.Timestamp(START_DATE)]
        latest = df["henry_hub_price_mmbtu"].iloc[-1]
        note   = f"{in_win['date'].min().date()} → {in_win['date'].max().date()} | latest=${latest:.2f}/MMBtu"
        log("OK", name, outfile, len(in_win), note)
        print(f" {len(in_win):,} rows in project window | {note}")
        print(f"     Latest tolling cost: ${df['tolling_cost_per_mwh'].iloc[-1]:.2f}/MWh")
        

    except Exception as e:
        log("FAIL", name, None, note=str(e)[:120])
        print(f" ERROR: {e}")



# ERCOT Wind / Solar —

def download_ercot_generation(label, rtid, outfile_name, num):
    name    = f"ERCOT {label} Generation (Hourly by Region)"
    outfile = os.path.join(DATA_FOLDER, outfile_name)

    try:
        print(f"  Fetching document list (reportTypeId={rtid})...")
        docs = get_cdr_docs(rtid)
        print(f"  Found {len(docs)} documents — downloading up to 100 (≈3 months)")

        if not docs:
            raise ValueError("No documents returned from CDR")

        chunks  = []
        target  = docs[:100]
        success = 0
        failed  = 0

        for i, dw in enumerate(target):
            d        = dw.get("Document", dw)
            doc_id   = d.get("DocID","")
            doc_name = d.get("FriendlyName","") or d.get("ConstructedName","")
            if not doc_id:
                continue

            print(f"\r  [{i+1:>3}/{len(target)}] {doc_name[:52]:<52} ok={success} fail={failed}",
                  end="", flush=True)
            try:
                content = dl_bytes(doc_id)
                df = parse_zip_or_csv(content, doc_name)
                if not df.empty:
                    chunks.append(df)
                    success += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            time.sleep(0.15)

        print(f"\n  Downloaded {success} files, {failed} failed")

        if not chunks:
            raise ValueError("All files failed to parse")

        raw = pd.concat(chunks, ignore_index=True)
        raw = add_datetime(raw)
        raw = raw[raw["datetime_cst"] >= pd.Timestamp(START_DATE)]
        raw = raw.sort_values("datetime_cst").reset_index(drop=True)

        # Drop exact duplicate rows
        raw = raw.drop_duplicates().reset_index(drop=True)

        raw.to_csv(outfile, index=False)
        date_range = f"{raw['datetime_cst'].min().date()} → {raw['datetime_cst'].max().date()}"
        log("OK", name, outfile, len(raw), date_range)
        data_cols = [c for c in raw.columns if c != "datetime_cst"]
        print(f"Rows: {len(raw):,} | {date_range}")
        print(f"     Data columns: {data_cols[:8]}")
        

    except Exception as e:
        log("FAIL", name, None, note=str(e)[:120])
        print(f"CDR failed: {e}")

        # gridstatus fallback
        print("  Trying gridstatus fallback...")
        try:
            import gridstatus
            ercot   = gridstatus.Ercot()
            method  = ercot.get_wind_forecast if "Wind" in label else ercot.get_solar_forecast
            chunks  = []
            current = pd.Timestamp(START_DATE)
            end_ts  = pd.Timestamp(END_DATE)

            while current <= end_ts:
                wk = min(current + timedelta(days=6), end_ts)
                print(f"\r  Fetching {current.date()} → {wk.date()}...", end="", flush=True)
                try:
                    df = method(date=current.strftime("%Y-%m-%d"),
                                end=wk.strftime("%Y-%m-%d"), verbose=False)
                    chunks.append(df)
                except Exception:
                    pass
                current = wk + timedelta(days=1)
                time.sleep(0.3)

            print()
            if chunks:
                raw = pd.concat(chunks, ignore_index=True)
                raw.to_csv(outfile, index=False)
                log("OK (gridstatus)", name, outfile, len(raw))
                print(f" Saved via gridstatus fallback: {outfile}")
            else:
                log("FAIL", name, None, note="Both CDR and gridstatus returned no data")
                print(" No data from gridstatus either")

        except Exception as e2:
            log("FAIL", name, None, note=f"Both methods failed: {str(e2)[:80]}")
            print(f" Gridstatus fallback failed: {e2}")



# Summary

def print_summary():
    
    
    print(f"Folder:  {DATA_FOLDER}")
    print(f"Date range: {START_DATE} → {END_DATE}")
    print()
    print(f"{'Status':<14}  {'Dataset':<40}  {'Rows':<10}  Note")
    
    for e in download_log:
        print(f"  {e['status']:<14}  {e['dataset']:<40}  {e['rows']:<10}  {e['note']}")

    ok = sum(1 for e in download_log if "OK" in e["status"])
    print()
    print(f"Result: {ok}/{len(download_log)} datasets succeeded")
    print()
    print("Files saved:")
    if os.path.exists(DATA_FOLDER):
        for f in sorted(os.listdir(DATA_FOLDER)):
            if f.endswith(".csv"):
                kb = os.path.getsize(os.path.join(DATA_FOLDER, f)) / 1024
                print(f"{f:<52}  {kb:>8.0f} KB")
    print("="*70)


# Main 

if __name__ == "__main__":

    download_ercot_lmp()
    download_henry_hub()
    download_ercot_generation("Wind",  WIND_RTID,  "ercot_wind_hourly.csv",  3)
    download_ercot_generation("Solar", SOLAR_RTID, "ercot_solar_hourly.csv", 4)

    print_summary()