import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine, text
import tempfile
import os
from datetime import datetime, timedelta

app = FastAPI()

# === ALLOW CORS UNTUK IONIC ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DATABASE CONNECTION STRING GLOBAL ===
DB_URL = "postgresql://postgres.uonphwbbgemsvqzrdcwp:Ika_15050107@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"
engine = create_engine(
    DB_URL,
    pool_size=40,       # kurang dari 48
    max_overflow=8,     # boleh tambah sementara bila perlu
    pool_timeout=30     # tunggu 30s kalau semua busy
)

# === FUNGSI PEMBERSIH HEADER ===
def clean_header(col):
    col = str(col)
    col = re.sub(r"\s+", "", col)
    col = re.sub(r"&|/|\(|\)|\\n|RM", "", col, flags=re.IGNORECASE)
    return col

# === FUNGSI TARIKH 6AM-TO-6AM ===
def parse_date_range(start_date, end_date, filter_6am=True):
    if start_date and not end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") + (timedelta(hours=6) if filter_6am else timedelta())
        end_dt = start_dt + timedelta(days=1)
    elif start_date and end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") + (timedelta(hours=6) if filter_6am else timedelta())
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + (timedelta(hours=6) if filter_6am else timedelta(days=1))
    else:
        now = datetime.now()
        start_dt = (now - timedelta(days=2)).replace(hour=6 if filter_6am else 0, minute=0, second=0)
        end_dt = now.replace(hour=6 if filter_6am else 23, minute=0 if filter_6am else 59, second=0 if filter_6am else 59)
    return start_dt, end_dt

# === /upload ENDPOINT ===
@app.post("/upload")
async def upload_csv(files: list[UploadFile] = File(...), preview: bool = Query(False)):
    """
    preview=True → hanya tapis & simpan CSV
    preview=False → tapis + masukkan ke database
    """
    try:
        dfs = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            temp = pd.read_csv(tmp_path, header=None, dtype=str, encoding='utf-8', skip_blank_lines=True)
            row6 = temp.iloc[5].fillna("")
            row7 = temp.iloc[6].fillna("")

            combined_header = []
            for h6, h7 in zip(row6, row7):
                combined_header.append(str(h7).strip() or str(h6))

            temp.columns = combined_header
            temp = temp.iloc[7:].reset_index(drop=True)
            dfs.append(temp)
            os.remove(tmp_path)

        # === GABUNG SEMUA CSV ===
        df = pd.concat(dfs, ignore_index=True)
        df.columns = [clean_header(c) for c in df.columns]

        # Buang column tak perlu
        drop_cols = ["Exit", "Class", "Exceptional"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        # Susunan column akhir
        final_cols = [
            "TrxNo", "PlazaNo", "LaneNo", "EntryPlaza", "JobNo", "TransactionDateTime",
            "Trx", "AVC", "PaymentMode", "FareAmount", "MfgNoTagID", "PaidAmount",
            "Balance", "AccountType", "VehicleNo", "Code", "Remark", "PenaltyCode"
        ]
        final_cols = [c for c in final_cols if c in df.columns]
        df = df[final_cols]

        # === PEMBERSIH NILAI NUMERIC ===
        money_cols = ["FareAmount", "PaidAmount", "Balance"]
        for col in money_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .replace(r"[^\d.\-]", "", regex=True)
                    .replace(r"^\s*$", np.nan, regex=True)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # === FORMAT SEMULA TARIKH ===
        if "TransactionDateTime" in df.columns:
            df["TransactionDateTime"] = (
                df["TransactionDateTime"]
                .astype(str)
                .str.replace(r"\s*(AM|PM)", "", regex=True)
                .str.strip()
            )
            df["TransactionDateTime"] = pd.to_datetime(df["TransactionDateTime"], dayfirst=True, errors="coerce")
            df["TransactionDateTime"] = df["TransactionDateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Bersih whitespace & nilai kosong
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df.replace(
            ["", " ", "NaN", "nan", "NULL", "null", "None", "N/A", "-", "--"],
            np.nan,
            inplace=True,
        )

        non_numeric_cols = [c for c in df.columns if c not in money_cols]
        df[non_numeric_cols] = df[non_numeric_cols].fillna("NULL")

        # === AMBIL DATA SEDIA ADA ===
        try:
            existing_df = pd.read_sql("SELECT * FROM public.sde22", engine)
        except Exception:
            existing_df = pd.DataFrame(columns=df.columns)

        # === SAMAKAN STRUKTUR ===
        for col in df.columns:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        existing_df = existing_df[df.columns]

        # === TUKAR SEMUA KE STRING ===
        df = df.astype(str)
        existing_df = existing_df.astype(str)

        # === BUANG DUPLIKAT (PERBANDINGAN PENUH SATU BARIS) ===
        merged_df = df.merge(
            existing_df.drop_duplicates(),
            on=list(df.columns),
            how='left',
            indicator=True
        )
        new_rows = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # === SIMPAN CSV HASIL TAPIS ===
        save_folder = r"C:\Users\awang\Compiler\filter"
        os.makedirs(save_folder, exist_ok=True)
        temp_csv_path = os.path.join(save_folder, "filtered_new_rows.csv")
        new_rows.to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
        print(f"📁 Filtered CSV saved at: {temp_csv_path}")

        # === DEBUG ===
        print(f"💾 Existing rows: {len(existing_df)}")
        print(f"🆕 New rows before filtering: {len(df)}")
        print(f"✅ New unique rows to insert: {len(new_rows)}")

        if new_rows.empty:
            return {
                "status": "success",
                "rows": 0,
                "message": "Semua data telah wujud, tiada data baru dimasukkan.",
                "download": None
            }

        # === KALAU MODE PREVIEW ===
        if preview:
            return {
                "status": "preview",
                "rows": len(new_rows),
                "message": f"{len(new_rows)} rekod baru ditemui (belum dimasukkan).",
                "download": "/download-filtered"
            }

        # === MASUKKAN DATA BARU ===
        new_rows.to_sql("sde22", engine, schema="public", if_exists="append", index=False)
        # === DOUBLE CHECK: BUANG DUPLIKAT DALAM DATABASE ===
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TEMP TABLE temp_sde22 AS
                SELECT DISTINCT * FROM public.sde22;
            """))
            conn.execute(text("TRUNCATE TABLE public.sde22;"))
            conn.execute(text("""
                INSERT INTO public.sde22
                SELECT * FROM temp_sde22;
            """))

        return {
            "status": "success",
            "rows": len(new_rows),
            "message": "Success",
            "download": "/download-filtered"
        }

    except Exception as e:
        import traceback
        print("❌ ERROR:", traceback.format_exc())
        return {"status": "error", "message": str(e)}


# === /download-filtered ENDPOINT ===
@app.get("/download-filtered")
async def download_filtered():
    save_folder = r"C:\Users\awang\Compiler\filter"
    base_name = "filtered_new_rows"
    ext = ".csv"
    temp_csv_path = os.path.join(save_folder, base_name + ext)

    if not os.path.exists(temp_csv_path):
        return {"status": "error", "message": "Tiada fail hasil tapis ditemui."}

    counter = 1
    new_path = temp_csv_path
    while os.path.exists(new_path):
        new_path = os.path.join(save_folder, f"{base_name}_{counter}{ext}")
        counter += 1

    import shutil
    shutil.copy(temp_csv_path, new_path)

    return FileResponse(
        new_path,
        filename=os.path.basename(new_path),
        media_type="text/csv"
    )


# === /tng-data ENDPOINT ===
@app.get("/tng-data")
def get_tng_data(
    start_date: str = None,
    end_date: str = None,
    filter_6am: bool = True,
    plazas: str = Query(None, description="Senarai PlazaNo dipisah koma"),
    limit: int = 100000,
    offset: int = 0
):
    try:
        start_dt, end_dt = parse_date_range(start_date, end_date, filter_6am)

        query = """
            SELECT
                "Trx", "TrxNo", "PlazaNo", "EntryPlaza", "LaneNo",
                "TransactionDateTime", "PaidAmount", "MfgNoTagID",
                "FareAmount", "VehicleNo", "PaymentMode",
                "Balance", "Code", "PenaltyCode", "Remark", "AVC"
            FROM public.sde22
            WHERE "PaymentMode" = 'TNG'
              AND "TransactionDateTime" >= :start
              AND "TransactionDateTime" < :end
        """
        params = {"start": start_dt, "end": end_dt}

        if plazas:
            plaza_list = [p.strip() for p in plazas.split(",") if p.strip()]
            placeholders = ", ".join([f":p{i}" for i in range(len(plaza_list))])
            for i, val in enumerate(plaza_list):
                params[f"p{i}"] = val
            query += f' AND "PlazaNo" IN ({placeholders})'

        query += ' ORDER BY "TransactionDateTime" LIMIT :limit OFFSET :offset'
        params.update({"limit": limit, "offset": offset})

        df = pd.read_sql(text(query), engine, params=params)

        chart_entry = df.groupby("EntryPlaza").size().reset_index(name="total").to_dict(orient="records") if not df.empty else []
        chart_plaza = df.groupby("PlazaNo")["PaidAmount"].sum().reset_index().to_dict(orient="records") if not df.empty else []
        chart_avc = df.groupby("AVC").size().reset_index(name="total").to_dict(orient="records") if "AVC" in df.columns and not df.empty else []

        return {
            "status": "success",
            "count": len(df),
            "data": df.to_dict(orient="records"),
            "chart_entry": chart_entry,
            "chart_plaza": chart_plaza,
            "chart_avc": chart_avc
        }
    except Exception:
        import traceback
        print("❌ ERROR in /tng-data:\n", traceback.format_exc())
        return {"status": "error", "message": "Ralat semasa ambil data."}


# === /plaza-list ENDPOINT ===
@app.get("/plaza-list")
def get_plaza_list():
    try:
        query = """
            SELECT DISTINCT "PlazaNo"
            FROM public.sde22
            WHERE "PlazaNo" IS NOT NULL
            ORDER BY "PlazaNo";
        """
        df = pd.read_sql(text(query), engine)
        plaza_list = df["PlazaNo"].tolist()
        return {"plazaList": plaza_list}
    except Exception:
        import traceback
        print("❌ ERROR in /plaza-list:\n", traceback.format_exc())
        return {"status": "error", "message": "Ralat semasa ambil plaza list."}


# === /wtng ENDPOINT ===
# === /wtng ENDPOINT (gabung semua + filter 6AM hanya utk TNG) ===
@app.get("/wtng")
def get_wtng_data(
    start_date: str = None,
    end_date: str = None,
    plazas: str = Query(None, description="Senarai PlazaNo dipisah koma"),
    payment_modes: str = Query(None, description="Senarai PaymentMode dipisah koma"),
    limit: int = 100000,
    offset: int = 0
):
    try:
        # 1️⃣ Buat dua range masa
        start_dt_normal, end_dt_normal = parse_date_range(start_date, end_date, filter_6am=False)
        start_dt_tng, end_dt_tng = parse_date_range(start_date, end_date, filter_6am=True)

        # 2️⃣ Query untuk data bukan TNG (masa biasa)
        query_non_tng = """
            SELECT
                "Trx", "TrxNo", "PlazaNo", "EntryPlaza", "LaneNo",
                "TransactionDateTime", "PaidAmount", "MfgNoTagID",
                "FareAmount", "VehicleNo", "PaymentMode",
                "Balance", "Code", "PenaltyCode", "Remark", "AVC"
            FROM public.sde22
            WHERE "PaymentMode" != 'TNG'
              AND "TransactionDateTime" >= :start_normal
              AND "TransactionDateTime" < :end_normal
        """

        # 3️⃣ Query untuk data TNG (masa 6AM)
        query_tng = """
            SELECT
                "Trx", "TrxNo", "PlazaNo", "EntryPlaza", "LaneNo",
                "TransactionDateTime", "PaidAmount", "MfgNoTagID",
                "FareAmount", "VehicleNo", "PaymentMode",
                "Balance", "Code", "PenaltyCode", "Remark", "AVC"
            FROM public.sde22
            WHERE "PaymentMode" = 'TNG'
              AND "TransactionDateTime" >= :start_tng
              AND "TransactionDateTime" < :end_tng
        """

        # 4️⃣ Apply plaza & payment mode filter (optional)
        plaza_clause = ""
        payment_clause = ""
        params = {
            "start_normal": start_dt_normal,
            "end_normal": end_dt_normal,
            "start_tng": start_dt_tng,
            "end_tng": end_dt_tng
        }

        if plazas:
            plaza_list = [p.strip() for p in plazas.split(",") if p.strip()]
            placeholders = ", ".join([f":p{i}" for i in range(len(plaza_list))])
            for i, val in enumerate(plaza_list):
                params[f"p{i}"] = val
            plaza_clause = f' AND "PlazaNo" IN ({placeholders})'

        if payment_modes:
            pm_list = [p.strip() for p in payment_modes.split(",") if p.strip()]
            placeholders_pm = ", ".join([f":pm{i}" for i in range(len(pm_list))])
            for i, val in enumerate(pm_list):
                params[f"pm{i}"] = val
            payment_clause = f' AND "PaymentMode" IN ({placeholders_pm})'

        # Tambah klausa filter (kalau ada)
        query_non_tng += plaza_clause + payment_clause
        query_tng += plaza_clause + payment_clause

        # 5️⃣ Gabungkan hasil dua query
        full_query = f"""
            ({query_tng})
            UNION ALL
            ({query_non_tng})
            ORDER BY "TransactionDateTime"
            LIMIT :limit OFFSET :offset
        """
        params.update({"limit": limit, "offset": offset})

        df = pd.read_sql(text(full_query), engine, params=params)

        # 6️⃣ Carta & output
        chart_entry = df.groupby("EntryPlaza").size().reset_index(name="total").to_dict(orient="records") if not df.empty else []
        chart_plaza = df.groupby("PlazaNo")["PaidAmount"].sum().reset_index().to_dict(orient="records") if not df.empty else []
        chart_avc = df.groupby("AVC").size().reset_index(name="total").to_dict(orient="records") if "AVC" in df.columns and not df.empty else []

        return {
            "status": "success",
            "count": len(df),
            "data": df.to_dict(orient="records"),
            "chart_entry": chart_entry,
            "chart_plaza": chart_plaza,
            "chart_avc": chart_avc
        }

    except Exception:
        import traceback
        print("❌ ERROR in /wtng:\n", traceback.format_exc())
        return {"status": "error", "message": "Ralat semasa ambil data WTNG (gabungan)."}



# === /traffic-summary ENDPOINT ===
@app.get("/traffic-summary")
def get_traffic_summary(
    start_date: str = None,
    end_date: str = None,
    plazas: str = Query(None, description="Senarai EntryPlaza dipisah koma")
):
    try:
        start_dt, end_dt = parse_date_range(start_date, end_date, filter_6am=False)

        query = """
            SELECT
                "Trx" AS class,
                "EntryPlaza",
                COUNT(*) AS total_traffic,
                SUM(COALESCE("PaidAmount",0)::numeric) AS total_paid
            FROM public.sde22
            WHERE "TransactionDateTime" >= :start
              AND "TransactionDateTime" < :end
        """
        params = {"start": start_dt, "end": end_dt}

        if plazas:
            plaza_list = [p.strip() for p in plazas.split(",") if p.strip()]
            placeholders = ", ".join([f":p{i}" for i in range(len(plaza_list))])
            for i, val in enumerate(plaza_list):
                params[f"p{i}"] = val
            query += f' AND "EntryPlaza" IN ({placeholders})'

        query += ' GROUP BY "Trx", "EntryPlaza" ORDER BY "Trx", "EntryPlaza"'

        df = pd.read_sql(text(query), engine, params=params)
        if df.empty:
            return {"status": "success", "data": [], "columns": []}

        traffic_pivot = df.pivot_table(
            index="class", columns="EntryPlaza", values="total_traffic", aggfunc="sum", fill_value=0
        )
        paid_pivot = df.pivot_table(
            index="class", columns="EntryPlaza", values="total_paid", aggfunc="sum", fill_value=0
        )

        all_plazas = sorted(df["EntryPlaza"].unique())
        table_data = []

        for cls in traffic_pivot.index:
            row = {"class": cls}
            for plaza in all_plazas:
                row[f"{plaza}_traffic"] = int(traffic_pivot.loc[cls].get(plaza, 0))
                row[f"{plaza}_paid"] = float(paid_pivot.loc[cls].get(plaza, 0))
            table_data.append(row)

        return {"status": "success", "data": table_data, "columns": all_plazas}

    except Exception:
        import traceback
        print("❌ ERROR in /traffic-summary:\n", traceback.format_exc())
        return {"status": "error", "message": "Ralat semasa ambil traffic summary."}

@app.get("/")
def read_root():
  return {"message": "Backend SDE22 is running successfully!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
