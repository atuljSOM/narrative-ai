
from __future__ import annotations
import pandas as pd, numpy as np
from typing import Dict, Any, Tuple
from .adapters import ShopifyAdapter
from pathlib import Path
from .utils import (
    winsorize_series, write_json, load_category_map, dominant_category_for_order,
    standardize_customer_key, standardize_order_key,
)

MONETARY = ["Subtotal","Total Discount","Shipping","Taxes","Total","Lineitem price","Lineitem discount"]
REQUIRED = [
    "Name","Created at","Lineitem name","Lineitem quantity","Lineitem price",
    "Lineitem discount","Financial Status","Fulfillment Status","Subtotal",
    "Total Discount","Shipping","Taxes","Total","Currency",
    "Customer Email","Billing Name","Shipping Province","Shipping Country"
]

def robust_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Preserve original headers but add simple, predictable aliases for common fields
    df.columns = [c.strip() for c in df.columns]

    # Strict-but-simple header normalization: add snake_case aliases alongside originals
    # Do not overwrite any existing columns; only add when alias not present
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    def _add_alias(src_key: str, alias: str):
        if src_key in lower_map and alias not in df.columns:
            df[alias] = df[lower_map[src_key]]

    # Spaces → underscore + canonical aliases
    # Core identity/time keys
    _add_alias('created at', 'created_at')
    _add_alias('customer email', 'customer_email')
    _add_alias('name', 'order_id')  # Shopify order name → order_id
    _add_alias('order name', 'order_id')
    _add_alias('order id', 'order_id')
    # Line-item fields
    _add_alias('lineitem name', 'lineitem_name')
    _add_alias('lineitem quantity', 'lineitem_quantity')
    _add_alias('lineitem price', 'lineitem_price')
    _add_alias('lineitem discount', 'lineitem_discount')
    # Monetary fields (used in some charts/validations)
    _add_alias('total discount', 'total_discount')

    # Backfill snake_case → Title Case for canonical columns used downstream
    # This ensures charts/validations that expect Title Case work even when uploads are snake_case.
    def _backfill(source: str, target: str):
        if source in df.columns:
            if target not in df.columns:
                df[target] = df[source]
            else:
                tgt = df[target]
                src = df[source]
                empty_mask = tgt.isna() | (tgt.astype(str).str.strip() == '')
                # Align indices and fill only where target is empty
                try:
                    df.loc[empty_mask, target] = src.loc[empty_mask]
                except Exception:
                    # Fallback if alignment fails for any reason
                    df[target] = tgt.where(~empty_mask, src)

    # Identity/time
    _backfill('order_id', 'Name')
    _backfill('created_at', 'Created at')
    _backfill('customer_email', 'Customer Email')
    _backfill('cancelled_at', 'Cancelled at')
    # Line items
    _backfill('lineitem_name', 'Lineitem name')
    _backfill('lineitem_quantity', 'Lineitem quantity')
    _backfill('lineitem_price', 'Lineitem price')
    _backfill('lineitem_discount', 'Lineitem discount')
    # Monetary
    _backfill('total_discount', 'Total Discount')
    _backfill('total', 'Total')
    _backfill('shipping', 'Shipping')
    _backfill('taxes', 'Taxes')

    def _parse_money_series(s: pd.Series | None) -> pd.Series:
        """Make money columns numeric. Survives $, commas, NBSPs, unicode, (negatives)."""
        if s is None:
            return pd.Series(dtype=float)
        raw = s.astype(str)
        # detect parentheses negatives BEFORE strip
        neg_mask = raw.str.contains(r"^\s*\(.*\)\s*$", na=False)
        # strip everything except digits, dot, minus
        cleaned = raw.str.replace(r"[^\d\.\-]", "", regex=True)
        out = pd.to_numeric(cleaned, errors="coerce")
        # apply negative for parentheses cases
        out.loc[neg_mask] = -out.loc[neg_mask].abs()
        return out

    # Ensure required columns exist (fill with NaN if missing)
    for c in REQUIRED:
        if c not in df.columns:
            df[c] = np.nan

    # Parse monetary columns robustly (create all of them if absent)
    for c in MONETARY:
        if c in df.columns:
            df[c] = _parse_money_series(df[c])
        else:
            df[c] = np.nan

    # Quantities → numeric (guard)
    if "Lineitem quantity" in df.columns:
        df["Lineitem quantity"] = pd.to_numeric(df["Lineitem quantity"], errors="coerce")

    # Dates → datetime (guard)
    if "Created at" in df.columns:
        df["Created at"] = pd.to_datetime(df["Created at"], errors="coerce", utc=True).dt.tz_localize(None)
    if "Cancelled at" in df.columns:
        df["Cancelled at"] = pd.to_datetime(df["Cancelled at"], errors="coerce", utc=True).dt.tz_localize(None)

    return df


def preprocess(df: pd.DataFrame):
    qa = {}
    df = df.copy()
    # Phase 0: standardize identities early to keep KPIs, Actions, Segments consistent
    try:
        # Order key normalized to 'Name' semantics
        std_order = standardize_order_key(df)
        if std_order is not None and len(std_order) == len(df):
            df['Name'] = std_order.astype(str)
    except Exception:
        # fallback: preserve existing behavior
        pass
    try:
        # Customer key normalized (email->id->name|province)
        std_cust = standardize_customer_key(df)
        if std_cust is not None and len(std_cust) == len(df):
            df['customer_id'] = std_cust.astype(str)
    except Exception:
        pass
    # Ensure expected monetary columns exist (especially when caller bypassed robust_read_csv)
    for c in MONETARY:
        if c not in df.columns:
            df[c] = np.nan
    # Ensure cols used below exist to avoid KeyErrors in flexible loader paths
    if 'Financial Status' not in df.columns:
        df['Financial Status'] = ''
    # 'Name' and 'customer_id' already standardized above; ensure existence as strings
    if 'Name' not in df.columns:
        df['Name'] = df.index.astype(str)
    if 'customer_id' not in df.columns:
        df['customer_id'] = df.get('Customer Email', pd.Series('', index=df.index)).astype(str).str.strip().str.lower()
    for c in MONETARY:
        df[c] = winsorize_series(df[c])
    df["net_sales"] = df["Subtotal"] - df["Total Discount"]
    df["discount_rate"] = (df["Total Discount"] / df["Subtotal"]).replace([np.inf,-np.inf], np.nan)
    df["units_per_order"] = df["Lineitem quantity"]
    mask_refund = df["Financial Status"].astype(str).str.lower().str.contains("refunded|chargeback")
    mask_test = df["Name"].astype(str).str.contains("test", case=False, na=False)
    before = len(df); df = df[~(mask_refund|mask_test)].copy()
    qa["excluded_refund_or_test"] = before - len(df)
    qa["raw_rows"] = int(before); qa["final_rows"] = int(len(df))
    qa["min_date"] = str(pd.to_datetime(df["Created at"]).min())
    qa["max_date"] = str(pd.to_datetime(df["Created at"]).max())

    # --- Category tagging (dominant per order) ---
    try:
        cat_map = load_category_map()
        if cat_map and 'Name' in df.columns:
            cats: dict[str, str] = {}
            for name, rows in df.groupby('Name'):
                cats[name] = dominant_category_for_order(rows, cat_map)
            df['category'] = df['Name'].map(cats).fillna('unknown')
        else:
            df['category'] = 'unknown'
    except Exception:
        df['category'] = 'unknown'
    return df, qa

def load_csv(path: str, qa_out_path: str|None=None):
    df = robust_read_csv(path); df, qa = preprocess(df)
    if qa_out_path: write_json(qa_out_path, qa)
    return df, qa

# -------------- Flexible Orders/Items loading -------------- #
def detect_line_item_format(df: pd.DataFrame) -> bool:
    cols = {str(c).strip().lower() for c in df.columns}
    signals = [
        'line_item_id' in cols,
        'sku' in cols or 'lineitem sku' in cols or 'variant sku' in cols,
        'variant id' in cols or 'variant_id' in cols,
        (('quantity' in cols or 'lineitem quantity' in cols) and ('product' in cols or 'lineitem name' in cols)),
    ]
    # duplicate order ids suggests line-item granularity
    order_col = None
    for k in ['order_id','name','order','order name','order id','order_number','id_order']:
        if k in cols:
            order_col = [c for c in df.columns if str(c).strip().lower() == k][0]
            break
    if order_col is not None and df[order_col].duplicated().any():
        signals.append(True)
    return sum(bool(s) for s in signals) >= 2

def load_order_items_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    items = ShopifyAdapter.normalize_items(raw)
    return items

def normalize_line_item_orders(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Produce (orders_df, items_df) from a line-item level CSV.
    orders_df retains denormalized columns for downstream compatibility and includes line_items_json.
    """
    items = ShopifyAdapter.normalize_items(df)
    items_df = items.copy()
    # Ensure numeric unit price, quantity, and discount
    if 'line_item_price' not in items_df.columns and 'Lineitem price' in df.columns:
        items_df['line_item_price'] = pd.to_numeric(df['Lineitem price'], errors='coerce')
    items_df['line_item_price'] = pd.to_numeric(items_df.get('line_item_price', 0), errors='coerce').fillna(0.0)
    items_df['quantity'] = pd.to_numeric(items_df.get('quantity', 1), errors='coerce').fillna(1)
    items_df['line_item_discount'] = pd.to_numeric(items_df.get('line_item_discount', 0), errors='coerce').fillna(0.0)
    # Compute per-line totals (unit price × qty) and net after discount
    items_df['_line_total'] = items_df['line_item_price'] * items_df['quantity']
    items_df['_line_discount'] = items_df['line_item_discount']
    items_df['_line_net'] = items_df['_line_total'] - items_df['_line_discount']
    # Aggregate to order-level summary
    keep_first = {}
    for c in ['customer_email', 'created_at']:
        if c in items_df.columns:
            keep_first[c] = 'first'
    agg = items_df.groupby('order_id').agg({
        **keep_first,
        '_line_total': 'sum',
        '_line_discount': 'sum',
        '_line_net': 'sum',
        'quantity': 'sum'
    }).rename(columns={'_line_total': 'Subtotal', '_line_discount': 'Total Discount', '_line_net': 'net_sales_items', 'quantity': 'units_total'})
    li_cols = [c for c in ['sku','quantity','line_item_price','line_item_discount','variant_id','product_title'] if c in items_df.columns]
    line_json = (items_df.groupby('order_id')[li_cols]
                 .apply(lambda x: x.to_dict('records')).rename('line_items_json'))
    orders_df = agg.join(line_json)
    orders_df = orders_df.reset_index().rename(columns={'order_id': 'Name', 'created_at': 'Created at'})
    # Fill monetary columns used downstream (guard against missing columns returning scalars)
    orders_df['Subtotal'] = pd.to_numeric(orders_df['Subtotal'], errors='coerce').fillna(0.0)
    orders_df['Total Discount'] = pd.to_numeric(orders_df['Total Discount'], errors='coerce').fillna(0.0)
    ship_series = pd.to_numeric(orders_df['Shipping'], errors='coerce').fillna(0.0) if 'Shipping' in orders_df.columns else pd.Series(0.0, index=orders_df.index)
    tax_series = pd.to_numeric(orders_df['Taxes'], errors='coerce').fillna(0.0) if 'Taxes' in orders_df.columns else pd.Series(0.0, index=orders_df.index)
    orders_df['Shipping'] = ship_series
    orders_df['Taxes'] = tax_series
    orders_df['Total'] = orders_df['Subtotal']  # shipping/taxes not present at line level; keep neutral
    if 'Currency' not in orders_df.columns:
        orders_df['Currency'] = 'USD'
    # Put a representative item name (first) for product field usage
    if 'product_title' in items_df.columns:
        first_names = items_df.groupby('order_id')['product_title'].first()
        orders_df = orders_df.merge(first_names.rename('Lineitem name'), left_on='Name', right_index=True, how='left')
    else:
        orders_df['Lineitem name'] = ''
    orders_df['Lineitem quantity'] = pd.to_numeric(orders_df.get('units_total', 0), errors='coerce').fillna(0).astype(float)
    return orders_df, items_df

def enhance_order_level(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Normalize order identifier to 'Name'
    cols_lower = {str(c).strip().lower(): c for c in d.columns}
    if 'name' in cols_lower and 'Name' not in d.columns:
        d['Name'] = d[cols_lower['name']]
    elif 'order id' in cols_lower and 'Name' not in d.columns:
        d['Name'] = d[cols_lower['order id']]
    elif 'order_id' in cols_lower and 'Name' not in d.columns:
        d['Name'] = d[cols_lower['order_id']]
    elif 'order number' in cols_lower and 'Name' not in d.columns:
        d['Name'] = d[cols_lower['order number']]

    # Normalize created-at timestamp to 'Created at'
    created_candidates = [
        'created at','created_at','created','created date','processed at','processed_at','date','order date','order_date'
    ]
    if 'Created at' not in d.columns:
        src = None
        for key in created_candidates:
            if key in cols_lower:
                src = cols_lower[key]
                break
        if src is not None:
            d['Created at'] = d[src]
    # Ensure dtype for Created at if present
    if 'Created at' in d.columns:
        d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')

    # Map common monetary aliases into expected columns
    def _map_first(target: str, candidates: list[str]):
        if target in d.columns:
            return
        for key in candidates:
            lk = key.strip().lower()
            if lk in cols_lower:
                d[target] = d[cols_lower[lk]]
                return
        # default if no source
        d[target] = np.nan

    _map_first("Subtotal", ["subtotal", "order_subtotal", "pre_discount_total", "total_before_discount"])
    _map_first("Total Discount", ["discount_amount", "discounts", "total_discount", "discount"])
    _map_first("Shipping", ["shipping_amount", "shipping", "shipping_cost"])
    _map_first("Taxes", ["tax_amount", "tax", "taxes", "total_tax"])
    _map_first("Total", ["total_amount", "order_total", "grand_total", "amount_total"])

    # Optional identity helpers
    if 'Customer Email' not in d.columns:
        for key in ["customer_email", "email"]:
            if key in cols_lower:
                d['Customer Email'] = d[cols_lower[key]]
                break
    if 'Billing Name' not in d.columns:
        for key in ["customer_name", "billing_name", "name_customer"]:
            if key in cols_lower:
                d['Billing Name'] = d[cols_lower[key]]
                break
    if 'Shipping Country' not in d.columns:
        for key in ["shipping_country", "shipping address country", "shipping_address_country", "country", "ship_country"]:
            if key in cols_lower:
                d['Shipping Country'] = d[cols_lower[key]]
                break

    # Ensure essential columns exist post-mapping
    # Backfill snake_case line item fields into Title Case expected downstream
    def _backfill(src_key_lower: str, target_col: str):
        try:
            if src_key_lower in cols_lower:
                src_col = cols_lower[src_key_lower]
                if target_col not in d.columns:
                    d[target_col] = d[src_col]
                else:
                    tgt = d[target_col]
                    src = d[src_col]
                    empty_mask = tgt.isna() | (tgt.astype(str).str.strip() == '')
                    d.loc[empty_mask, target_col] = src.loc[empty_mask]
        except Exception:
            pass

    _backfill('lineitem_name', 'Lineitem name')
    _backfill('lineitem_quantity', 'Lineitem quantity')
    _backfill('lineitem_price', 'Lineitem price')
    _backfill('lineitem_discount', 'Lineitem discount')

    if 'Currency' not in d.columns:
        d['Currency'] = 'USD'
    if 'Lineitem name' not in d.columns:
        d['Lineitem name'] = ''
    if 'Lineitem quantity' not in d.columns:
        d['Lineitem quantity'] = 1
    return d

def extract_items_from_orders(orders_denorm: pd.DataFrame) -> pd.DataFrame | None:
    cols = orders_denorm.columns
    if 'Lineitem name' in cols and 'Lineitem quantity' in cols:
        out = pd.DataFrame({
            'order_id': orders_denorm.get('Name'),
            'product_title': orders_denorm.get('Lineitem name'),
            'quantity': pd.to_numeric(orders_denorm.get('Lineitem quantity'), errors='coerce'),
        })
        return out
    return None

def load_orders_csv(filepath: str, has_line_items: bool | None = None) -> Tuple[pd.DataFrame, bool, pd.DataFrame | None]:
    raw = pd.read_csv(filepath, low_memory=False)
    if has_line_items is None:
        has_line_items = detect_line_item_format(raw)
    if has_line_items:
        orders_df, items_df = normalize_line_item_orders(raw)
        return orders_df, True, items_df
    else:
        orders_df = enhance_order_level(raw)
        return orders_df, False, None

# --- Inventory ingestion ---
def load_inventory_csv(path: str) -> pd.DataFrame:
    """Load and normalize a Shopify Inventory export.
    Expected columns (any of these aliases):
      - SKU / Variant SKU -> sku
      - Variant ID / Variantid -> variant_id
      - Product Title / Title -> product
      - Available / Inventory Quantity / On Hand -> available
      - Incoming -> incoming (default 0)
      - Updated At / Last Updated -> updated_at
      - Location (optional), Reorder Point (optional)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Inventory file not found: {path}")
    df = pd.read_csv(p)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    sku_c = pick('sku', 'variant sku')
    var_c = pick('variant id', 'variantid')
    prod_c= pick('product title', 'title', 'product', 'product_name')
    avail_c= pick('available', 'inventory quantity', 'on hand', 'stock_quantity')
    inc_c = pick('incoming')
    upd_c = pick('updated at', 'last updated', 'updated_at')
    loc_c = pick('location')
    rop_c = pick('reorder point', 'reorderpoint')
    out = pd.DataFrame()
    if sku_c is not None:
        out['sku'] = df[sku_c].astype(str)
    elif var_c is not None:
        out['sku'] = df[var_c].astype(str)
    else:
        out['sku'] = df.index.astype(str)
    out['variant_id'] = df[var_c].astype(str) if var_c is not None else None
    out['product'] = df[prod_c].astype(str) if prod_c is not None else None
    out['available'] = pd.to_numeric(df[avail_c], errors='coerce') if avail_c is not None else 0
    out['incoming'] = pd.to_numeric(df[inc_c], errors='coerce') if inc_c is not None else 0
    if upd_c is not None:
        out['updated_at'] = pd.to_datetime(df[upd_c], errors='coerce')
    else:
        out['updated_at'] = pd.Timestamp.now()
    out['location'] = df[loc_c].astype(str) if loc_c is not None else None
    out['reorder_point'] = pd.to_numeric(df[rop_c], errors='coerce') if rop_c is not None else np.nan
    # coerce negatives to 0
    out['available'] = out['available'].clip(lower=0)
    out['incoming'] = out['incoming'].fillna(0).clip(lower=0)
    # aggregate duplicates by sku
    out = (out.groupby(['sku','variant_id','product'], dropna=False)
              .agg({
                  'available':'sum', 'incoming':'sum', 'updated_at':'max',
                  'location':'first', 'reorder_point':'max'
              }).reset_index())
    return out

def compute_inventory_metrics(inventory_df: pd.DataFrame, orders_df: pd.DataFrame,
                              lead_time_days: int = 14, z: float = 1.64,
                              safety_floor: int = 0) -> pd.DataFrame:
    """Compute per-SKU weighted velocity, safety stock, cover days, and trust.
    orders_df must include line items with 'Created at', 'lineitem_any' or 'SKU' if present.
    """
    inv = inventory_df.copy()
    inv['updated_at'] = pd.to_datetime(inv['updated_at'], errors='coerce')
    now = pd.Timestamp.now()
    inv['age_days'] = (now - inv['updated_at']).dt.days.clip(lower=0)
    # Build orders by day x sku
    dd = orders_df.copy()
    if 'Created at' in dd.columns:
        dd['Created at'] = pd.to_datetime(dd['Created at'], errors='coerce')
    dd = dd.dropna(subset=['Created at'])
    dd['days_ago'] = (now.normalize() - dd['Created at'].dt.normalize()).dt.days
    # Use SKU if present else fallback to lineitem name as pseudo-sku, then lineitem_any
    sku_col = None
    lower_map = {str(c).strip().lower(): c for c in dd.columns}
    for key in ['sku', 'variant sku', 'variant_id', 'lineitem name', 'product', 'product_title', 'lineitem_any']:
        if key in lower_map:
            sku_col = lower_map[key]
            break
    if sku_col is None:
        # Last resort: synthesize from order id so code runs; velocity by order_id is not ideal but safe
        sku_col = 'Name' if 'Name' in dd.columns else None
    if sku_col is not None:
        dd['_sku'] = dd[sku_col].astype(str)
    else:
        dd['_sku'] = 'unknown'
    # units per order: use Lineitem quantity if present else 1
    units = pd.to_numeric(dd.get('Lineitem quantity', pd.Series(1, index=dd.index)), errors='coerce').fillna(1)
    dd['_units'] = units
    def weighted_velocity(group):
        r7 = group.loc[group['days_ago'] <= 7, '_units'].sum() / 7.0
        o21 = group.loc[(group['days_ago'] > 7) & (group['days_ago'] <= 28), '_units'].sum() / 21.0
        return 0.7 * r7 + 0.3 * o21
    # Compute recent weighted velocity per SKU (robust reset of name column)
    vel = (
        dd[dd['days_ago'] <= 28]
        .groupby('_sku', dropna=False)
        .apply(weighted_velocity)
        .reset_index(name='daily_velocity')
    )
    inv = inv.merge(vel, left_on='sku', right_on='_sku', how='left')
    inv.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)
    inv['daily_velocity'] = inv['daily_velocity'].fillna(0.0)
    # safety stock
    inv['safety_week'] = inv['daily_velocity'] * 7.0
    inv['safety_stat'] = np.sqrt(inv['daily_velocity'].clip(lower=1.0) * float(lead_time_days)) * float(z)
    inv['safety_floor'] = float(safety_floor)
    inv['safety_stock'] = inv[['safety_week','safety_stat']].max(axis=1)
    inv['safety_stock'] = np.maximum(inv['safety_stock'], inv['safety_floor'])
    # cover days using available + incoming - safety
    inv['available_net'] = (pd.to_numeric(inv['available'], errors='coerce').fillna(0) +
                            pd.to_numeric(inv['incoming'], errors='coerce').fillna(0) -
                            pd.to_numeric(inv['safety_stock'], errors='coerce').fillna(0))
    inv['available_net'] = inv['available_net'].clip(lower=0)
    inv['cover_days'] = inv['available_net'] / inv['daily_velocity'].replace(0, np.nan)
    inv['cover_days'] = inv['cover_days'].fillna(np.inf)
    # trust factor by age
    inv['trust_factor'] = (1.0 - (inv['age_days'] / 14.0)).clip(lower=0.5, upper=1.0)
    # reorder warnings
    inv['below_reorder'] = (inv.get('reorder_point', np.nan) >= 0) & (inv['available'] <= inv.get('reorder_point', np.inf)) & (inv['incoming'] <= 0)
    return inv
