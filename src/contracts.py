from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from .utils import identity_coverage, categorize_product, normalize_product_name


@dataclass
class IngestionContract:
    orders_df: pd.DataFrame
    items_df: Optional[pd.DataFrame] = None

    def data_quality(self) -> Dict[str, Any]:
        d = {}
        odf = self.orders_df if self.orders_df is not None else pd.DataFrame()
        idf = self.items_df if self.items_df is not None else pd.DataFrame()

        # Identity coverage on orders
        try:
            d.update(identity_coverage(odf))
        except Exception:
            d.update({
                'identity_coverage': 0.0,
                'orders_with_customer_key': 0,
                'orders_total': int(len(odf))
            })

        # Line items presence
        has_line_items = bool(idf is not None and not getattr(idf, 'empty', True))
        d['has_line_items'] = has_line_items

        # SKU presence
        try:
            if has_line_items:
                d['has_sku'] = bool(any(c in idf.columns for c in ['sku','variant_id','product_id']))
            else:
                low = {str(c).strip().lower(): c for c in odf.columns}
                d['has_sku'] = bool(any(k in low for k in ['sku','variant sku','variant_id']))
        except Exception:
            d['has_sku'] = False

        # Product coverage on orders (whether we can infer a product token per order)
        product_coverage = 0.0
        try:
            if 'products_concat' in odf.columns:
                obj = odf['products_concat']
                s = (obj if isinstance(obj, pd.Series) else pd.Series(obj, index=odf.index)).astype(str).str.strip()
                product_coverage = float((s != '').mean())
            elif 'Lineitem name' in odf.columns:
                obj = odf['Lineitem name']
                s = (obj if isinstance(obj, pd.Series) else pd.Series(obj, index=odf.index)).astype(str).str.strip()
                product_coverage = float((s != '').mean())
        except Exception:
            pass
        d['product_coverage'] = product_coverage

        return d


class FeatureContract:
    @staticmethod
    def build_g_orders(orders_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Return an order-level frame enriched with:
          - primary_product: best product key per order
          - products_concat: top-3 product keys joined by '|'
          - flags: has_sample, has_supplement (best-effort)

        Accepts order-level or line-level inputs. If `items_df` provided, prefer it.
        """
        meta: Dict[str, Any] = {"source": "orders_only", "n_orders": 0}
        if orders_df is None or len(orders_df) == 0:
            return pd.DataFrame(), meta

        d = orders_df.copy()
        # Robustly ensure 'Name' is a string Series aligned to index
        try:
            name_like = d['Name'] if 'Name' in d.columns else d.index
            d['Name'] = (name_like if isinstance(name_like, pd.Series) else pd.Series(name_like, index=d.index)).astype(str)
        except Exception:
            d['Name'] = d.index.astype(str)

        # Start with no products and fill based on availability
        d['primary_product'] = ''
        d['products_concat'] = ''
        d['products_concat_qty'] = ''
        d['products_struct'] = None
        d['category_mode_qty'] = None
        d['category_value_dom'] = None

        # Helper: tokens from items_df
        if items_df is not None and not getattr(items_df, 'empty', True):
            idf = items_df.copy()
            # Choose product key priority: sku -> product_id -> variant_id -> product_title
            key = None
            for c in ['sku','product_id','variant_id','product_title']:
                if c in idf.columns:
                    key = c; break
            qty = pd.to_numeric(idf.get('quantity', 1), errors='coerce').fillna(1)
            # Robust product key series (avoid .astype on a python str default)
            if key is not None:
                key_series = idf[key].astype(str)
            elif 'product_title' in idf.columns:
                key_series = idf['product_title'].astype(str)
            else:
                key_series = pd.Series(idf.index.astype(str), index=idf.index)
            idf['_k'] = key_series
            idf['_q'] = qty
            # Prepare normalization helpers
            if 'product_title' in idf.columns:
                idf['_title'] = idf['product_title'].astype(str)
            else:
                idf['_title'] = idf['_k']

            def _order_struct(x: pd.DataFrame) -> pd.Series:
                if x.empty:
                    return pd.Series({'primary_product': '', 'products_concat': '', 'products_concat_qty': '', 'products_struct': [], 'category_mode_qty': None})
                by_key = x.groupby('_k')['_q'].sum().sort_values(ascending=False)
                top_keys = by_key.head(3)
                primary = str(top_keys.index[0]) if len(top_keys) else ''
                concat = '|'.join([str(k) for k in top_keys.index])
                concat_qty = '|'.join([f"{k}:{int(q)}" for k, q in top_keys.items()])
                first_title = x.groupby('_k')['_title'].first()
                struct = []
                for k, q in top_keys.items():
                    t = str(first_title.get(k, k))
                    b, sz = normalize_product_name(t)
                    struct.append({'product_key': str(k), 'title': t, 'qty': int(q), 'base': b, 'size': sz})
                # Category mode by qty using categorize_product on title
                cats = x.assign(_cat=x['_title'].apply(lambda s: categorize_product(s)[0]))
                counts = cats.groupby('_cat')['_q'].sum().sort_values(ascending=False)
                cat_mode = str(counts.index[0]) if len(counts) else None
                return pd.Series({'primary_product': primary, 'products_concat': concat, 'products_concat_qty': concat_qty, 'products_struct': struct, 'category_mode_qty': cat_mode})

            grp = (idf.groupby('order_id').apply(_order_struct).reset_index())
            d = d.merge(grp, left_on='Name', right_on='order_id', how='left')
            d.drop(columns=['order_id'], inplace=True, errors='ignore')
            meta['source'] = 'items_df'
        else:
            # Derive from orders frame: gather first 3 distinct lineitem names per order
            if 'Lineitem name' in d.columns:
                tmp = d[['Name','Lineitem name']].copy()
                tmp['Lineitem name'] = tmp['Lineitem name'].astype(str)
                # Default qty=1 per row; aggregate by product
                agg_qty = (tmp.assign(_q=1)
                             .groupby(['Name','Lineitem name'])['_q'].sum().reset_index())
                def _mk_struct(dfN: pd.DataFrame) -> pd.Series:
                    by = dfN.sort_values('_q', ascending=False)
                    top = by.head(3)
                    primary = str(top['Lineitem name'].iloc[0]) if len(top) else ''
                    concat = '|'.join(top['Lineitem name'].astype(str))
                    concat_qty = '|'.join([f"{str(r['Lineitem name'])}:{int(r['_q'])}" for _, r in top.iterrows()])
                    struct = []
                    for _, r in top.iterrows():
                        t = str(r['Lineitem name'])
                        b, sz = normalize_product_name(t)
                        struct.append({'product_key': t, 'title': t, 'qty': int(r['_q']), 'base': b, 'size': sz})
                    # approximate category via product tokens
                    cats = dfN.assign(_cat=dfN['Lineitem name'].astype(str).apply(lambda s: categorize_product(s)[0]))
                    counts = cats.groupby('_cat')['_q'].sum().sort_values(ascending=False)
                    cat_mode = str(counts.index[0]) if len(counts) else None
                    return pd.Series({'primary_product': primary, 'products_concat': concat, 'products_concat_qty': concat_qty, 'products_struct': struct, 'category_mode_qty': cat_mode})
                grp2 = agg_qty.groupby('Name').apply(_mk_struct).reset_index()
                d = d.merge(grp2, on='Name', how='left')
                meta['source'] = 'orders_frame'

        # Flags
        try:
            pp = d['primary_product'] if 'primary_product' in d.columns else pd.Series('', index=d.index)
            prod_series = (pp if isinstance(pp, pd.Series) else pd.Series(pp, index=d.index)).astype(str).str.lower()
        except Exception:
            prod_series = pd.Series('', index=d.index)
        d['has_sample'] = prod_series.str.contains(r"sample|travel|mini|trial", regex=True, na=False)
        try:
            # Supplement detection via categorize_product
            d['has_supplement'] = prod_series.apply(lambda s: categorize_product(s)[0] == 'supplement')
        except Exception:
            d['has_supplement'] = False

        meta['n_orders'] = int(d['Name'].nunique()) if 'Name' in d.columns else int(len(d))
        # Prefer category_mode_qty when present
        if 'category_mode_qty' in d.columns:
            d['category'] = d['category_mode_qty'].fillna(d.get('category'))
        return d, meta
