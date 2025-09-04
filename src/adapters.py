from __future__ import annotations
import pandas as pd


class ShopifyAdapter:
    """Normalize common Shopify export column names to canonical schema.

    Canonical (items): order_id, created_at, customer_email, sku, variant_id,
    product_title, quantity, line_item_price, line_item_discount, line_item_id
    """

    ITEMS_MAP = {
        # order id
        'name': 'order_id', 'order': 'order_id', 'order name': 'order_id',
        'order id': 'order_id', 'order_number': 'order_id', 'id_order': 'order_id',
        # sku
        'sku': 'sku', 'lineitem sku': 'sku', 'variant sku': 'sku', 'product sku': 'sku',
        'item_sku': 'sku', 'product_code': 'sku',
        # variant id
        'variant id': 'variant_id', 'variant id ': 'variant_id', 'lineitem variant_id': 'variant_id',
        'product_variant_id': 'variant_id',
        # quantity
        'lineitem quantity': 'quantity', 'qty': 'quantity', 'qty ': 'quantity', 'amount': 'quantity',
        'units': 'quantity', 'item_quantity': 'quantity',
        # price
        'lineitem price': 'line_item_price', 'price': 'line_item_price', 'item price': 'line_item_price',
        'unit price': 'line_item_price', 'unit_price': 'line_item_price', 'item_total': 'line_item_price', 'subtotal': 'line_item_price',
        # discount
        'lineitem discount': 'line_item_discount', 'discount': 'line_item_discount',
        'item discount': 'line_item_discount', 'discount_amount': 'line_item_discount',
        # product identifiers
        # Keep a canonical product_id column if present (support common aliases)
        'product_id': 'product_id', 'product id': 'product_id', 'productid': 'product_id',
        # Product title/name aliases
        'lineitem name': 'product_title', 'product': 'product_title', 'product name': 'product_title',
        'title': 'product_title', 'item': 'product_title', 'description': 'product_title', 'product_name': 'product_title',
        # created at
        'created at': 'created_at', 'date': 'created_at', 'order date': 'created_at', 'ordered at': 'created_at',
        'order_created_at': 'created_at',
        # email
        'email': 'customer_email', 'customer email': 'customer_email', 'billing email': 'customer_email',
        # ids
        'lineitem id': 'line_item_id', 'item id': 'line_item_id', 'id': 'line_item_id', 'order_item_id': 'line_item_id',
    }

    @classmethod
    def normalize_items(cls, df: pd.DataFrame) -> pd.DataFrame:
        cols = {str(c).strip().lower(): c for c in df.columns}
        rename = {}
        for lc, canon in cls.ITEMS_MAP.items():
            if lc in cols:
                rename[cols[lc]] = canon
        out = df.rename(columns=rename).copy()
        # If multiple source columns map to the same canonical name, deduplicate keeping the first
        try:
            if out.columns.duplicated().any():
                out = out.loc[:, ~out.columns.duplicated()]
        except Exception:
            pass
        # Best-effort normalize dates and order id name formatting
        if 'created_at' in out.columns:
            out['created_at'] = pd.to_datetime(out['created_at'], errors='coerce')
        # Map order_name like #1001 to 1001
        if 'order_id' in out.columns:
            out['order_id'] = out['order_id'].astype(str).str.replace('#', '', regex=False)
        else:
            # Try to derive order_id from common alternatives if missing
            for col in ['Order ID', 'Name', 'Order Name']:
                if col in df.columns:
                    out['order_id'] = df[col].astype(str).str.replace('#', '', regex=False)
                    break
        # Ensure product_id exists; fallback to multiple candidates
        if 'product_id' not in out.columns:
            # First try explicit product id style columns from original df
            for col in ['product_id', 'Product ID', 'Lineitem name', 'Product']:
                if col in df.columns:
                    out['product_id'] = df[col].astype(str)
                    break
        if 'product_id' not in out.columns:
            # Then try SKU-based identifiers
            for col in ['sku', 'SKU', 'Variant SKU', 'Lineitem sku']:
                if col in df.columns:
                    out['product_id'] = df[col].astype(str)
                    break
        if 'product_id' not in out.columns and 'product_title' in out.columns:
            # As a last resort, use title as identifier
            out['product_id'] = out['product_title'].astype(str)
        # Ensure we have a product_title for downstream debug/UX; fallback to product_id
        if 'product_title' not in out.columns and 'product_id' in out.columns:
            try:
                out['product_title'] = out['product_id'].astype(str)
            except Exception:
                out['product_title'] = out['product_id']
        # Coerce numeric fields
        for numc in ['quantity', 'line_item_price', 'line_item_discount']:
            if numc in out.columns:
                # Ensure we are operating on a 1-D Series even if duplicates slipped through
                try:
                    series = out[numc]
                    if hasattr(series, 'ndim') and getattr(series, 'ndim', 1) > 1:
                        series = series.iloc[:, 0]
                    out[numc] = pd.to_numeric(series, errors='coerce')
                except Exception:
                    # Fallback: leave as-is if coercion fails
                    pass
        return out
