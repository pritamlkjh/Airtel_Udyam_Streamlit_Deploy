# Project Udyam Digital — Streamlit Demo

This repository contains a competition prototype for demonstrating a digital MSME discovery, quotation, KYC pre-validation, and order-tracking journey.

## Important demo limitations

- All lead records, business names, identifiers, registry entries, and KYC specimen documents in this repository are synthetic test data. They are not real Airtel customer records.
- This is a prototype, not an official Airtel production service.
- Do not upload real identity, tax, address, or customer documents to a public demo deployment.
- Registry lookups and order-tracking data are mock/demo integrations. Production use would require Airtel-approved systems, security review, and legal/compliance approval.
- The KYC specimen bench measures agreement against a small supplied synthetic test set; it is not a production accuracy or compliance certification.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

- Main file: `app.py`
- Python dependencies: `requirements.txt`
- Linux dependency for optional live OCR: `packages.txt` (`tesseract-ocr`)
- Keep the data CSV/JSON files and `kyc_documents/` beside `app.py`.

The application can use deterministic recommendation explanations without an Anthropic API key. Do not commit API keys or `.streamlit/secrets.toml` to Git.
