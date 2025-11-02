# Version Comparison Tool

**Version Comparison Tool**  
A lightweight Python + Streamlit application to compare OLD and NEW store CSV files.  
It matches stores intelligently using fuzzy address comparison, store IDs, phone numbers, and geo-coordinates.

---

## 📁 Project Structure

```
Version_Comparision_automation/
├─ compare_engine.py        # backend matching logic
├─ compare_ui.py            # Streamlit UI
├─ requirements.txt
├─ README.md
├─ old/                     # optional for batch runs
├─ new/                     # optional for batch runs
└─ compared_data/           # outputs (auto-created)
```

---

## ⚙️ Features
- Fuzzy address comparison (handles small text variations)
- Supports multiple identifiers: store id, phone number, address, geo-coordinates
- Identifies **new**, **closed**, and **matched** locations
- Merges changed data fields intelligently (doesn’t overwrite city/state/zip)
- Streamlit-based UI for file uploads
- Batch comparison by brand (filename prefix)
- Safe CSV writing (no overwrites)

---

## 🧩 Matching Logic
1. **Store Number match** → Highest priority  
2. If no Store Number → compare **address** (fuzzy match on street, city, state, zip)  
3. If address partially differs → use **phone number** or **latitude/longitude** as fallback corroboration  
4. If OLD location not found in NEW → marked `closed`  
   If NEW not found in OLD → marked `new`  
   If both match → marked `matched` (with `changed_columns`)

---

## 📦 Installation
1. Clone this repo or unzip the project folder:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit UI:
   ```bash
   streamlit run compare_ui.py
   ```

3. Upload one or more OLD/NEW CSVs → download compared CSV.

---

## 🧠 Output Columns
| Column | Description |
|--------|--------------|
| flag | new / closed / matched |
| changed_columns | List of columns that were updated |
| ... | All other columns from OLD schema (merged) |

---

## 🔧 Configuration
Edit top-level tunables in `compare_engine.py`:
```python
PAIRING_STRATEGY = "closest-prev"   # or "same-date"
MAX_METERS = 200                    # geo corroboration distance
ADDR_STRICT = 0.90                  # fuzzy match strict threshold
ADDR_LOOSE = 0.75                   # loose threshold for phone/geo tie-breaks
```

---

## 🧰 Requirements
```
streamlit>=1.18.0
pandas>=1.3
numpy
```

---

## 💡 Tips
- If `address1` or `address_1` is used instead of `address`, it will be auto-mapped.
- Slight differences in punctuation or casing won’t break address matching.
- City/State/Zip are never overwritten by NEW data.
- All outputs are saved in the `compared_data/` folder.

---

## 🧭 Example Run (CLI)
For folder-based automation:
```
python compare_engine.py
```

---

## 🪪 License
MIT License — free to modify and use.

---

**Author:** Prachi Kabra 

