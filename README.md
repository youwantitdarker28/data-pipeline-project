# Open-Source ETL Pipeline Project

A robust Python-based ETL (Extract, Transform, Load) pipeline that automates fetching data from a public API, cleaning it with Pandas, and storing it in a structured SQLite database.

## 🚀 Project Overview

This project demonstrates a production-ready structure for data engineering. It separates concerns into modular scripts for extraction, transformation, and loading.

## Key Features

- **Automated Extraction**: Fetches live data using requests library
- **Data Transformation**: Uses pandas to clean and structure raw JSON data
- **Relational Storage**: Loads data into SQLite via sqlalchemy
- **Modular Architecture**: Logic is organized in a /scripts directory

## 🛠️ Technologies Used

- Python 3.x
- Pandas
- SQLAlchemy
- Requests
- SQLite

## 📂 Repository Structure

```
main.py              # Entry point for full execution
/scripts/            # Modular ETL logic
requirements.txt      # Project dependencies
.gitignore           # Blocks local data/databases from GitHub
```

## ⚙️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline:**
   ```bash
   python main.py
   ```
