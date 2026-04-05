"""Frozen evaluation test bank for the V2 finetuning pipeline.

Contains 50 examples per track (300 total) covering easy/medium/hard
difficulties across diverse subcategories.  Each example is a dict with:
    track, category, prompt, reference, difficulty

Usage::

    from src.finetune.v2.eval.test_bank import get_test_bank

    all_examples = get_test_bank()
    excel_only   = get_test_bank(track="excel_csv")
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal builder helpers
# ---------------------------------------------------------------------------

def _ex(
    track: str,
    category: str,
    prompt: str,
    reference: dict,
    difficulty: str = "medium",
) -> dict:
    return {
        "track": track,
        "category": category,
        "prompt": prompt,
        "reference": reference,
        "difficulty": difficulty,
    }


# ===================================================================
# Track 1 -- Excel / CSV Intelligence  (50 examples)
# ===================================================================

def _build_excel_csv() -> List[dict]:
    items: List[dict] = []

    # -- single_sheet_lookup (10) ---
    items.append(_ex("excel_csv", "single_sheet_lookup",
        "Given the following spreadsheet data:\n\n"
        "[SPREADSHEET: employees.xlsx / Sheet1]\n"
        "| Name | Department | Salary |\n| Alice | Engineering | 95000 |\n"
        "| Bob | Marketing | 72000 |\n| Carol | Engineering | 102000 |\n"
        "| Dave | Sales | 68000 |\n\nWhat is Carol's salary?",
        {"expected_answer": "102000", "expected_values": {"Carol": "102000"},
         "data_types": ["currency"], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: inventory.csv]\n"
        "| SKU | Product | Qty | Price |\n| A100 | Widget | 340 | 12.50 |\n"
        "| A101 | Gadget | 125 | 29.99 |\n| A102 | Gizmo | 78 | 45.00 |\n\n"
        "How many Gadgets are in stock?",
        {"expected_answer": "125", "expected_values": {"Gadget": "125"},
         "data_types": [], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: contacts.xlsx / Sheet1]\n"
        "| ID | Name | Email | Phone |\n"
        "| 1 | John Smith | john@acme.com | 555-0101 |\n"
        "| 2 | Jane Doe | jane@globex.com | 555-0102 |\n"
        "| 3 | Bob Lee | bob@initech.com | 555-0103 |\n\n"
        "What is Jane Doe's email address?",
        {"expected_answer": "jane@globex.com", "expected_values": {"Jane Doe": "jane@globex.com"},
         "data_types": ["email"], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: orders.csv]\n"
        "| OrderID | Customer | Total | Status |\n"
        "| 5001 | Acme Corp | 15400 | Shipped |\n"
        "| 5002 | Globex | 8200 | Pending |\n"
        "| 5003 | Initech | 22100 | Delivered |\n"
        "| 5004 | Umbrella | 3100 | Cancelled |\n\n"
        "What is the status of order 5003?",
        {"expected_answer": "Delivered", "expected_values": {"5003": "Delivered"},
         "data_types": [], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: grades.xlsx / Sheet1]\n"
        "| Student | Math | Science | English |\n"
        "| Kim | 92 | 88 | 95 |\n"
        "| Lee | 78 | 91 | 84 |\n"
        "| Park | 85 | 79 | 90 |\n\n"
        "What did Lee score in Science?",
        {"expected_answer": "91", "expected_values": {"Lee": "91"},
         "data_types": [], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: assets.xlsx / Sheet1]\n"
        "| AssetID | Type | Location | Value |\n"
        "| AST-001 | Laptop | NYC | 1200 |\n"
        "| AST-002 | Monitor | CHI | 450 |\n"
        "| AST-003 | Desk | NYC | 800 |\n"
        "| AST-004 | Chair | LA | 350 |\n\n"
        "What is the value of asset AST-003?",
        {"expected_answer": "800", "expected_values": {"AST-003": "800"},
         "data_types": ["currency"], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: flights.csv]\n"
        "| Flight | Origin | Dest | Dep | Arr |\n"
        "| AA100 | JFK | LAX | 08:00 | 11:30 |\n"
        "| UA200 | ORD | SFO | 09:15 | 11:45 |\n"
        "| DL300 | ATL | BOS | 07:30 | 10:00 |\n\n"
        "What time does flight UA200 depart?",
        {"expected_answer": "09:15", "expected_values": {"UA200": "09:15"},
         "data_types": ["time"], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: meds.xlsx / Sheet1]\n"
        "| PatientID | Medication | Dosage | Frequency |\n"
        "| P001 | Lisinopril | 10mg | daily |\n"
        "| P002 | Metformin | 500mg | twice daily |\n"
        "| P003 | Amoxicillin | 250mg | 3x daily |\n\n"
        "What medication is patient P002 taking?",
        {"expected_answer": "Metformin", "expected_values": {"P002": "Metformin"},
         "data_types": ["dosage"], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: tickets.csv]\n"
        "| TicketID | Priority | Assignee | Status |\n"
        "| TK-101 | High | Sarah | Open |\n"
        "| TK-102 | Low | Bob | Closed |\n"
        "| TK-103 | Critical | Sarah | In Progress |\n"
        "| TK-104 | Medium | Tom | Open |\n\n"
        "How many tickets are assigned to Sarah?",
        {"expected_answer": "2", "expected_values": {"Sarah": "2"},
         "data_types": [], "aggregation": "2", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "single_sheet_lookup",
        "[SPREADSHEET: weather.xlsx / Sheet1]\n"
        "| City | TempC | Humidity | Conditions |\n"
        "| NYC | 18 | 65 | Partly Cloudy |\n"
        "| CHI | 12 | 72 | Rain |\n"
        "| LA | 25 | 40 | Sunny |\n"
        "| MIA | 30 | 85 | Thunderstorm |\n\n"
        "What are the conditions in Chicago?",
        {"expected_answer": "Rain", "expected_values": {"CHI": "Rain"},
         "data_types": ["temperature"], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    # -- aggregation (10) ---
    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: sales.xlsx / Sheet1]\n"
        "| Month | Revenue |\n| Jan | 12000 |\n| Feb | 15000 |\n"
        "| Mar | 18000 |\n| Apr | 14000 |\n\n"
        "What is the total revenue across all months?",
        {"expected_answer": "59000", "expected_values": {},
         "data_types": ["currency"], "aggregation": "59000", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: expenses.csv]\n"
        "| Category | Q1 | Q2 | Q3 | Q4 |\n"
        "| Rent | 5000 | 5000 | 5200 | 5200 |\n"
        "| Utilities | 800 | 750 | 900 | 850 |\n"
        "| Supplies | 1200 | 1100 | 1300 | 1400 |\n\n"
        "What is the average quarterly rent?",
        {"expected_answer": "5100", "expected_values": {},
         "data_types": ["currency"], "aggregation": "5100", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: headcount.xlsx / Sheet1]\n"
        "| Department | FTE |\n| Engineering | 45 |\n| Sales | 30 |\n"
        "| Marketing | 18 |\n| HR | 8 |\n| Finance | 12 |\n\n"
        "What is the total headcount?",
        {"expected_answer": "113", "expected_values": {},
         "data_types": [], "aggregation": "113", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: scores.csv]\n"
        "| Player | Points |\n| A | 23 |\n| B | 31 |\n| C | 18 |\n"
        "| D | 27 |\n| E | 35 |\n\nWho scored the maximum points?",
        {"expected_answer": "E", "expected_values": {"E": "35"},
         "data_types": [], "aggregation": "35", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: invoices.xlsx / Sheet1]\n"
        "| Invoice | Amount | Tax |\n| INV-001 | 5000 | 450 |\n"
        "| INV-002 | 8000 | 720 |\n| INV-003 | 3200 | 288 |\n\n"
        "What is the total amount including tax for all invoices?",
        {"expected_answer": "17658", "expected_values": {},
         "data_types": ["currency", "tax"], "aggregation": "17658", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: miles.csv]\n"
        "| Driver | Mon | Tue | Wed | Thu | Fri |\n"
        "| Amy | 45 | 62 | 38 | 55 | 70 |\n"
        "| Ben | 30 | 48 | 52 | 41 | 29 |\n\n"
        "What is Amy's total mileage for the week?",
        {"expected_answer": "270", "expected_values": {"Amy": "270"},
         "data_types": ["mileage"], "aggregation": "270", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: survey.xlsx / Sheet1]\n"
        "| Question | Strongly Agree | Agree | Neutral | Disagree | Strongly Disagree |\n"
        "| Q1 | 45 | 30 | 15 | 8 | 2 |\n"
        "| Q2 | 20 | 35 | 25 | 12 | 8 |\n\n"
        "What is the total number of respondents for Q1?",
        {"expected_answer": "100", "expected_values": {},
         "data_types": ["count"], "aggregation": "100", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: production.csv]\n"
        "| Line | Units_Produced | Defects |\n"
        "| A | 5000 | 23 |\n| B | 4800 | 45 |\n| C | 5200 | 18 |\n\n"
        "What is the overall defect rate as a percentage?",
        {"expected_answer": "0.57", "expected_values": {},
         "data_types": ["percentage", "defect"], "aggregation": "0.57", "cross_sheet_entities": []},
        "hard"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: energy.xlsx / Sheet1]\n"
        "| Month | kWh | Cost |\n| Jan | 12000 | 1440 |\n"
        "| Feb | 11500 | 1380 |\n| Mar | 10800 | 1296 |\n\n"
        "What is the average cost per kWh across all months?",
        {"expected_answer": "0.12", "expected_values": {},
         "data_types": ["currency", "energy"], "aggregation": "0.12", "cross_sheet_entities": []},
        "hard"))

    items.append(_ex("excel_csv", "aggregation",
        "[SPREADSHEET: portfolio.csv]\n"
        "| Stock | Shares | Price | Dividend |\n"
        "| AAPL | 100 | 185.50 | 0.96 |\n"
        "| MSFT | 50 | 420.00 | 3.00 |\n"
        "| GOOG | 30 | 155.00 | 0.00 |\n\n"
        "What is the total portfolio value?",
        {"expected_answer": "44200", "expected_values": {},
         "data_types": ["currency", "shares"], "aggregation": "44200", "cross_sheet_entities": []},
        "medium"))

    # -- multi_sheet_reasoning (10) ---
    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: company.xlsx / Employees]\n"
        "| EmpID | Name | DeptID |\n| E1 | Alice | D10 |\n| E2 | Bob | D20 |\n"
        "| E3 | Carol | D10 |\n\n"
        "[SPREADSHEET: company.xlsx / Departments]\n"
        "| DeptID | DeptName | Budget |\n| D10 | Engineering | 500000 |\n"
        "| D20 | Marketing | 300000 |\n\n"
        "Which department does Bob belong to, and what is that department's budget?",
        {"expected_answer": "Marketing", "expected_values": {"Bob": "Marketing", "Marketing": "300000"},
         "data_types": ["currency"], "aggregation": "300000",
         "cross_sheet_entities": ["Bob", "D20", "Marketing"]},
        "medium"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: procurement.xlsx / Orders]\n"
        "| OrderID | VendorID | Amount |\n| PO-1 | V100 | 12000 |\n"
        "| PO-2 | V200 | 8500 |\n| PO-3 | V100 | 6700 |\n\n"
        "[SPREADSHEET: procurement.xlsx / Vendors]\n"
        "| VendorID | Name | Rating |\n| V100 | Acme Corp | 4.5 |\n"
        "| V200 | Globex Inc | 3.8 |\n\n"
        "What is the total order amount for Acme Corp?",
        {"expected_answer": "18700", "expected_values": {"Acme Corp": "18700"},
         "data_types": ["currency"], "aggregation": "18700",
         "cross_sheet_entities": ["Acme Corp", "V100"]},
        "medium"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: school.xlsx / Students]\n"
        "| StudentID | Name | ClassID |\n| S1 | Kim | C1 |\n| S2 | Lee | C2 |\n"
        "| S3 | Park | C1 |\n\n"
        "[SPREADSHEET: school.xlsx / Classes]\n"
        "| ClassID | Teacher | Room |\n| C1 | Ms. Smith | 201 |\n| C2 | Mr. Jones | 305 |\n\n"
        "Who is Kim's teacher and what room are they in?",
        {"expected_answer": "Ms. Smith", "expected_values": {"Kim": "Ms. Smith", "room": "201"},
         "data_types": [], "aggregation": "",
         "cross_sheet_entities": ["Kim", "C1", "Ms. Smith"]},
        "medium"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: retail.xlsx / Products]\n"
        "| ProdID | Name | CategoryID | Price |\n| P1 | Laptop | CAT1 | 999 |\n"
        "| P2 | Mouse | CAT2 | 25 |\n| P3 | Monitor | CAT1 | 450 |\n\n"
        "[SPREADSHEET: retail.xlsx / Categories]\n"
        "| CategoryID | CategoryName |\n| CAT1 | Electronics |\n| CAT2 | Accessories |\n\n"
        "List all products in the Electronics category with their prices.",
        {"expected_answer": "Laptop, Monitor", "expected_values": {"Laptop": "999", "Monitor": "450"},
         "data_types": ["currency"], "aggregation": "",
         "cross_sheet_entities": ["CAT1", "Electronics", "Laptop", "Monitor"]},
        "medium"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: hr.xlsx / Employees]\n"
        "| EmpID | Name | ManagerID |\n| E1 | Alice | E5 |\n| E2 | Bob | E5 |\n"
        "| E5 | Carol | null |\n\n"
        "[SPREADSHEET: hr.xlsx / Salaries]\n"
        "| EmpID | BaseSalary | Bonus |\n| E1 | 90000 | 5000 |\n"
        "| E2 | 85000 | 4000 |\n| E5 | 120000 | 15000 |\n\n"
        "What is the total compensation (base + bonus) for Carol's direct reports?",
        {"expected_answer": "184000", "expected_values": {},
         "data_types": ["currency", "bonus"], "aggregation": "184000",
         "cross_sheet_entities": ["E1", "E2", "E5", "Carol"]},
        "hard"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: logistics.xlsx / Shipments]\n"
        "| ShipID | WarehouseID | Weight_kg |\n| S1 | W1 | 450 |\n"
        "| S2 | W2 | 320 |\n| S3 | W1 | 180 |\n| S4 | W2 | 510 |\n\n"
        "[SPREADSHEET: logistics.xlsx / Warehouses]\n"
        "| WarehouseID | Location | Capacity_kg |\n| W1 | Chicago | 1000 |\n"
        "| W2 | Dallas | 800 |\n\n"
        "Which warehouse is closer to capacity based on total shipment weight?",
        {"expected_answer": "Dallas", "expected_values": {"W2": "830", "W1": "630"},
         "data_types": ["weight"], "aggregation": "830",
         "cross_sheet_entities": ["W1", "W2", "Chicago", "Dallas"]},
        "hard"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: finance.xlsx / Transactions]\n"
        "| TxnID | AccountID | Amount | Type |\n| T1 | A1 | 5000 | credit |\n"
        "| T2 | A1 | 2000 | debit |\n| T3 | A2 | 8000 | credit |\n"
        "| T4 | A2 | 3000 | debit |\n\n"
        "[SPREADSHEET: finance.xlsx / Accounts]\n"
        "| AccountID | Owner | OpeningBalance |\n| A1 | Smith | 10000 |\n"
        "| A2 | Jones | 15000 |\n\n"
        "What is the current balance for each account?",
        {"expected_answer": "Smith: 13000, Jones: 20000",
         "expected_values": {"Smith": "13000", "Jones": "20000"},
         "data_types": ["currency", "balance"], "aggregation": "",
         "cross_sheet_entities": ["A1", "A2", "Smith", "Jones"]},
        "hard"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: project.xlsx / Tasks]\n"
        "| TaskID | ProjectID | Hours |\n| T1 | P1 | 40 |\n| T2 | P1 | 25 |\n"
        "| T3 | P2 | 60 |\n| T4 | P2 | 35 |\n\n"
        "[SPREADSHEET: project.xlsx / Projects]\n"
        "| ProjectID | Name | Budget_hours |\n| P1 | Alpha | 80 |\n| P2 | Beta | 100 |\n\n"
        "Which project has used a higher percentage of its budget hours?",
        {"expected_answer": "Beta", "expected_values": {"Alpha": "81.25", "Beta": "95"},
         "data_types": ["percentage"], "aggregation": "",
         "cross_sheet_entities": ["P1", "P2", "Alpha", "Beta"]},
        "hard"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: clinic.xlsx / Appointments]\n"
        "| ApptID | DoctorID | PatientID | Date |\n| A1 | D1 | PT1 | 2025-03-01 |\n"
        "| A2 | D2 | PT2 | 2025-03-01 |\n| A3 | D1 | PT3 | 2025-03-02 |\n\n"
        "[SPREADSHEET: clinic.xlsx / Doctors]\n"
        "| DoctorID | Name | Specialty |\n| D1 | Dr. Patel | Cardiology |\n"
        "| D2 | Dr. Kim | Neurology |\n\n"
        "How many appointments does the Cardiology department have?",
        {"expected_answer": "2", "expected_values": {"Cardiology": "2"},
         "data_types": ["date"], "aggregation": "2",
         "cross_sheet_entities": ["D1", "Dr. Patel", "Cardiology"]},
        "medium"))

    items.append(_ex("excel_csv", "multi_sheet_reasoning",
        "[SPREADSHEET: fleet.xlsx / Vehicles]\n"
        "| VehicleID | DriverID | Mileage |\n| V1 | DR1 | 45000 |\n"
        "| V2 | DR2 | 62000 |\n| V3 | DR1 | 38000 |\n\n"
        "[SPREADSHEET: fleet.xlsx / Drivers]\n"
        "| DriverID | Name | License |\n| DR1 | Mike | CDL-A |\n| DR2 | Sara | CDL-B |\n\n"
        "What is the total mileage driven by Mike?",
        {"expected_answer": "83000", "expected_values": {"Mike": "83000"},
         "data_types": ["mileage"], "aggregation": "83000",
         "cross_sheet_entities": ["DR1", "Mike"]},
        "medium"))

    # -- data_type_handling (10) ---
    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: dates.csv]\n"
        "| Event | Start | End |\n| Conference | 2025-06-15 | 2025-06-18 |\n"
        "| Workshop | 2025-07-01 | 2025-07-03 |\n\n"
        "How many days is the Conference?",
        {"expected_answer": "3", "expected_values": {},
         "data_types": ["date", "duration"], "aggregation": "3", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: currencies.xlsx / Sheet1]\n"
        "| Item | USD | EUR | GBP |\n| License | 10000 | 9200 | 7900 |\n"
        "| Support | 5000 | 4600 | 3950 |\n\n"
        "What is the total cost in EUR?",
        {"expected_answer": "13800", "expected_values": {},
         "data_types": ["currency", "EUR"], "aggregation": "13800", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: percentages.csv]\n"
        "| Region | Growth_Pct | Market_Share_Pct |\n"
        "| North | 12.5 | 34.2 |\n| South | 8.3 | 28.7 |\n"
        "| East | 15.1 | 22.4 |\n| West | 6.9 | 14.7 |\n\n"
        "Which region has the highest growth percentage?",
        {"expected_answer": "East", "expected_values": {"East": "15.1"},
         "data_types": ["percentage", "growth"], "aggregation": "15.1", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: mixed_types.xlsx / Sheet1]\n"
        "| ID | Value | Unit | Status |\n| 1 | 42.5 | kg | active |\n"
        "| 2 | 1200 | ml | inactive |\n| 3 | 0.75 | m | active |\n\n"
        "List all active items with their values and units.",
        {"expected_answer": "42.5 kg, 0.75 m",
         "expected_values": {"42.5": "kg", "0.75": "m"},
         "data_types": ["unit", "status"], "aggregation": "", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: timestamps.csv]\n"
        "| Event | Timestamp |\n| Login | 2025-03-15 08:30:00 |\n"
        "| Action | 2025-03-15 09:15:00 |\n| Logout | 2025-03-15 17:45:00 |\n\n"
        "How long was the session from Login to Logout?",
        {"expected_answer": "9 hours 15 minutes",
         "expected_values": {},
         "data_types": ["timestamp", "duration"], "aggregation": "", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: booleans.csv]\n"
        "| Feature | Enabled | Premium |\n"
        "| SSO | TRUE | TRUE |\n"
        "| MFA | TRUE | FALSE |\n"
        "| Audit Log | FALSE | TRUE |\n"
        "| API Access | TRUE | TRUE |\n\n"
        "Which features are enabled but not premium?",
        {"expected_answer": "MFA", "expected_values": {"MFA": "TRUE"},
         "data_types": ["boolean"], "aggregation": "", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: geo.xlsx / Sheet1]\n"
        "| City | Lat | Lon | Population |\n"
        "| NYC | 40.71 | -74.01 | 8336817 |\n"
        "| LA | 34.05 | -118.24 | 3979576 |\n"
        "| CHI | 41.88 | -87.63 | 2693976 |\n\n"
        "Which city has the largest population?",
        {"expected_answer": "NYC", "expected_values": {"NYC": "8336817"},
         "data_types": ["coordinate", "population"], "aggregation": "8336817", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: intervals.csv]\n"
        "| Task | Start_Min | End_Min | Priority |\n"
        "| Backup | 0 | 30 | High |\n"
        "| Sync | 15 | 45 | Medium |\n"
        "| Report | 60 | 90 | Low |\n\n"
        "Which tasks overlap in their time intervals?",
        {"expected_answer": "Backup and Sync",
         "expected_values": {"Backup": "0-30", "Sync": "15-45"},
         "data_types": ["interval", "overlap"], "aggregation": "", "cross_sheet_entities": []},
        "hard"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: versions.xlsx / Sheet1]\n"
        "| Package | Version | Release_Date |\n"
        "| numpy | 1.26.4 | 2024-02-05 |\n"
        "| pandas | 2.2.1 | 2024-02-22 |\n"
        "| scipy | 1.12.0 | 2024-01-23 |\n\n"
        "Which package has the newest release?",
        {"expected_answer": "pandas", "expected_values": {"pandas": "2024-02-22"},
         "data_types": ["version", "date"], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    items.append(_ex("excel_csv", "data_type_handling",
        "[SPREADSHEET: ip_addresses.csv]\n"
        "| Server | IP | Port | Status |\n"
        "| Web1 | 10.0.1.10 | 443 | up |\n"
        "| Web2 | 10.0.1.11 | 443 | down |\n"
        "| DB1 | 10.0.2.10 | 5432 | up |\n\n"
        "Which servers are currently down?",
        {"expected_answer": "Web2", "expected_values": {"Web2": "down"},
         "data_types": ["IP", "port"], "aggregation": "", "cross_sheet_entities": []},
        "easy"))

    # -- formula_interpretation (10) ---
    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: budget.xlsx / Sheet1]\n"
        "| Category | Planned | Actual |\n| Marketing | 50000 | 48000 |\n"
        "| R&D | 80000 | 92000 |\n| Operations | 35000 | 33000 |\n\n"
        "Which categories are over budget and by how much?",
        {"expected_answer": "R&D is over budget by 12000",
         "expected_values": {"R&D": "12000"},
         "data_types": ["currency", "variance"], "aggregation": "12000", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: commission.csv]\n"
        "| Salesperson | Sales | Rate |\n| Amy | 120000 | 0.08 |\n"
        "| Ben | 95000 | 0.10 |\n| Cara | 150000 | 0.07 |\n\n"
        "Calculate each salesperson's commission (Sales * Rate).",
        {"expected_answer": "Amy: 9600, Ben: 9500, Cara: 10500",
         "expected_values": {"Amy": "9600", "Ben": "9500", "Cara": "10500"},
         "data_types": ["currency", "rate"], "aggregation": "", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: growth.xlsx / Sheet1]\n"
        "| Year | Revenue |\n| 2022 | 1000000 |\n| 2023 | 1150000 |\n"
        "| 2024 | 1350000 |\n\nWhat is the year-over-year growth rate for 2024?",
        {"expected_answer": "17.39", "expected_values": {},
         "data_types": ["percentage", "growth"], "aggregation": "17.39", "cross_sheet_entities": []},
        "hard"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: weighted.csv]\n"
        "| Component | Score | Weight |\n| Exam | 85 | 0.40 |\n"
        "| Project | 92 | 0.35 |\n| Participation | 78 | 0.25 |\n\n"
        "What is the weighted average score?",
        {"expected_answer": "85.7", "expected_values": {},
         "data_types": ["score", "weight"], "aggregation": "85.7", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: discount.xlsx / Sheet1]\n"
        "| Product | ListPrice | Discount_Pct |\n| A | 200 | 15 |\n"
        "| B | 450 | 10 |\n| C | 120 | 20 |\n\n"
        "What is the net price for each product after discount?",
        {"expected_answer": "A: 170, B: 405, C: 96",
         "expected_values": {"A": "170", "B": "405", "C": "96"},
         "data_types": ["currency", "discount", "percentage"], "aggregation": "", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: roi.csv]\n"
        "| Project | Investment | Return |\n"
        "| X | 50000 | 72000 |\n| Y | 30000 | 39000 |\n| Z | 80000 | 88000 |\n\n"
        "Calculate the ROI percentage for each project.",
        {"expected_answer": "X: 44%, Y: 30%, Z: 10%",
         "expected_values": {"X": "44", "Y": "30", "Z": "10"},
         "data_types": ["percentage", "ROI"], "aggregation": "", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: tax.xlsx / Sheet1]\n"
        "| Item | Price | TaxRate |\n| Widget | 25.00 | 0.08 |\n"
        "| Gadget | 45.00 | 0.08 |\n| Gizmo | 12.50 | 0.06 |\n\n"
        "What is the total price including tax for all items?",
        {"expected_answer": "88.25", "expected_values": {},
         "data_types": ["currency", "tax"], "aggregation": "88.25", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: compound.csv]\n"
        "| Principal | Rate | Years |\n| 10000 | 0.05 | 3 |\n\n"
        "What is the compound interest earned (A = P*(1+r)^n)?",
        {"expected_answer": "1576.25", "expected_values": {},
         "data_types": ["currency", "interest"], "aggregation": "1576.25", "cross_sheet_entities": []},
        "hard"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: margin.xlsx / Sheet1]\n"
        "| Product | Revenue | Cost |\n| A | 100 | 60 |\n"
        "| B | 200 | 150 |\n| C | 80 | 40 |\n\n"
        "Calculate the profit margin percentage for each product.",
        {"expected_answer": "A: 40%, B: 25%, C: 50%",
         "expected_values": {"A": "40", "B": "25", "C": "50"},
         "data_types": ["percentage", "margin"], "aggregation": "", "cross_sheet_entities": []},
        "medium"))

    items.append(_ex("excel_csv", "formula_interpretation",
        "[SPREADSHEET: velocity.csv]\n"
        "| Sprint | Story_Points | Days |\n"
        "| S1 | 34 | 10 |\n| S2 | 42 | 10 |\n| S3 | 38 | 10 |\n\n"
        "What is the average sprint velocity?",
        {"expected_answer": "38", "expected_values": {},
         "data_types": ["velocity"], "aggregation": "38", "cross_sheet_entities": []},
        "easy"))

    return items


# ===================================================================
# Track 2 -- Layout Intelligence  (50 examples)
# ===================================================================

def _build_layout() -> List[dict]:
    items: List[dict] = []

    # -- field_extraction (12) ---
    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "=== INVOICE ===\n"
        "Invoice Number: INV-2025-0042\n"
        "Date: March 15, 2025\n"
        "Bill To: Acme Corporation\n"
        "         123 Main Street\n"
        "         Springfield, IL 62701\n"
        "---\n"
        "| Item | Qty | Price |\n"
        "| Widget A | 10 | $25.00 |\n"
        "| Widget B | 5 | $42.00 |\n"
        "---\n"
        "Subtotal: $460.00\n"
        "Tax (8%): $36.80\n"
        "Total: $496.80\n\n"
        "Extract all key fields from this invoice.",
        {"expected_fields": ["invoice number", "date", "bill to", "subtotal", "tax", "total"],
         "expected_relationships": ["Acme Corporation"],
         "noise_tokens": [],
         "completeness_fields": ["INV-2025-0042", "March 15, 2025", "Acme Corporation",
                                 "460.00", "36.80", "496.80"]},
        "easy"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "PURCHASE ORDER #PO-9981\n"
        "Vendor: GlobalTech Solutions\n"
        "Ship To: Warehouse B, Dock 4\n"
        "Requested By: J. Martinez, Procurement\n"
        "Delivery Date: April 30, 2025\n"
        "Payment Terms: Net 45\n"
        "---\n"
        "| Part No | Description | Qty | Unit Cost |\n"
        "| GT-100 | Sensor Module | 50 | $18.50 |\n"
        "| GT-200 | Control Board | 20 | $74.00 |\n"
        "| GT-305 | Connector Cable | 100 | $3.25 |\n"
        "---\n"
        "Total: $2,730.00\n\n"
        "Extract all header fields and line items from this purchase order.",
        {"expected_fields": ["purchase order", "vendor", "ship to", "delivery date",
                            "payment terms", "total"],
         "expected_relationships": ["GlobalTech Solutions", "J. Martinez"],
         "noise_tokens": [],
         "completeness_fields": ["PO-9981", "GlobalTech Solutions", "Warehouse B",
                                 "April 30, 2025", "Net 45", "2,730.00",
                                 "GT-100", "GT-200", "GT-305"]},
        "medium"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "EMPLOYMENT CONTRACT\n"
        "Employee: Sarah J. Thompson\n"
        "Position: Senior Software Engineer\n"
        "Department: Platform Engineering\n"
        "Start Date: May 1, 2025\n"
        "Salary: $145,000 per annum\n"
        "Manager: David Chen, VP Engineering\n"
        "Location: Austin, TX (Hybrid)\n\n"
        "Extract all contract fields.",
        {"expected_fields": ["employee", "position", "department", "start date",
                            "salary", "manager", "location"],
         "expected_relationships": ["Sarah J. Thompson", "David Chen"],
         "noise_tokens": [],
         "completeness_fields": ["Sarah J. Thompson", "Senior Software Engineer",
                                 "Platform Engineering", "May 1, 2025",
                                 "145,000", "David Chen", "Austin"]},
        "easy"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "MEDICAL RECORD\n"
        "Patient: Robert M. Garcia, DOB: 1978-11-22\n"
        "MRN: 4401882\n"
        "Provider: Dr. Lisa Wong, MD\n"
        "Visit Date: 2025-03-10\n"
        "Chief Complaint: Persistent cough, 3 weeks\n"
        "Vitals: BP 128/82, HR 76, Temp 98.6F, SpO2 97%\n"
        "Assessment: Acute bronchitis\n"
        "Plan: Azithromycin 250mg x5 days, follow-up 2 weeks\n\n"
        "Extract all clinical fields from this record.",
        {"expected_fields": ["patient", "MRN", "provider", "visit date",
                            "chief complaint", "vitals", "assessment", "plan"],
         "expected_relationships": ["Robert M. Garcia", "Dr. Lisa Wong"],
         "noise_tokens": [],
         "completeness_fields": ["Robert M. Garcia", "4401882", "Dr. Lisa Wong",
                                 "2025-03-10", "cough", "128/82", "bronchitis",
                                 "Azithromycin"]},
        "medium"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "INSURANCE CLAIM FORM\n"
        "Claim #: CLM-2025-77432\n"
        "Policy #: POL-889012\n"
        "Claimant: Jennifer Wu\n"
        "Date of Loss: February 28, 2025\n"
        "Type of Loss: Water Damage\n"
        "Estimated Amount: $12,500\n"
        "Adjuster: Mark Reynolds\n"
        "Status: Under Review\n\n"
        "Extract all fields from this insurance claim.",
        {"expected_fields": ["claim", "policy", "claimant", "date of loss",
                            "type of loss", "estimated amount", "adjuster", "status"],
         "expected_relationships": ["Jennifer Wu", "Mark Reynolds"],
         "noise_tokens": [],
         "completeness_fields": ["CLM-2025-77432", "POL-889012", "Jennifer Wu",
                                 "February 28, 2025", "Water Damage", "12,500",
                                 "Mark Reynolds", "Under Review"]},
        "easy"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "VEHICLE REGISTRATION\n"
        "VIN: 1HGCM82633A004352\n"
        "Make: Honda\n"
        "Model: Accord\n"
        "Year: 2025\n"
        "Color: Silver\n"
        "Owner: Michael Torres\n"
        "Plate: ABC-1234\n"
        "Expiry: 2026-03-31\n\n"
        "Extract all registration details.",
        {"expected_fields": ["VIN", "make", "model", "year", "color", "owner", "plate", "expiry"],
         "expected_relationships": ["Michael Torres"],
         "noise_tokens": [],
         "completeness_fields": ["1HGCM82633A004352", "Honda", "Accord",
                                 "2025", "Silver", "Michael Torres", "ABC-1234"]},
        "easy"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "BANK STATEMENT\n"
        "Account Holder: Emily R. Chen\n"
        "Account Number: ****4589\n"
        "Statement Period: March 1-31, 2025\n"
        "Opening Balance: $12,450.00\n"
        "Total Credits: $8,200.00\n"
        "Total Debits: $6,780.00\n"
        "Closing Balance: $13,870.00\n\n"
        "Extract the account summary.",
        {"expected_fields": ["account holder", "account number", "statement period",
                            "opening balance", "total credits", "total debits", "closing balance"],
         "expected_relationships": ["Emily R. Chen"],
         "noise_tokens": [],
         "completeness_fields": ["Emily R. Chen", "4589", "March",
                                 "12,450.00", "8,200.00", "6,780.00", "13,870.00"]},
        "easy"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "SHIPPING MANIFEST\n"
        "Vessel: MV Pacific Star\n"
        "Voyage: PS-2025-042\n"
        "Origin: Shanghai, China\n"
        "Destination: Long Beach, CA\n"
        "ETA: April 15, 2025\n"
        "Containers: 45\n"
        "Total Weight: 892 metric tons\n"
        "Customs Broker: Pacific Trade Services\n\n"
        "Extract the shipping details.",
        {"expected_fields": ["vessel", "voyage", "origin", "destination",
                            "ETA", "containers", "total weight", "customs broker"],
         "expected_relationships": ["Pacific Star", "Pacific Trade Services"],
         "noise_tokens": [],
         "completeness_fields": ["Pacific Star", "PS-2025-042", "Shanghai",
                                 "Long Beach", "April 15, 2025", "45",
                                 "892", "Pacific Trade Services"]},
        "medium"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "POWER OF ATTORNEY\n"
        "Principal: Margaret A. Sullivan\n"
        "Agent: Thomas J. Sullivan\n"
        "Effective Date: April 1, 2025\n"
        "Scope: Financial decisions, property management\n"
        "Duration: Until revoked in writing\n"
        "Notary: Barbara Klein, Commission #NY-44521\n"
        "Notarized: March 28, 2025\n\n"
        "Extract all fields from this Power of Attorney.",
        {"expected_fields": ["principal", "agent", "effective date", "scope",
                            "duration", "notary"],
         "expected_relationships": ["Margaret A. Sullivan", "Thomas J. Sullivan", "Barbara Klein"],
         "noise_tokens": [],
         "completeness_fields": ["Margaret A. Sullivan", "Thomas J. Sullivan",
                                 "April 1, 2025", "Financial decisions",
                                 "property management", "Barbara Klein"]},
        "medium"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "BUILDING PERMIT\n"
        "Permit No: BP-2025-0331\n"
        "Property: 1200 Oak Avenue, Portland OR 97201\n"
        "Owner: Cascade Development LLC\n"
        "Contractor: Summit Builders Inc\n"
        "Work Type: Commercial Renovation\n"
        "Estimated Cost: $380,000\n"
        "Issued: March 20, 2025\n"
        "Expires: September 20, 2025\n\n"
        "Extract all permit information.",
        {"expected_fields": ["permit no", "property", "owner", "contractor",
                            "work type", "estimated cost", "issued", "expires"],
         "expected_relationships": ["Cascade Development", "Summit Builders"],
         "noise_tokens": [],
         "completeness_fields": ["BP-2025-0331", "1200 Oak Avenue", "Cascade Development",
                                 "Summit Builders", "Commercial Renovation",
                                 "380,000", "March 20, 2025"]},
        "easy"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "DEATH CERTIFICATE\n"
        "Name: Harold P. Morrison\n"
        "DOB: June 12, 1942\n"
        "Date of Death: March 8, 2025\n"
        "Place of Death: St. Mary's Hospital, Boston\n"
        "Cause: Cardiac arrest\n"
        "Certifier: Dr. Anna Petrova, MD\n"
        "Filed: March 10, 2025\n\n"
        "Extract all certificate details.",
        {"expected_fields": ["name", "DOB", "date of death", "place of death",
                            "cause", "certifier", "filed"],
         "expected_relationships": ["Harold P. Morrison", "Dr. Anna Petrova"],
         "noise_tokens": [],
         "completeness_fields": ["Harold P. Morrison", "June 12, 1942",
                                 "March 8, 2025", "St. Mary's Hospital",
                                 "Cardiac arrest", "Dr. Anna Petrova"]},
        "easy"))

    items.append(_ex("layout", "field_extraction",
        "[DOCUMENT LAYOUT]\n"
        "PATENT APPLICATION\n"
        "Title: Adaptive Neural Network Compression System\n"
        "Inventors: Dr. Wei Zhang, Dr. Maria Lopez\n"
        "Filing Date: February 14, 2025\n"
        "Application No: US-2025-0041892\n"
        "Assignee: TechVision Labs Inc\n"
        "Classification: G06N 3/08\n"
        "Abstract: A system for dynamically compressing neural network models...\n\n"
        "Extract all patent application fields.",
        {"expected_fields": ["title", "inventors", "filing date", "application no",
                            "assignee", "classification", "abstract"],
         "expected_relationships": ["Dr. Wei Zhang", "Dr. Maria Lopez", "TechVision Labs"],
         "noise_tokens": [],
         "completeness_fields": ["Adaptive Neural Network", "Dr. Wei Zhang",
                                 "Dr. Maria Lopez", "February 14, 2025",
                                 "US-2025-0041892", "TechVision Labs", "G06N 3/08"]},
        "medium"))

    # -- multi_column_layout (8) ---
    items.append(_ex("layout", "multi_column_layout",
        "[DOCUMENT LAYOUT -- 2 COLUMNS]\n"
        "LEFT COLUMN:                    RIGHT COLUMN:\n"
        "Company: Nexus Corp             Contact: Lisa Park\n"
        "Address: 500 Tech Blvd          Phone: 555-0200\n"
        "         San Jose, CA           Email: lisa@nexus.com\n"
        "Founded: 2018                   Role: CTO\n\n"
        "Parse both columns and identify the company contact information.",
        {"expected_fields": ["company", "address", "contact", "phone", "email", "role"],
         "expected_relationships": ["Nexus Corp", "Lisa Park"],
         "noise_tokens": [],
         "completeness_fields": ["Nexus Corp", "500 Tech Blvd", "Lisa Park",
                                 "555-0200", "lisa@nexus.com", "CTO"]},
        "medium"))

    items.append(_ex("layout", "multi_column_layout",
        "[DOCUMENT LAYOUT -- SIDE BY SIDE]\n"
        "--- SELLER ---         --- BUYER ---\n"
        "Name: TechPro          Name: MedCo\n"
        "EIN: 12-345            EIN: 67-890\n"
        "State: CA              State: NY\n\n"
        "Extract both the seller and buyer information.",
        {"expected_fields": ["seller", "buyer", "name", "EIN", "state"],
         "expected_relationships": ["TechPro", "MedCo"],
         "noise_tokens": [],
         "completeness_fields": ["TechPro", "12-345", "CA", "MedCo", "67-890", "NY"]},
        "medium"))

    items.append(_ex("layout", "multi_column_layout",
        "[DOCUMENT LAYOUT -- 3 COLUMNS]\n"
        "PLAN A              PLAN B              PLAN C\n"
        "Premium: $200/mo    Premium: $350/mo    Premium: $500/mo\n"
        "Deductible: $5000   Deductible: $2500   Deductible: $1000\n"
        "Copay: $40          Copay: $25          Copay: $15\n"
        "Network: Basic      Network: Standard   Network: Premium\n\n"
        "Compare the three insurance plans.",
        {"expected_fields": ["premium", "deductible", "copay", "network"],
         "expected_relationships": ["Plan A", "Plan B", "Plan C"],
         "noise_tokens": [],
         "completeness_fields": ["200", "350", "500", "5000", "2500", "1000",
                                 "40", "25", "15", "Basic", "Standard", "Premium"]},
        "medium"))

    items.append(_ex("layout", "multi_column_layout",
        "[DOCUMENT LAYOUT -- COMPARISON TABLE]\n"
        "Feature          | Option 1    | Option 2    | Option 3\n"
        "Price            | $99/yr      | $199/yr     | $499/yr\n"
        "Users            | 5           | 25          | Unlimited\n"
        "Storage          | 10 GB       | 100 GB      | 1 TB\n"
        "Support          | Email       | Phone+Email | 24/7 Priority\n"
        "API Access       | No          | Yes         | Yes\n\n"
        "Extract all features for each option.",
        {"expected_fields": ["price", "users", "storage", "support", "API access"],
         "expected_relationships": ["Option 1", "Option 2", "Option 3"],
         "noise_tokens": [],
         "completeness_fields": ["99", "199", "499", "5", "25", "Unlimited",
                                 "10 GB", "100 GB", "1 TB"]},
        "easy"))

    items.append(_ex("layout", "multi_column_layout",
        "[DOCUMENT LAYOUT -- PARALLEL SECTIONS]\n"
        "=== CURRENT POLICY ===          === PROPOSED CHANGES ===\n"
        "Max PTO: 15 days                Max PTO: 20 days\n"
        "Remote Work: 2 days/week        Remote Work: 3 days/week\n"
        "Health Stipend: $500/yr         Health Stipend: $1000/yr\n"
        "401k Match: 3%                  401k Match: 5%\n\n"
        "What are the proposed changes compared to the current policy?",
        {"expected_fields": ["PTO", "remote work", "health stipend", "401k"],
         "expected_relationships": ["current", "proposed"],
         "noise_tokens": [],
         "completeness_fields": ["15 days", "20 days", "2 days", "3 days",
                                 "500", "1000", "3%", "5%"]},
        "medium"))

    items.append(_ex("layout", "multi_column_layout",
        "[DOCUMENT LAYOUT -- BILINGUAL]\n"
        "ENGLISH                          SPANISH\n"
        "Name: Universal Health Plan      Nombre: Plan de Salud Universal\n"
        "Coverage: Full medical           Cobertura: Medica completa\n"
        "Premium: $450/month              Prima: $450/mes\n"
        "Effective: January 1, 2025       Vigente: 1 de enero de 2025\n\n"
        "Extract the plan details from both language columns.",
        {"expected_fields": ["name", "coverage", "premium", "effective"],
         "expected_relationships": ["Universal Health Plan"],
         "noise_tokens": [],
         "completeness_fields": ["Universal Health Plan", "Full medical",
                                 "450", "January 1, 2025"]},
        "medium"))

    items.append(_ex("layout", "multi_column_layout",
        "[DOCUMENT LAYOUT -- BEFORE/AFTER]\n"
        "BEFORE RENOVATION               AFTER RENOVATION\n"
        "Sq Ft: 2,400                    Sq Ft: 3,100\n"
        "Bedrooms: 3                     Bedrooms: 4\n"
        "Bathrooms: 1.5                  Bathrooms: 2.5\n"
        "Garage: 1-car                   Garage: 2-car\n"
        "Estimated Value: $350K          Estimated Value: $485K\n\n"
        "What changes were made during renovation?",
        {"expected_fields": ["sq ft", "bedrooms", "bathrooms", "garage", "estimated value"],
         "expected_relationships": ["before", "after"],
         "noise_tokens": [],
         "completeness_fields": ["2,400", "3,100", "3", "4", "1.5", "2.5",
                                 "350", "485"]},
        "medium"))

    items.append(_ex("layout", "multi_column_layout",
        "[DOCUMENT LAYOUT -- DUAL PARTY]\n"
        "PLAINTIFF                        DEFENDANT\n"
        "Name: Jackson LLC                Name: Rivera Corp\n"
        "Attorney: M. Williams            Attorney: K. Tanaka\n"
        "Case: 2025-CV-11023              Filed: March 5, 2025\n"
        "Claim: Breach of contract        Amount: $2.4M\n\n"
        "Extract both party details and the case information.",
        {"expected_fields": ["plaintiff", "defendant", "attorney", "case",
                            "claim", "amount"],
         "expected_relationships": ["Jackson LLC", "Rivera Corp",
                                   "M. Williams", "K. Tanaka"],
         "noise_tokens": [],
         "completeness_fields": ["Jackson LLC", "Rivera Corp", "M. Williams",
                                 "K. Tanaka", "2025-CV-11023", "2.4M"]},
        "medium"))

    # -- nested_structure (10) ---
    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "1. PARTIES\n"
        "   1.1 Licensor: DataCorp Inc.\n"
        "   1.2 Licensee: SmallBiz LLC\n"
        "2. TERMS\n"
        "   2.1 Duration: 24 months\n"
        "   2.2 Renewal: Automatic unless 90-day notice given\n"
        "   2.3 Fees\n"
        "       2.3.1 Base License: $50,000/year\n"
        "       2.3.2 Per-Seat Fee: $200/user/month\n"
        "       2.3.3 Overage: $0.05 per API call above 1M/month\n"
        "3. TERMINATION\n"
        "   3.1 For Cause: 30-day cure period\n"
        "   3.2 Convenience: 60-day written notice\n\n"
        "Extract the hierarchical structure of this agreement.",
        {"expected_fields": ["parties", "licensor", "licensee", "duration",
                            "renewal", "fees", "base license", "per-seat",
                            "termination"],
         "expected_relationships": ["DataCorp Inc.", "SmallBiz LLC"],
         "noise_tokens": [],
         "completeness_fields": ["DataCorp Inc.", "SmallBiz LLC", "24 months",
                                 "90-day", "50,000", "200", "0.05",
                                 "30-day", "60-day"]},
        "hard"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "COMPLIANCE AUDIT REPORT\n"
        "Section A: Data Protection\n"
        "  A.1 Encryption at rest: PASS\n"
        "  A.2 Encryption in transit: PASS\n"
        "  A.3 Access controls\n"
        "      A.3.1 MFA enabled: PASS\n"
        "      A.3.2 Role-based access: FAIL -- 3 users with excess privileges\n"
        "      A.3.3 Audit logging: PASS\n"
        "Section B: Business Continuity\n"
        "  B.1 Backup frequency: PASS -- daily automated\n"
        "  B.2 Recovery time objective: FAIL -- tested at 8 hrs, target 4 hrs\n"
        "  B.3 Disaster recovery plan: PASS\n\n"
        "Summarize all FAIL findings with their details.",
        {"expected_fields": ["role-based access", "recovery time objective"],
         "expected_relationships": ["FAIL"],
         "noise_tokens": [],
         "completeness_fields": ["A.3.2", "excess privileges", "B.2",
                                 "8 hrs", "4 hrs"]},
        "medium"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "PRODUCT SPECIFICATION\n"
        "1. Overview\n"
        "   Product: SmartSensor X200\n"
        "   Version: 3.1\n"
        "2. Dimensions\n"
        "   2.1 Height: 45mm\n"
        "   2.2 Width: 30mm\n"
        "   2.3 Depth: 12mm\n"
        "   2.4 Weight: 28g\n"
        "3. Electrical\n"
        "   3.1 Voltage: 3.3V DC\n"
        "   3.2 Current: 15mA active, 0.5uA sleep\n"
        "   3.3 Interface: I2C, SPI\n"
        "4. Environmental\n"
        "   4.1 Operating Temp: -20C to 85C\n"
        "   4.2 IP Rating: IP67\n\n"
        "Extract all specifications organized by section.",
        {"expected_fields": ["product", "version", "height", "width", "depth",
                            "weight", "voltage", "current", "interface", "operating temp", "IP rating"],
         "expected_relationships": ["SmartSensor X200"],
         "noise_tokens": [],
         "completeness_fields": ["SmartSensor X200", "3.1", "45mm", "30mm",
                                 "12mm", "28g", "3.3V", "15mA", "I2C", "SPI",
                                 "-20C", "85C", "IP67"]},
        "medium"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "MEETING MINUTES -- Board Meeting 2025-03-20\n"
        "Attendees: J. Smith (Chair), L. Kim, R. Patel, M. Torres\n"
        "1. Call to Order -- 10:00 AM\n"
        "2. Approval of Previous Minutes -- Unanimously approved\n"
        "3. Financial Report\n"
        "   3.1 Q1 Revenue: $4.2M (up 12% YoY)\n"
        "   3.2 Q1 Expenses: $3.8M\n"
        "   3.3 Net Income: $400K\n"
        "   3.4 Action Item: CFO to present cost reduction plan by April 15\n"
        "4. New Business\n"
        "   4.1 Proposal to open Denver office -- approved 3-1\n"
        "   4.2 Hiring freeze lifted for Engineering\n"
        "5. Adjournment -- 11:45 AM\n\n"
        "Extract all action items and decisions from these minutes.",
        {"expected_fields": ["action item", "approved", "hiring freeze"],
         "expected_relationships": ["CFO", "Denver", "Engineering"],
         "noise_tokens": [],
         "completeness_fields": ["cost reduction plan", "April 15",
                                 "Denver office", "approved 3-1",
                                 "hiring freeze lifted"]},
        "medium"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "ORGANIZATIONAL CHART\n"
        "CEO: Maria Santos\n"
        " -- CTO: James Liu\n"
        "      -- VP Engineering: Raj Patel\n"
        "           -- Sr. Engineer: Alex Kim\n"
        "           -- Sr. Engineer: Pat Chen\n"
        "      -- VP Product: Dana Lee\n"
        " -- CFO: Tom Brown\n"
        "      -- Controller: Sue Park\n"
        " -- COO: Nina White\n"
        "      -- VP Operations: Mike Green\n\n"
        "Who are the direct reports of the CTO?",
        {"expected_fields": ["CTO", "direct reports"],
         "expected_relationships": ["James Liu", "Raj Patel", "Dana Lee"],
         "noise_tokens": [],
         "completeness_fields": ["James Liu", "Raj Patel", "Dana Lee"]},
        "easy"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "RISK REGISTER\n"
        "R1: Cybersecurity Breach\n"
        "   Likelihood: High\n"
        "   Impact: Critical\n"
        "   Mitigation:\n"
        "     M1.1: Deploy SIEM system\n"
        "     M1.2: Quarterly pen tests\n"
        "     M1.3: Security awareness training\n"
        "R2: Key Person Dependency\n"
        "   Likelihood: Medium\n"
        "   Impact: High\n"
        "   Mitigation:\n"
        "     M2.1: Cross-training program\n"
        "     M2.2: Documentation initiative\n\n"
        "Extract all risks and their mitigations.",
        {"expected_fields": ["cybersecurity", "key person", "likelihood", "impact", "mitigation"],
         "expected_relationships": [],
         "noise_tokens": [],
         "completeness_fields": ["Cybersecurity Breach", "High", "Critical",
                                 "SIEM", "pen tests", "Key Person", "Cross-training"]},
        "medium"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "API DOCUMENTATION\n"
        "Endpoint: POST /api/v2/users\n"
        "  Description: Create a new user\n"
        "  Headers:\n"
        "    Authorization: Bearer {token}\n"
        "    Content-Type: application/json\n"
        "  Request Body:\n"
        "    name: string (required)\n"
        "    email: string (required)\n"
        "    role: string (optional, default: viewer)\n"
        "  Responses:\n"
        "    201: User created successfully\n"
        "    400: Validation error\n"
        "    401: Unauthorized\n"
        "    409: Email already exists\n\n"
        "Extract the API endpoint specification.",
        {"expected_fields": ["endpoint", "description", "headers", "request body", "responses"],
         "expected_relationships": [],
         "noise_tokens": [],
         "completeness_fields": ["POST", "/api/v2/users", "Authorization",
                                 "name", "email", "role", "201", "400", "401", "409"]},
        "medium"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "RECIPE: Classic Beef Stew\n"
        "Prep Time: 20 min | Cook Time: 2 hrs | Serves: 6\n"
        "Ingredients:\n"
        "  Meat:\n"
        "    2 lbs beef chuck, cubed\n"
        "    2 tbsp flour\n"
        "  Vegetables:\n"
        "    3 carrots, sliced\n"
        "    2 potatoes, cubed\n"
        "    1 onion, diced\n"
        "  Liquids:\n"
        "    2 cups beef broth\n"
        "    1 cup red wine\n"
        "Instructions:\n"
        "  1. Season and flour beef\n"
        "  2. Brown in Dutch oven\n"
        "  3. Add vegetables and liquids\n"
        "  4. Simmer 2 hours\n\n"
        "Extract the recipe structure.",
        {"expected_fields": ["prep time", "cook time", "serves", "ingredients", "instructions"],
         "expected_relationships": [],
         "noise_tokens": [],
         "completeness_fields": ["20 min", "2 hrs", "6", "beef chuck",
                                 "carrots", "potatoes", "beef broth",
                                 "red wine", "Brown", "Simmer"]},
        "easy"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "COURSE SYLLABUS -- CS 301 Data Structures\n"
        "Instructor: Dr. Amanda Foster\n"
        "Schedule: MWF 10:00-10:50\n"
        "Grading:\n"
        "  Homework: 20%\n"
        "  Midterm: 25%\n"
        "  Final: 30%\n"
        "  Projects: 25%\n"
        "Topics:\n"
        "  Week 1-3: Arrays, Linked Lists\n"
        "  Week 4-6: Trees, BST\n"
        "  Week 7-9: Graphs, Shortest Path\n"
        "  Week 10-12: Hash Tables, Advanced Topics\n\n"
        "Extract the syllabus structure.",
        {"expected_fields": ["instructor", "schedule", "grading", "topics"],
         "expected_relationships": ["Dr. Amanda Foster"],
         "noise_tokens": [],
         "completeness_fields": ["Dr. Amanda Foster", "MWF", "Homework", "20%",
                                 "Midterm", "25%", "Final", "30%",
                                 "Arrays", "Trees", "Graphs", "Hash Tables"]},
        "easy"))

    items.append(_ex("layout", "nested_structure",
        "[DOCUMENT LAYOUT]\n"
        "WARRANTY TERMS\n"
        "1. Coverage Period\n"
        "   1.1 Hardware: 36 months from purchase\n"
        "   1.2 Software: 12 months, updates included\n"
        "   1.3 Battery: 24 months, prorated after 12\n"
        "2. Exclusions\n"
        "   2.1 Physical damage from misuse\n"
        "   2.2 Water damage (non-waterproof models)\n"
        "   2.3 Unauthorized modifications\n"
        "3. Claim Process\n"
        "   3.1 Contact support with proof of purchase\n"
        "   3.2 Ship product to service center (prepaid label provided)\n"
        "   3.3 Turnaround: 5-10 business days\n\n"
        "Extract the warranty terms hierarchy.",
        {"expected_fields": ["coverage period", "hardware", "software", "battery",
                            "exclusions", "claim process"],
         "expected_relationships": [],
         "noise_tokens": [],
         "completeness_fields": ["36 months", "12 months", "24 months",
                                 "Physical damage", "Water damage",
                                 "proof of purchase", "5-10 business days"]},
        "medium"))

    # -- noisy_document (10) ---
    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- SCANNED WITH ARTIFACTS]\n"
        "~~~ HEADER ARTIFACT ~~~\n"
        "LEASE AGREEMENT\n"
        "[watermark: DRAFT COPY]\n"
        "Landlord: Riverside Properties LLC\n"
        "Tenant: John & Mary Davis\n"
        "[smudge] Monthly Rent: $2,800\n"
        "Security Deposit: $5,600\n"
        "Lease Term: 12 months\n"
        "Start Date: June 1, 2025\n"
        "[page fold artifact]\n"
        "Pet Policy: No pets allowed\n"
        "[footer: Page 1 of 8 -- CONFIDENTIAL]\n\n"
        "Extract the lease terms, ignoring scan artifacts.",
        {"expected_fields": ["landlord", "tenant", "monthly rent",
                            "security deposit", "lease term", "start date", "pet policy"],
         "expected_relationships": ["Riverside Properties LLC", "Davis"],
         "noise_tokens": ["watermark", "smudge", "page fold artifact", "HEADER ARTIFACT"],
         "completeness_fields": ["Riverside Properties LLC", "Davis",
                                 "2,800", "5,600", "12 months",
                                 "June 1, 2025", "No pets"]},
        "medium"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- OCR OUTPUT WITH ERRORS]\n"
        "PAYR0LL SUMMARY -- March 2O25\n"
        "[header noise: !!!@@@###]\n"
        "Emp1oyee: Sarah Johnson\n"
        "Gross Pay: $8,333.33\n"
        "Federal Tax: $1,666.67\n"
        "State Tax: $416.67\n"
        "401(k): $500.00\n"
        "Health Ins: $250.00\n"
        "Net Pay: $5,499.99\n"
        "[garbled: x#@k!zz]\n\n"
        "Extract the payroll details despite OCR errors.",
        {"expected_fields": ["employee", "gross pay", "federal tax", "state tax",
                            "401(k)", "health", "net pay"],
         "expected_relationships": ["Sarah Johnson"],
         "noise_tokens": ["!!!@@@###", "x#@k!zz", "PAYR0LL", "2O25", "Emp1oyee"],
         "completeness_fields": ["Sarah Johnson", "8,333.33", "1,666.67",
                                 "416.67", "500.00", "250.00", "5,499.99"]},
        "hard"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- MULTI-PAGE SCAN]\n"
        "--- PAGE 1 HEADER: ACME CORP INTERNAL ---\n"
        "[stamp: RECEIVED MAR 15 2025]\n"
        "VENDOR AGREEMENT\n"
        "Between: Acme Corp (\"Client\")\n"
        "And: FastShip Logistics (\"Vendor\")\n"
        "Effective Date: April 1, 2025\n"
        "[coffee stain obscuring text]\n"
        "--- PAGE 2 ---\n"
        "Service Level: 99.5% uptime\n"
        "Response Time: 4 hours\n"
        "Penalty: 2% credit per SLA breach\n"
        "[torn corner -- text missing]\n"
        "Term: 36 months\n\n"
        "Extract the agreement details, noting any missing information.",
        {"expected_fields": ["client", "vendor", "effective date", "service level",
                            "response time", "penalty", "term"],
         "expected_relationships": ["Acme Corp", "FastShip Logistics"],
         "noise_tokens": ["coffee stain", "torn corner", "stamp", "RECEIVED"],
         "completeness_fields": ["Acme Corp", "FastShip Logistics",
                                 "April 1, 2025", "99.5%", "4 hours",
                                 "2%", "36 months"]},
        "hard"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- DEGRADED FAX]\n"
        "[static noise: ################]\n"
        "PRESCRIPTION\n"
        "Patient: [partially illegible] ...riguez\n"
        "DOB: 04/12/1990\n"
        "Rx: Lisinopril 10mg\n"
        "Sig: Take 1 tablet daily\n"
        "Qty: 30\n"
        "Refills: 3\n"
        "Prescriber: Dr. A. Nakamura\n"
        "[fax footer: TX CONFIRMED 03/15/2025 14:22]\n\n"
        "Extract all readable prescription fields.",
        {"expected_fields": ["patient", "DOB", "Rx", "sig", "qty", "refills", "prescriber"],
         "expected_relationships": ["Dr. A. Nakamura"],
         "noise_tokens": ["static noise", "####", "fax footer", "TX CONFIRMED"],
         "completeness_fields": ["riguez", "04/12/1990", "Lisinopril",
                                 "10mg", "1 tablet daily", "30", "3",
                                 "Dr. A. Nakamura"]},
        "hard"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- SCANNED RECEIPT]\n"
        "[thermal fade: ........]\n"
        "STORE: QuickMart #442\n"
        "DATE: 03/14/2025  TIME: 16:45\n"
        "---\n"
        "Milk 2%         $4.29\n"
        "Bread Wheat     $3.49\n"
        "[smudged line]\n"
        "Eggs Large      $5.99\n"
        "---\n"
        "SUBTOTAL        $13.77\n"
        "TAX             $0.96\n"
        "TOTAL           $14.73\n"
        "PAID: VISA ****4521\n"
        "[crumpled paper artifact]\n\n"
        "Extract all items and the payment information.",
        {"expected_fields": ["store", "date", "subtotal", "tax", "total", "paid"],
         "expected_relationships": ["QuickMart"],
         "noise_tokens": ["thermal fade", "smudged line", "crumpled paper artifact"],
         "completeness_fields": ["QuickMart", "03/14/2025", "Milk",
                                 "Bread", "Eggs", "13.77", "0.96",
                                 "14.73", "4521"]},
        "medium"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- POOR SCAN QUALITY]\n"
        "[blur artifact across top]\n"
        "NOTICE OF EVICTION\n"
        "To: [partially illegible]...son, Apt 4B\n"
        "Date: March 1, 2025\n"
        "Reason: Non-payment of rent (2 months overdue)\n"
        "Amount Due: $7,000\n"
        "Vacate By: April 1, 2025\n"
        "[ink bleed on signature line]\n"
        "Issued by: Property Management Office\n\n"
        "Extract all readable notice details.",
        {"expected_fields": ["tenant", "date", "reason", "amount due", "vacate by", "issued by"],
         "expected_relationships": ["Property Management Office"],
         "noise_tokens": ["blur artifact", "ink bleed", "partially illegible"],
         "completeness_fields": ["Apt 4B", "March 1, 2025", "Non-payment",
                                 "7,000", "April 1, 2025", "Property Management"]},
        "hard"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- WATER DAMAGED]\n"
        "CERTIFICATE OF ANALYSIS\n"
        "Batch: B-2025-0442\n"
        "Product: [water damage] ...itamin D3\n"
        "Test Date: March 12, 2025\n"
        "[water stain covering table header]\n"
        "Potency: 1000 IU [PASS]\n"
        "Heavy Metals: <0.1 ppm [PASS]\n"
        "Microbial: Negative [PASS]\n"
        "[page wrinkle]\n"
        "Released By: QC Lab Manager\n\n"
        "Extract the lab results despite water damage.",
        {"expected_fields": ["batch", "product", "test date", "potency",
                            "heavy metals", "microbial", "released by"],
         "expected_relationships": ["QC Lab Manager"],
         "noise_tokens": ["water damage", "water stain", "page wrinkle"],
         "completeness_fields": ["B-2025-0442", "itamin D3", "March 12, 2025",
                                 "1000 IU", "PASS", "0.1 ppm", "Negative"]},
        "hard"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- PHOTOCOPIED WITH NOISE]\n"
        "[black border from copier]\n"
        "[skew angle: 3 degrees]\n"
        "WORK ORDER #WO-5521\n"
        "Requested By: Facilities Dept\n"
        "Assigned To: ABC Electrical\n"
        "Priority: Urgent\n"
        "Description: Replace panel in Server Room B\n"
        "Scheduled: March 18, 2025 at 6:00 AM\n"
        "[copier streak across bottom]\n"
        "Estimated Cost: $4,200\n"
        "Approval: Director of Operations\n\n"
        "Extract the work order details.",
        {"expected_fields": ["work order", "requested by", "assigned to",
                            "priority", "description", "scheduled", "estimated cost"],
         "expected_relationships": ["ABC Electrical", "Facilities Dept"],
         "noise_tokens": ["black border", "skew angle", "copier streak"],
         "completeness_fields": ["WO-5521", "Facilities", "ABC Electrical",
                                 "Urgent", "Server Room B", "March 18, 2025",
                                 "4,200"]},
        "medium"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- OVEREXPOSED SCAN]\n"
        "[washed out header area]\n"
        "TRAVEL AUTHORIZATION\n"
        "Employee: K. Yamamoto\n"
        "Destination: London, UK\n"
        "Dates: April 5-9, 2025\n"
        "Purpose: Client conference\n"
        "[overexposed: flight details partially visible]\n"
        "Flight: ...A 178 JFK-LHR\n"
        "Hotel: Marriott Canary Wharf, 4 nights\n"
        "Est. Budget: $4,800\n"
        "Approved By: VP Sales\n"
        "[stamp: APPROVED]\n\n"
        "Extract the travel details.",
        {"expected_fields": ["employee", "destination", "dates", "purpose",
                            "flight", "hotel", "budget", "approved by"],
         "expected_relationships": ["K. Yamamoto"],
         "noise_tokens": ["washed out", "overexposed", "partially visible"],
         "completeness_fields": ["K. Yamamoto", "London", "April 5-9, 2025",
                                 "Client conference", "JFK-LHR",
                                 "Marriott", "4,800", "VP Sales"]},
        "medium"))

    items.append(_ex("layout", "noisy_document",
        "[DOCUMENT LAYOUT -- REDACTED DOCUMENT]\n"
        "SECURITY CLEARANCE APPLICATION\n"
        "Applicant: [REDACTED]\n"
        "DOB: [REDACTED]\n"
        "Citizenship: US\n"
        "Clearance Level: Secret\n"
        "Sponsor: Department of Defense\n"
        "Investigation: NACLC\n"
        "Status: Pending adjudication\n"
        "[REDACTED SECTION: Employment history]\n"
        "Polygraph: Not required\n\n"
        "Extract all non-redacted fields.",
        {"expected_fields": ["citizenship", "clearance level", "sponsor",
                            "investigation", "status", "polygraph"],
         "expected_relationships": ["Department of Defense"],
         "noise_tokens": ["REDACTED"],
         "completeness_fields": ["US", "Secret", "Department of Defense",
                                 "NACLC", "Pending adjudication", "Not required"]},
        "medium"))

    # -- header_footer (5) ---
    items.append(_ex("layout", "header_footer",
        "[DOCUMENT LAYOUT]\n"
        "[HEADER: ACME CORP | Confidential | Rev 2.1]\n\n"
        "STATEMENT OF WORK\n"
        "Project: Cloud Migration Phase 2\n"
        "Client: Acme Corp\n"
        "Vendor: CloudFirst Inc.\n"
        "Duration: 6 months\n"
        "Budget: $450,000\n\n"
        "[FOOTER: Page 1 of 12 | Last Modified: 2025-03-10 | Author: K. Yamamoto]\n\n"
        "Extract the SOW details and document metadata.",
        {"expected_fields": ["project", "client", "vendor", "duration", "budget",
                            "revision", "author", "last modified"],
         "expected_relationships": ["Acme Corp", "CloudFirst Inc.", "K. Yamamoto"],
         "noise_tokens": [],
         "completeness_fields": ["Cloud Migration Phase 2", "Acme Corp",
                                 "CloudFirst Inc.", "6 months", "450,000",
                                 "Rev 2.1", "K. Yamamoto", "2025-03-10"]},
        "medium"))

    items.append(_ex("layout", "header_footer",
        "[DOCUMENT LAYOUT]\n"
        "[LETTERHEAD: Morrison & Partners LLP -- Attorneys at Law]\n"
        "[ADDRESS: 100 Legal Ave, Suite 800, New York, NY 10001]\n\n"
        "March 18, 2025\n\n"
        "Re: Settlement Agreement -- Case No. 2024-CV-5591\n\n"
        "Dear Ms. Chen,\n\n"
        "This letter confirms the terms of settlement reached on March 12, 2025. "
        "The defendant agrees to pay $75,000 within 30 days.\n\n"
        "Sincerely,\n"
        "Robert Morrison, Esq.\n\n"
        "[FOOTER: Morrison & Partners LLP | Tel: 212-555-0300]\n\n"
        "Extract the letter metadata and key settlement terms.",
        {"expected_fields": ["firm", "case number", "date", "settlement amount",
                            "payment terms", "sender"],
         "expected_relationships": ["Morrison & Partners", "Ms. Chen", "Robert Morrison"],
         "noise_tokens": [],
         "completeness_fields": ["Morrison & Partners", "2024-CV-5591",
                                 "March 18, 2025", "75,000", "30 days",
                                 "Robert Morrison"]},
        "medium"))

    items.append(_ex("layout", "header_footer",
        "[DOCUMENT LAYOUT]\n"
        "[HEADER: RESTRICTED -- FOR INTERNAL USE ONLY]\n"
        "[LOGO: TechVentures Capital]\n\n"
        "INVESTMENT MEMO\n"
        "Company: NeuralFlow AI\n"
        "Round: Series B\n"
        "Ask: $25M at $200M pre-money valuation\n"
        "Lead: TechVentures Capital\n"
        "Recommendation: PROCEED\n\n"
        "[FOOTER: Prepared by M. Goldstein | Q1 2025 | Page 1/5]\n\n"
        "Extract the investment details and document metadata.",
        {"expected_fields": ["company", "round", "ask", "valuation",
                            "lead", "recommendation", "prepared by"],
         "expected_relationships": ["NeuralFlow AI", "TechVentures Capital", "M. Goldstein"],
         "noise_tokens": [],
         "completeness_fields": ["NeuralFlow AI", "Series B", "25M",
                                 "200M", "TechVentures Capital", "PROCEED",
                                 "M. Goldstein"]},
        "easy"))

    items.append(_ex("layout", "header_footer",
        "[DOCUMENT LAYOUT]\n"
        "[HEADER: STATE OF CALIFORNIA | DEPARTMENT OF LABOR]\n\n"
        "NOTICE TO EMPLOYEE\n"
        "Employer: Pacific Tech Corp\n"
        "Employee: Daniel Reyes\n"
        "Effective Date: April 1, 2025\n"
        "Subject: Change in Employment Terms\n"
        "New Salary: $92,000/year (from $85,000)\n"
        "New Title: Senior Analyst\n\n"
        "[FOOTER: Form DL-2025 | Rev 01/2025 | Page 1 of 1]\n\n"
        "Extract the employment change details.",
        {"expected_fields": ["employer", "employee", "effective date", "subject",
                            "new salary", "new title"],
         "expected_relationships": ["Pacific Tech Corp", "Daniel Reyes"],
         "noise_tokens": [],
         "completeness_fields": ["Pacific Tech Corp", "Daniel Reyes",
                                 "April 1, 2025", "92,000", "85,000",
                                 "Senior Analyst"]},
        "easy"))

    items.append(_ex("layout", "header_footer",
        "[DOCUMENT LAYOUT]\n"
        "[HEADER: QUARTERLY BOARD REPORT | Q1 2025 | CONFIDENTIAL]\n\n"
        "Executive Summary\n"
        "Revenue: $14.2M (Target: $13.5M)\n"
        "EBITDA: $3.1M (Margin: 21.8%)\n"
        "Headcount: 312 (up 18 from Q4)\n"
        "Key Win: $2.5M enterprise contract with MegaBank\n"
        "Key Risk: Supply chain delays (Est. $800K impact)\n\n"
        "[FOOTER: Prepared by CFO Office | Distribution: Board Members Only]\n\n"
        "Extract the executive summary and metadata.",
        {"expected_fields": ["revenue", "EBITDA", "headcount", "key win", "key risk"],
         "expected_relationships": ["MegaBank", "CFO Office"],
         "noise_tokens": [],
         "completeness_fields": ["14.2M", "13.5M", "3.1M", "21.8%",
                                 "312", "2.5M", "MegaBank", "800K"]},
        "easy"))

    # -- table_in_layout (5) ---
    items.append(_ex("layout", "table_in_layout",
        "[DOCUMENT LAYOUT]\n"
        "QUARTERLY FINANCIAL SUMMARY\n"
        "Company: Zenith Technologies | Period: Q1 2025\n\n"
        "| Metric | Q1 2025 | Q4 2024 | Change |\n"
        "| Revenue | $12.4M | $11.8M | +5.1% |\n"
        "| COGS | $7.2M | $7.0M | +2.9% |\n"
        "| Gross Margin | 41.9% | 40.7% | +1.2pp |\n"
        "| Net Income | $1.8M | $1.5M | +20.0% |\n\n"
        "What are the key financial changes from Q4 2024 to Q1 2025?",
        {"expected_fields": ["revenue", "COGS", "gross margin", "net income"],
         "expected_relationships": ["Zenith Technologies"],
         "noise_tokens": [],
         "completeness_fields": ["12.4M", "11.8M", "5.1%", "7.2M",
                                 "41.9%", "40.7%", "1.8M", "1.5M", "20.0%"]},
        "medium"))

    items.append(_ex("layout", "table_in_layout",
        "[DOCUMENT LAYOUT]\n"
        "SERVICE LEVEL REPORT -- February 2025\n"
        "Client: MegaBank\n\n"
        "| Service | Target | Actual | Status |\n"
        "| Uptime | 99.9% | 99.95% | PASS |\n"
        "| Response Time | <200ms | 145ms | PASS |\n"
        "| Error Rate | <0.1% | 0.08% | PASS |\n"
        "| Backup Success | 100% | 98% | FAIL |\n\n"
        "Which SLAs were met and which failed?",
        {"expected_fields": ["uptime", "response time", "error rate", "backup"],
         "expected_relationships": ["MegaBank"],
         "noise_tokens": [],
         "completeness_fields": ["99.95%", "145ms", "0.08%", "98%", "FAIL", "Backup"]},
        "easy"))

    items.append(_ex("layout", "table_in_layout",
        "[DOCUMENT LAYOUT]\n"
        "EMPLOYEE BENEFITS SUMMARY\n\n"
        "| Benefit | Coverage | Employee Cost | Employer Cost |\n"
        "| Medical | Family | $250/mo | $800/mo |\n"
        "| Dental | Individual | $25/mo | $50/mo |\n"
        "| Vision | Family | $15/mo | $30/mo |\n"
        "| Life Insurance | 2x salary | $0 | $45/mo |\n"
        "| 401(k) | Up to 6% match | Varies | Up to 6% |\n\n"
        "Extract all benefits and their cost breakdown.",
        {"expected_fields": ["medical", "dental", "vision", "life insurance", "401(k)"],
         "expected_relationships": [],
         "noise_tokens": [],
         "completeness_fields": ["250", "800", "25", "50", "15", "30",
                                 "2x salary", "6%"]},
        "easy"))

    items.append(_ex("layout", "table_in_layout",
        "[DOCUMENT LAYOUT]\n"
        "VENDOR SCORECARD -- Q1 2025\n\n"
        "| Vendor | Quality | Delivery | Price | Overall |\n"
        "| Acme Corp | 4.5 | 4.0 | 3.5 | 4.0 |\n"
        "| Globex | 3.8 | 4.5 | 4.2 | 4.2 |\n"
        "| Initech | 4.2 | 3.0 | 4.0 | 3.7 |\n\n"
        "Which vendor scored highest overall?",
        {"expected_fields": ["quality", "delivery", "price", "overall"],
         "expected_relationships": ["Globex"],
         "noise_tokens": [],
         "completeness_fields": ["4.5", "4.0", "3.5", "Acme",
                                 "3.8", "4.5", "4.2", "Globex",
                                 "4.2", "3.0", "Initech"]},
        "easy"))

    items.append(_ex("layout", "table_in_layout",
        "[DOCUMENT LAYOUT]\n"
        "PROJECT MILESTONE TRACKER\n"
        "Project: Mercury Launch\n\n"
        "| Milestone | Due Date | Status | Owner |\n"
        "| Requirements | Feb 28 | Complete | Sarah |\n"
        "| Design | Mar 15 | Complete | Tom |\n"
        "| Development | Apr 30 | In Progress | Team |\n"
        "| Testing | May 31 | Not Started | QA |\n"
        "| Launch | Jun 15 | Not Started | PM |\n\n"
        "What milestones are remaining and when are they due?",
        {"expected_fields": ["development", "testing", "launch", "due date", "status"],
         "expected_relationships": ["Mercury Launch"],
         "noise_tokens": [],
         "completeness_fields": ["Apr 30", "In Progress", "May 31",
                                 "Not Started", "Jun 15"]},
        "easy"))

    return items


# ===================================================================
# Track 3 -- OCR & Vision  (50 examples)
# ===================================================================

def _build_ocr_vision() -> List[dict]:
    items: List[dict] = []

    # -- printed_text_extraction (10) ---
    for i, (prompt, ref_text, diff) in enumerate([
        ("What does this letter say about the start date and location?",
         "start date is April 14, 2025 report to Building C Room 302 at 9:00 AM", "easy"),
        ("Extract all shipping details from this label.",
         "FROM: DataWare Inc TO: PharmaGen Labs TRACKING: 1Z999AA10123456784 WEIGHT: 12.5 lbs SERVICE: 2-Day Air", "easy"),
        ("Extract all contact information from this business card.",
         "ALEXANDRA CHEN Chief Technology Officer NexGen Solutions alex.chen@nexgen.io 415 555-0188 San Francisco CA", "easy"),
        ("What are the key details of this legal notice?",
         "Case No 2025-CV-88412 Blue Sky Enterprises LLC default hearing April 22 2025 10:00 AM Courtroom 5B", "easy"),
        ("Extract the certificate details.",
         "PRIYA SHARMA Advanced Data Engineering Professional Program 120 hours Score 94 Certificate ID ADEP-2025-04421 TechEd Global Academy", "easy"),
        ("What is the parking violation for?",
         "Vehicle Honda Civic plate XYZ-789 violation expired meter location 200 Main St fine $75 due date April 15 2025", "easy"),
        ("Read the apartment listing details.",
         "2BR 1BA 950 sqft Rent $1850 month Available April 1 pet friendly parking included laundry in unit", "easy"),
        ("What does the warranty card say?",
         "Product SmartWatch Pro serial SWP-2025-44521 warranty 24 months purchase date March 1 2025 register at warranty.smartwatch.com", "easy"),
        ("Extract the conference badge information.",
         "ATTENDEE Dr. Sarah Kim ORGANIZATION MIT ROLE Speaker TALK AI in Healthcare SESSION March 20 Hall B", "medium"),
        ("Read the product nutrition label.",
         "Serving Size 1 cup Calories 120 Total Fat 3g Sodium 140mg Carbs 22g Protein 4g", "easy"),
    ]):
        ctx_block = (
            "[IMAGE: document_scan.png]\n"
            f"[OCR_OUTPUT: document_scan.png / confidence: 0.89]\n"
            f"{ref_text}\n\n"
        )
        items.append(_ex("ocr_vision", "printed_text_extraction",
            ctx_block + prompt,
            {"printed_text": ref_text, "handwritten_text": "",
             "diagram_elements": [], "table_data": [], "overlay_text": ""},
            diff))

    # -- handwriting_recognition (10) ---
    for i, (prompt, ref_hw, diff) in enumerate([
        ("What action items are in these handwritten notes?",
         "Call vendor re pricing due Fri Budget approval needed from Lisa Server migration pushed to April New hire starts Monday set up desk", "medium"),
        ("Extract the clinical information from these doctor notes.",
         "R Martinez 45M chest pain 2 days BP 142/88 HR 92 EKG normal sinus rhythm Stress test follow-up 1 week ASA 81mg daily", "hard"),
        ("What issues were found during this inspection?",
         "Building 7 Floor 3 T Nakamura Cracked tile elevator Exit sign bulb NE stairwell Fire extinguisher expired 02/2025 REPLACE remediation before occupancy", "medium"),
        ("What are the sprint goals and who is responsible?",
         "Sprint Goals Week 12 auth module Jake API rate limiting Sarah DB migration script Tom Update docs everyone BLOCKER CI pipeline broken", "medium"),
        ("What does this sticky note say?",
         "URGENT Client meeting moved to 2pm Conference room 4B Bring Q1 deck contracts Maria", "easy"),
        ("Transcribe the handwritten shopping list.",
         "eggs milk bread butter cheese yogurt apples chicken rice pasta olive oil onions garlic", "easy"),
        ("What does the handwritten lab notebook entry say?",
         "Experiment 42 pH 7.2 temperature 37C incubation 24 hours colony count 156 result positive for E. coli contamination suspected", "hard"),
        ("Read the handwritten phone message.",
         "For Dr. Kim From insurance company Re claim 77432 Please call back 555-0188 before 4pm today regarding authorization", "medium"),
        ("What meeting notes are on this page?",
         "Q2 planning discussed budget cuts 10% across departments except engineering new office Denver approved timeline 6 months action items due Friday", "medium"),
        ("Transcribe the handwritten feedback form.",
         "Overall rating 4 out of 5 Strengths clear communication good teamwork Areas for improvement time management meeting deadlines Recommend for promotion yes", "medium"),
    ]):
        ctx_block = (
            "[IMAGE: handwritten_doc.png]\n"
            f"[OCR_OUTPUT: handwritten_doc.png / confidence: 0.72]\n"
            f"{ref_hw}\n\n"
        )
        items.append(_ex("ocr_vision", "handwriting_recognition",
            ctx_block + prompt,
            {"printed_text": "", "handwritten_text": ref_hw,
             "diagram_elements": [], "table_data": [], "overlay_text": ""},
            diff))

    # -- diagram_understanding (10) ---
    for i, (prompt, elements, diff) in enumerate([
        ("Describe the network architecture shown in this diagram.",
         ["Firewall", "Load Balancer", "Web Server", "App Server", "Database", "PostgreSQL"], "medium"),
        ("Explain the relationships in this ER diagram.",
         ["Customer", "Order", "Product", "OrderItem", "one-to-many", "foreign key"], "medium"),
        ("Describe the hiring process shown in this flowchart.",
         ["Receive Application", "Background Check", "Schedule Interview", "Send Offer", "Send Rejection"], "easy"),
        ("Describe the microservices architecture.",
         ["React", "Kong", "User Service", "Order Service", "Payment Service", "RabbitMQ", "PostgreSQL", "Redis"], "medium"),
        ("Explain the swim-lane process for order fulfillment.",
         ["Customer", "Sales Team", "Warehouse", "Finance", "Submit Order", "Ship", "Generate Invoice"], "medium"),
        ("Describe the CI/CD pipeline diagram.",
         ["Source Code", "Build", "Unit Tests", "Integration Tests", "Deploy Staging", "Deploy Production", "Monitor"], "easy"),
        ("What does this state machine diagram show?",
         ["Idle", "Processing", "Completed", "Failed", "Retry", "transition", "timeout"], "medium"),
        ("Explain the data flow in this ETL pipeline diagram.",
         ["Source Database", "Extract", "Transform", "Load", "Data Warehouse", "validation", "cleansing"], "medium"),
        ("Describe the class hierarchy in this UML diagram.",
         ["Vehicle", "Car", "Truck", "ElectricCar", "inheritance", "abstract", "interface"], "medium"),
        ("What does the system architecture show?",
         ["Client", "API Gateway", "Auth Service", "Database", "Cache", "Message Queue", "Worker"], "medium"),
    ]):
        ctx_block = (
            "[IMAGE: diagram.png]\n"
            f"[OCR_OUTPUT: diagram.png / confidence: 0.85]\n"
            f"Components detected: {', '.join(elements)}\n"
            f"Connections: {' -> '.join(elements[:4])}\n\n"
        )
        items.append(_ex("ocr_vision", "diagram_understanding",
            ctx_block + prompt,
            {"printed_text": "", "handwritten_text": "",
             "diagram_elements": elements, "table_data": [], "overlay_text": ""},
            diff))

    # -- table_from_image (10) ---
    for i, (prompt, table, diff) in enumerate([
        ("Reconstruct the financial table from this scanned image.",
         [["Quarter", "Revenue", "Expenses", "Profit"],
          ["Q1 2024", "$2.1M", "$1.8M", "$0.3M"],
          ["Q2 2024", "$2.4M", "$1.9M", "$0.5M"],
          ["Q3 2024", "$2.8M", "$2.1M", "$0.7M"],
          ["Q4 2024", "$3.1M", "$2.3M", "$0.8M"]], "easy"),
        ("Extract the lab results and identify any flagged values.",
         [["Test", "Result", "Reference Range", "Flag"],
          ["Glucose", "110", "70-100", "HIGH"],
          ["Cholesterol", "195", "<200", "Normal"],
          ["Triglycerides", "165", "<150", "HIGH"]], "medium"),
        ("Reconstruct the pricing table.",
         [["Service", "Monthly", "Annual", "Savings"],
          ["Basic Plan", "$29", "$290", "17%"],
          ["Professional", "$79", "$790", "17%"],
          ["Enterprise", "$199", "$1,990", "17%"]], "easy"),
        ("Who had the most absences this week?",
         [["Employee", "Mon", "Tue", "Wed", "Thu", "Fri"],
          ["J. Smith", "P", "P", "A", "P", "P"],
          ["K. Patel", "P", "P", "P", "P", "P"],
          ["L. Garcia", "A", "P", "P", "P", "A"]], "medium"),
        ("Reconstruct the grade sheet.",
         [["Student", "Midterm", "Final", "Project", "Grade"],
          ["Adams", "88", "92", "95", "A"],
          ["Baker", "72", "68", "80", "C+"],
          ["Clark", "95", "91", "88", "A-"]], "easy"),
        ("Extract the schedule information.",
         [["Time", "Monday", "Tuesday", "Wednesday"],
          ["9:00", "Math", "English", "Science"],
          ["10:00", "History", "Math", "Art"],
          ["11:00", "PE", "Science", "English"]], "easy"),
        ("Reconstruct the inventory count.",
         [["Shelf", "Item", "Expected", "Actual", "Discrepancy"],
          ["A1", "Bolts", "500", "487", "-13"],
          ["A2", "Nuts", "500", "502", "+2"],
          ["B1", "Washers", "1000", "1000", "0"]], "medium"),
        ("Extract the election results table.",
         [["Candidate", "Party", "Votes", "Percentage"],
          ["Smith", "Dem", "12500", "48.2%"],
          ["Jones", "Rep", "11800", "45.5%"],
          ["Other", "Ind", "1650", "6.3%"]], "easy"),
        ("Reconstruct the medication schedule.",
         [["Time", "Medication", "Dosage", "Route"],
          ["08:00", "Metformin", "500mg", "Oral"],
          ["12:00", "Lisinopril", "10mg", "Oral"],
          ["20:00", "Metformin", "500mg", "Oral"]], "medium"),
        ("Extract the sports statistics table.",
         [["Player", "Goals", "Assists", "Minutes"],
          ["Garcia", "12", "8", "1800"],
          ["Kim", "8", "15", "2100"],
          ["Patel", "15", "5", "1950"]], "easy"),
    ]):
        # Build a pipe-delimited table from the data
        table_text = " | ".join(table[0]) + "\n"
        for row in table[1:]:
            table_text += " | ".join(row) + "\n"
        ctx_block = (
            "[IMAGE: scanned_table.png]\n"
            f"[OCR_OUTPUT: scanned_table.png / confidence: 0.91]\n"
            f"{table_text}\n"
        )
        items.append(_ex("ocr_vision", "table_from_image",
            ctx_block + prompt,
            {"printed_text": "", "handwritten_text": "",
             "diagram_elements": [], "table_data": table, "overlay_text": ""},
            diff))

    # -- overlay_text (5) ---
    items.append(_ex("ocr_vision", "overlay_text",
        "[IMAGE: annotated_contract.png]\n"
        "[OCR CONTEXT]\n"
        "Section 5.2 Limitation of Liability\n"
        "In no event shall either party's liability exceed the total fees "
        "paid in the preceding 12-month period.\n\n"
        "[OVERLAY -- HANDWRITTEN ANNOTATION]\n"
        "\"Need to increase to 24 months -- legal review required\" -- JK 3/12\n\n"
        "What does the overlay annotation say?",
        {"printed_text": "Section 5.2 Limitation of Liability",
         "handwritten_text": "",
         "diagram_elements": [],
         "table_data": [],
         "overlay_text": "Need to increase to 24 months legal review required JK 3/12"},
        "medium"))

    items.append(_ex("ocr_vision", "overlay_text",
        "[IMAGE: blueprint_with_notes.png]\n"
        "[OCR CONTEXT]\n"
        "Floor Plan -- Building A, Level 2\n"
        "Total Area: 12,500 sq ft\n\n"
        "[OVERLAY -- STAMPS AND MARKS]\n"
        "APPROVED -- City Building Dept -- Permit #BP-2025-0221\n"
        "[Red circle around Break Room] \"Expand by 500 sq ft per client request\"\n\n"
        "What do the overlay marks indicate?",
        {"printed_text": "Floor Plan Building A Level 2 Total Area 12500 sq ft",
         "handwritten_text": "",
         "diagram_elements": [],
         "table_data": [],
         "overlay_text": "APPROVED City Building Dept Permit BP-2025-0221 Expand by 500 sq ft per client request"},
        "medium"))

    items.append(_ex("ocr_vision", "overlay_text",
        "[IMAGE: financial_report_marked.png]\n"
        "[OCR CONTEXT]\n"
        "Annual Revenue: $45.2M\n"
        "Operating Margin: 18.3%\n"
        "Net Income: $5.8M\n\n"
        "[OVERLAY -- STICKY NOTE]\n"
        "\"Check Q4 numbers -- seems too high vs forecast\" -- CFO\n\n"
        "What concerns are raised in the overlay annotations?",
        {"printed_text": "Annual Revenue 45.2M Operating Margin 18.3% Net Income 5.8M",
         "handwritten_text": "",
         "diagram_elements": [],
         "table_data": [],
         "overlay_text": "Check Q4 numbers seems too high vs forecast CFO"},
        "medium"))

    items.append(_ex("ocr_vision", "overlay_text",
        "[IMAGE: map_with_annotations.png]\n"
        "[OCR CONTEXT]\n"
        "Site Survey Map -- Parcel 44B\n"
        "Scale: 1 inch = 50 feet\n\n"
        "[OVERLAY]\n"
        "\"Easement line disputed -- see legal file 2024-E-112\"\n"
        "[Arrow pointing to NE corner] \"Encroachment by neighbor fence approx 3 ft\"\n\n"
        "What issues are noted on the map overlay?",
        {"printed_text": "Site Survey Map Parcel 44B",
         "handwritten_text": "",
         "diagram_elements": [],
         "table_data": [],
         "overlay_text": "Easement line disputed legal file 2024-E-112 Encroachment neighbor fence 3 ft"},
        "hard"))

    items.append(_ex("ocr_vision", "overlay_text",
        "[IMAGE: xray_with_notes.png]\n"
        "[OCR CONTEXT]\n"
        "Chest X-Ray -- PA View\n"
        "Patient: MRN 5501234\n\n"
        "[OVERLAY -- RADIOLOGIST ANNOTATION]\n"
        "[Circle in right lower lobe] \"Opacity -- recommend CT follow-up\"\n"
        "[Arrow to cardiac silhouette] \"Mildly enlarged, stable from prior\"\n\n"
        "What are the radiologist's findings?",
        {"printed_text": "Chest X-Ray PA View Patient MRN 5501234",
         "handwritten_text": "",
         "diagram_elements": [],
         "table_data": [],
         "overlay_text": "Opacity right lower lobe recommend CT follow-up cardiac silhouette Mildly enlarged stable from prior"},
        "hard"))

    # -- mixed_content (5) ---
    items.append(_ex("ocr_vision", "mixed_content",
        "[IMAGE: report_page.png]\n"
        "[OCR CONTEXT]\n"
        "SITE INSPECTION REPORT\n"
        "Inspector: Carlos Mendez | Date: 03/14/2025\n\n"
        "[PRINTED TABLE]\n"
        "| Area | Rating | Notes |\n"
        "| Lobby | Good | Minor wear |\n"
        "| Parking | Fair | Cracks in lot B |\n"
        "| HVAC | Poor | Unit 3 failing |\n\n"
        "[HANDWRITTEN]\n\"HVAC replacement quote needed by end of month\" -- CM\n\n"
        "Summarize all findings from this report.",
        {"printed_text": "SITE INSPECTION REPORT Inspector Carlos Mendez",
         "handwritten_text": "HVAC replacement quote needed by end of month",
         "diagram_elements": [],
         "table_data": [["Area", "Rating", "Notes"],
                        ["Lobby", "Good", "Minor wear"],
                        ["Parking", "Fair", "Cracks in lot B"],
                        ["HVAC", "Poor", "Unit 3 failing"]],
         "overlay_text": ""},
        "hard"))

    items.append(_ex("ocr_vision", "mixed_content",
        "[IMAGE: engineering_doc.png]\n"
        "[OCR CONTEXT]\n"
        "MOTOR SPECIFICATION -- Model MX-450\n"
        "Voltage: 480V AC, 3-Phase\n"
        "Power: 15 HP\n"
        "RPM: 1750\n\n"
        "[DIAGRAM: Wiring]\nL1-->T1 L2-->T2 L3-->T3 Ground-->Frame\n\n"
        "[HANDWRITTEN]\n\"bearings need replacement, order SKU BRG-254\" -- Maint\n\n"
        "Extract specifications, wiring, and maintenance notes.",
        {"printed_text": "MOTOR SPECIFICATION Model MX-450 Voltage 480V AC Power 15 HP RPM 1750",
         "handwritten_text": "bearings need replacement order SKU BRG-254",
         "diagram_elements": ["L1", "T1", "L2", "T2", "L3", "T3", "Ground"],
         "table_data": [],
         "overlay_text": ""},
        "hard"))

    items.append(_ex("ocr_vision", "mixed_content",
        "[IMAGE: tax_form.png]\n"
        "[OCR CONTEXT]\n"
        "FORM W-2 -- Wage and Tax Statement 2024\n"
        "Employer: TechStart Inc. EIN: 82-1234567\n"
        "Employee: David Park SSN: ***-**-4589\n\n"
        "| Box | Description | Amount |\n"
        "| 1 | Wages | $125,000 |\n"
        "| 2 | Federal tax | $22,500 |\n"
        "| 4 | SS tax | $7,750 |\n"
        "| 6 | Medicare tax | $1,812.50 |\n\n"
        "Extract all tax form information.",
        {"printed_text": "FORM W-2 Wage Tax Statement 2024 TechStart Inc 82-1234567 David Park",
         "handwritten_text": "",
         "diagram_elements": [],
         "table_data": [["Box", "Description", "Amount"],
                        ["1", "Wages", "$125,000"],
                        ["2", "Federal tax", "$22,500"],
                        ["4", "SS tax", "$7,750"],
                        ["6", "Medicare tax", "$1,812.50"]],
         "overlay_text": ""},
        "medium"))

    items.append(_ex("ocr_vision", "mixed_content",
        "[IMAGE: patient_chart.png]\n"
        "[OCR CONTEXT]\n"
        "PATIENT: Elena Vasquez | MRN: 7782014\n\n"
        "| Time | BP | HR | Temp | SpO2 |\n"
        "| 08:00 | 130/85 | 78 | 98.4 | 97% |\n"
        "| 12:00 | 128/82 | 80 | 98.6 | 98% |\n"
        "| 20:00 | 140/90 | 88 | 99.8 | 95% |\n\n"
        "[HANDWRITTEN]\n\"BP trending up, temp rising. Notified Dr. Park 20:30. "
        "Ordered blood cultures.\" -- RN Williams\n\n"
        "What trends and concerns are shown?",
        {"printed_text": "PATIENT Elena Vasquez MRN 7782014",
         "handwritten_text": "BP trending up temp rising Notified Dr Park 20:30 Ordered blood cultures RN Williams",
         "diagram_elements": [],
         "table_data": [["Time", "BP", "HR", "Temp", "SpO2"],
                        ["08:00", "130/85", "78", "98.4", "97%"],
                        ["12:00", "128/82", "80", "98.6", "98%"],
                        ["20:00", "140/90", "88", "99.8", "95%"]],
         "overlay_text": ""},
        "hard"))

    items.append(_ex("ocr_vision", "mixed_content",
        "[IMAGE: warehouse_check.png]\n"
        "[OCR CONTEXT]\n"
        "WAREHOUSE INVENTORY CHECK -- Bay 4\n\n"
        "| Shelf | Item | Expected | Actual | Diff |\n"
        "| A1 | Bolts M8 | 500 | 487 | -13 |\n"
        "| A2 | Nuts M8 | 500 | 502 | +2 |\n"
        "| B2 | Spring Pins | 200 | 178 | -22 |\n\n"
        "[HANDWRITTEN]\n\"Spring pins -- possible theft? Report to supervisor\" -- JT\n\n"
        "Summarize findings and concerns.",
        {"printed_text": "WAREHOUSE INVENTORY CHECK Bay 4",
         "handwritten_text": "Spring pins possible theft Report to supervisor",
         "diagram_elements": [],
         "table_data": [["Shelf", "Item", "Expected", "Actual", "Diff"],
                        ["A1", "Bolts M8", "500", "487", "-13"],
                        ["A2", "Nuts M8", "500", "502", "+2"],
                        ["B2", "Spring Pins", "200", "178", "-22"]],
         "overlay_text": ""},
        "medium"))

    return items


# ===================================================================
# Track 4 -- Context & Reasoning  (50 examples)
# ===================================================================

def _build_reasoning() -> List[dict]:
    items: List[dict] = []

    # -- multi_document_synthesis (10) ---
    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Q1 2025 Financial Report]\n"
        "Revenue: $12.4M (up 8% YoY). Operating expenses rose 12% to $10.8M "
        "due to new hires. Net income: $1.1M.\n\n"
        "[DOCUMENT 2: Board Meeting Minutes]\n"
        "Board expressed concern about rising OpEx outpacing revenue growth. "
        "CFO committed to 5% cost reduction by Q3.\n\n"
        "[DOCUMENT 3: HR Headcount Report]\n"
        "Added 22 engineers in Q1. Attrition rate: 8%.\n\n"
        "Analyze the financial health considering all three documents.",
        {"expected_conclusion": "Revenue growth outpaced by expense growth but cost reduction plan addresses this",
         "key_evidence": ["12.4M", "8%", "12%", "10.8M", "5% cost reduction", "22 engineers"],
         "reasoning_steps": 4,
         "key_terms": ["revenue", "expenses", "growth", "cost reduction"]},
        "hard"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Vendor A Proposal]\n"
        "Price: $180,000/year. SLA: 99.9% uptime. Support: 24/7.\n\n"
        "[DOCUMENT 2: Vendor B Proposal]\n"
        "Price: $145,000/year. SLA: 99.5% uptime. Support: Business hours email.\n\n"
        "[DOCUMENT 3: Internal Requirements]\n"
        "Mission-critical system. Budget: $200K/year. Required uptime: 99.9%. Need 24/7 support.\n\n"
        "Which vendor should we select and why?",
        {"expected_conclusion": "Vendor A meets 99.9% uptime and 24/7 support requirements",
         "key_evidence": ["99.9%", "24/7", "mission-critical", "$180,000", "$145,000"],
         "reasoning_steps": 4,
         "key_terms": ["uptime", "support", "requirements", "SLA"]},
        "medium"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Performance Review -- Sarah Chen]\n"
        "Rating: Exceeds Expectations. Led migration of 3 legacy systems. "
        "Mentored 2 junior engineers.\n\n"
        "[DOCUMENT 2: Compensation Policy]\n"
        "Exceeds Expectations: 8-12% raise. Promotion requires manager recommendation "
        "+ 2 consecutive top ratings.\n\n"
        "[DOCUMENT 3: Manager Notes]\n"
        "Sarah had top rating last year too. Recommend promotion to Senior Engineer.\n\n"
        "Should Sarah Chen be promoted?",
        {"expected_conclusion": "Yes, Sarah qualifies with two consecutive top ratings and manager recommendation",
         "key_evidence": ["Exceeds Expectations", "3 legacy systems", "2 consecutive top ratings", "manager recommendation"],
         "reasoning_steps": 3,
         "key_terms": ["promotion", "performance", "rating", "recommendation"]},
        "easy"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Insurance Policy]\n"
        "Covers water damage from burst pipes. Deductible: $1,000. Max: $50,000.\n\n"
        "[DOCUMENT 2: Claim Report]\n"
        "Cause: Pipe burst. Damage: flooring $8K, drywall $3.5K, furniture $2.2K. Total: $13,700.\n\n"
        "[DOCUMENT 3: Adjuster Notes]\n"
        "Confirmed pipe burst. Furniture damage partially pre-existing.\n\n"
        "What amount should be paid on this claim?",
        {"expected_conclusion": "Approximately $12,700 minus $1,000 deductible with reduction for pre-existing furniture damage",
         "key_evidence": ["burst pipes", "$1,000 deductible", "$13,700", "pre-existing"],
         "reasoning_steps": 4,
         "key_terms": ["coverage", "deductible", "claim", "pre-existing"]},
        "hard"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Market Research]\n"
        "TAM: $2.5B, growing 12% annually. Top competitor: 35% share.\n\n"
        "[DOCUMENT 2: Product Roadmap]\n"
        "Q2: API marketplace. Q3: No-code connectors. Q4: Enterprise SSO.\n\n"
        "[DOCUMENT 3: Sales Pipeline]\n"
        "50 enterprise leads. 20% conversion. Avg deal: $150K. Main objection: lack of compliance certs.\n\n"
        "Assess the company's market strategy.",
        {"expected_conclusion": "Strategy addresses integration pain point but compliance gap delays enterprise adoption",
         "key_evidence": ["$2.5B", "12%", "compliance", "20% conversion", "no-code"],
         "reasoning_steps": 4,
         "key_terms": ["TAM", "compliance", "conversion", "enterprise"]},
        "hard"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Audit Report]\nFindings: 3 critical, 8 high, 12 medium vulnerabilities.\n\n"
        "[DOCUMENT 2: Incident Log]\n2 data breaches in 2024. Average response time: 6 hours.\n\n"
        "[DOCUMENT 3: Budget]\nSecurity budget: $200K. Industry benchmark: $500K.\n\n"
        "Assess the security posture and recommend priorities.",
        {"expected_conclusion": "Critically deficient security posture requiring immediate budget increase and vulnerability remediation",
         "key_evidence": ["3 critical", "2 data breaches", "$200K", "$500K"],
         "reasoning_steps": 4,
         "key_terms": ["vulnerability", "breach", "budget", "remediation"]},
        "medium"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Lease Agreement]\nMonthly rent: $3,500. 24-month term from Jan 1, 2025. "
        "3% annual increase.\n\n"
        "[DOCUMENT 2: AP Ledger]\nJan 2025: $3,500. Feb: $3,500. Mar: $3,605.\n\n"
        "[DOCUMENT 3: Building Policy]\nAll increases apply on lease anniversary.\n\n"
        "Is there a discrepancy in the rent payments?",
        {"expected_conclusion": "March rent of $3,605 is incorrect; 3% increase applies only at anniversary in January 2026",
         "key_evidence": ["$3,500", "$3,605", "annual increase", "3%", "anniversary"],
         "reasoning_steps": 3,
         "key_terms": ["discrepancy", "rent", "annual increase"]},
        "medium"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Drug Trial Protocol]\nPhase II trial for drug XR-42. "
        "Primary endpoint: 30% reduction in symptoms at 12 weeks.\n\n"
        "[DOCUMENT 2: Interim Results]\nWeek 6: 18% reduction in treatment group vs 8% placebo. "
        "p-value: 0.03. Adverse events: 12% vs 5%.\n\n"
        "[DOCUMENT 3: Safety Board Minutes]\nTwo serious adverse events reported. "
        "Board recommends continued monitoring.\n\n"
        "Should the trial continue? Assess based on all evidence.",
        {"expected_conclusion": "Trial shows promising efficacy but elevated adverse event rate warrants careful monitoring",
         "key_evidence": ["30%", "18%", "8%", "p-value 0.03", "adverse events 12%", "serious adverse events"],
         "reasoning_steps": 4,
         "key_terms": ["efficacy", "safety", "adverse events", "trial"]},
        "hard"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Energy Audit]\nBuilding: 50,000 sqft, built 1995. "
        "Annual energy cost: $180K. Gas furnace (80% efficiency).\n\n"
        "[DOCUMENT 2: Upgrade Proposal]\nHeat pump system (300% COP). Cost: $120K. "
        "Savings: $85K/year. Utility rebate: $30K.\n\n"
        "Is the energy upgrade worth the investment?",
        {"expected_conclusion": "Yes, payback period approximately 1 year after rebate with significant ongoing savings",
         "key_evidence": ["$180K", "80%", "300% COP", "$120K", "$85K/year", "$30K rebate"],
         "reasoning_steps": 3,
         "key_terms": ["payback", "savings", "efficiency", "investment"]},
        "easy"))

    items.append(_ex("reasoning", "multi_document_synthesis",
        "[DOCUMENT 1: Market Entry Analysis]\nTarget: Southeast Asia. Population: 680M. "
        "GDP growth: 5.2%. Smartphone: 78%. E-commerce: 20% CAGR.\n\n"
        "[DOCUMENT 2: Risk Assessment]\nRegulatory fragmentation across 10 countries. "
        "Local competitors entrenched.\n\n"
        "[DOCUMENT 3: Company Profile]\nAI platform. $50M war chest. Strong US brand.\n\n"
        "Should we enter the Southeast Asian market?",
        {"expected_conclusion": "Market attractive but regulatory fragmentation and local competitors require phased entry",
         "key_evidence": ["680M", "5.2%", "78%", "10 countries", "local competitors", "$50M"],
         "reasoning_steps": 4,
         "key_terms": ["market entry", "regulatory", "competition", "growth"]},
        "hard"))

    # -- causal_reasoning (10) ---
    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Incident Report -- Production Outage March 12, 2025]\n"
        "14:00 -- Deploy v2.8.1. 14:15 -- Memory spike 60% to 95%. "
        "14:22 -- Customer timeouts. 14:30 -- Auto-scaler failed. "
        "14:45 -- Root cause: N+1 query. 15:00 -- Rollback to v2.8.0.\n\n"
        "Analyze the causal chain that led to this outage.",
        {"expected_conclusion": "N+1 query in v2.8.1 caused memory spike leading to timeouts; auto-scaler ineffective because same bug",
         "key_evidence": ["v2.8.1", "memory spike", "N+1 query", "auto-scaler", "rollback"],
         "reasoning_steps": 5,
         "key_terms": ["deploy", "memory", "N+1", "rollback", "root cause"]},
        "medium"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Sales Decline Analysis]\n"
        "Q4 sales dropped 18% vs Q3. Factors: competitor free tier (Oct), "
        "15% price increase (Sept), key account manager left (Nov). "
        "Survey: 60% cited price.\n\n"
        "What is the primary cause of the sales decline?",
        {"expected_conclusion": "Price increase combined with competitor free tier is primary cause per customer survey",
         "key_evidence": ["18%", "free tier", "15% price increase", "60% cited price"],
         "reasoning_steps": 4,
         "key_terms": ["price", "competitor", "decline", "survey"]},
        "medium"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Quality Control Report]\n"
        "Defect rate: 0.5% to 2.3% in February. New supplier Jan 15 with "
        "8% higher impurities. Machine calibration deferred. "
        "Night shift 2x defect rate. 3 of 5 new operators still in training.\n\n"
        "Determine the root cause(s) of the defect increase.",
        {"expected_conclusion": "New supplier material impurities combined with deferred calibration and undertrained operators",
         "key_evidence": ["0.5% to 2.3%", "new supplier", "8% higher impurity", "calibration deferred", "training"],
         "reasoning_steps": 5,
         "key_terms": ["defect", "supplier", "impurity", "calibration", "training"]},
        "hard"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Website Analytics]\n"
        "March traffic dropped 25%. Organic down 40% (Google update Feb 28). "
        "Paid up 10%. Bounce rate 45% to 62%. Blog paused in February.\n\n"
        "Why did website traffic decline?",
        {"expected_conclusion": "Google algorithm update caused organic traffic loss compounded by paused content",
         "key_evidence": ["25%", "organic down 40%", "algorithm update", "bounce rate 62%", "content paused"],
         "reasoning_steps": 3,
         "key_terms": ["organic", "algorithm", "traffic", "content"]},
        "medium"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Hospital Readmission Analysis]\n"
        "Readmission rose from 12% to 19% in Q1. Factors: average stay reduced 1.2 days, "
        "digital-only discharge instructions, 2-week follow-up backlog, "
        "pharmacy reconciliation errors up 35%.\n\n"
        "What is driving the readmission increase?",
        {"expected_conclusion": "Shorter stays combined with pharmacy errors and delayed follow-ups",
         "key_evidence": ["12% to 19%", "1.2 days", "digital-only", "2 weeks backlog", "pharmacy errors 35%"],
         "reasoning_steps": 4,
         "key_terms": ["readmission", "discharge", "pharmacy", "follow-up"]},
        "hard"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Customer Churn Analysis]\n"
        "Churn increased from 3% to 7% over 6 months. API reliability dropped to 98.5% "
        "(target 99.9%). Support response time: 24hrs (was 4hrs). "
        "Competitor launched similar product at 20% lower price.\n\n"
        "Why is churn increasing?",
        {"expected_conclusion": "API reliability issues and degraded support response time are primary drivers, exacerbated by competitive pricing pressure",
         "key_evidence": ["3% to 7%", "98.5%", "99.9%", "24hrs", "4hrs", "20% lower price"],
         "reasoning_steps": 4,
         "key_terms": ["churn", "reliability", "support", "competitor"]},
        "medium"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Construction Delay Report]\n"
        "Project 60 days behind schedule. Root causes: permits delayed 3 weeks, "
        "steel delivery late 2 weeks due to supplier bankruptcy, "
        "rain days: 15 (normal: 5), inspector availability limited.\n\n"
        "What caused the construction delay?",
        {"expected_conclusion": "Multiple causes: permit delays, supplier bankruptcy affecting steel delivery, and abnormal weather",
         "key_evidence": ["60 days", "permits delayed 3 weeks", "steel late 2 weeks", "supplier bankruptcy", "rain days 15"],
         "reasoning_steps": 4,
         "key_terms": ["delay", "permits", "supplier", "weather"]},
        "medium"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Employee Satisfaction Survey]\n"
        "Overall satisfaction dropped from 4.2 to 3.4. Factors: return-to-office "
        "mandate (Jan), 2% pay raises vs 6% inflation, management restructure "
        "with 3 VP departures, parking fees introduced.\n\n"
        "What is causing the satisfaction decline?",
        {"expected_conclusion": "RTO mandate and below-inflation raises are primary factors, amplified by leadership instability",
         "key_evidence": ["4.2 to 3.4", "return-to-office", "2% vs 6%", "3 VP departures", "parking fees"],
         "reasoning_steps": 4,
         "key_terms": ["satisfaction", "return-to-office", "inflation", "management"]},
        "medium"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: App Performance Degradation]\n"
        "P95 latency increased from 200ms to 800ms over 2 weeks. "
        "Database size grew 40% (archival job failed). Cache hit rate dropped "
        "from 95% to 70%. No code deployments in this period.\n\n"
        "What caused the performance degradation?",
        {"expected_conclusion": "Failed archival job caused database bloat which reduced cache effectiveness",
         "key_evidence": ["200ms to 800ms", "database grew 40%", "archival job failed", "cache hit rate 70%"],
         "reasoning_steps": 3,
         "key_terms": ["latency", "database", "archival", "cache"]},
        "medium"))

    items.append(_ex("reasoning", "causal_reasoning",
        "[DOCUMENT: Supply Chain Disruption]\n"
        "Product delivery delayed 3 weeks. Shipping container stuck at port "
        "due to labor dispute. Alternative air freight cost 5x more. "
        "50 customer orders affected. Penalty clauses totaling $120K.\n\n"
        "Analyze the cascade effects of this disruption.",
        {"expected_conclusion": "Labor dispute caused port congestion leading to delays, creating costly air freight alternative and penalty exposure",
         "key_evidence": ["3 weeks", "labor dispute", "air freight 5x", "50 orders", "$120K penalties"],
         "reasoning_steps": 4,
         "key_terms": ["disruption", "port", "labor", "penalties", "air freight"]},
        "medium"))

    # -- comparative_analysis (10) ---
    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nAWS: $0.023/GB, 99.99% SLA, 200+ services, 33% market share.\n"
        "Azure: $0.018/GB, 99.95% SLA, 150+ services, 28% share, best MS integration.\n"
        "GCP: $0.020/GB, 99.95% SLA, 100+ services, best for ML.\n\n"
        "Our needs: Heavy Microsoft stack, moderate ML, budget-sensitive.\n\n"
        "Recommend the best cloud provider.",
        {"expected_conclusion": "Azure is best fit due to Microsoft integration and lower storage costs",
         "key_evidence": ["Microsoft stack", "$0.018/GB", "99.95%", "budget-sensitive"],
         "reasoning_steps": 4,
         "key_terms": ["Azure", "Microsoft", "storage", "budget"]},
        "medium"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DOCUMENT 1: Office A]\nDowntown. $45/sqft. 10K sqft. Parking: $200/spot. Near metro.\n\n"
        "[DOCUMENT 2: Office B]\nSuburbs. $28/sqft. 15K sqft. Free parking. No transit.\n\n"
        "80 employees, 60% drive, 40% transit.\nCompare total cost and employee needs.",
        {"expected_conclusion": "Option B cheaper overall but transit access for 40% is a concern",
         "key_evidence": ["$45/sqft", "$28/sqft", "60% drive", "40% transit", "$200/spot", "metro"],
         "reasoning_steps": 5,
         "key_terms": ["rent", "parking", "transit", "cost"]},
        "hard"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nSAP: License $500K, Impl $800K, 12 months. Industry-leading.\n"
        "NetSuite: License $180K, Impl $250K, 4 months. Limited manufacturing.\n\n"
        "Company: $50M revenue, 200 employees, manufacturing + distribution.\n\n"
        "Which ERP should we choose?",
        {"expected_conclusion": "SAP more suitable for manufacturing despite higher cost; NetSuite lacks manufacturing modules",
         "key_evidence": ["$500K", "$180K", "manufacturing", "12 months", "4 months"],
         "reasoning_steps": 3,
         "key_terms": ["ERP", "manufacturing", "cost", "implementation"]},
        "medium"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nProduct A: $49, NPS 72, Churn 5%, CSAT 4.2\n"
        "Product B: $39, NPS 68, Churn 8%, CSAT 3.9\n"
        "Product C: $69, NPS 85, Churn 3%, CSAT 4.6\n\n"
        "Which product offers the best value?",
        {"expected_conclusion": "Product A offers best balance of price and quality; Product C is premium; Product B has concerning churn",
         "key_evidence": ["NPS 72", "churn 5%", "$49", "NPS 85", "churn 8%"],
         "reasoning_steps": 3,
         "key_terms": ["NPS", "churn", "value", "price"]},
        "medium"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nBond Fund: 4.5% return, Low risk, High liquidity.\n"
        "Growth Fund: 12% return, High risk.\n"
        "Real Estate Fund: 8% return, Medium risk, Low liquidity.\n\n"
        "Investor: 55 years old, moderate risk, needs liquidity, $100K.\n\n"
        "Recommend allocation.",
        {"expected_conclusion": "Balanced allocation favoring bonds and moderate growth, limited real estate due to liquidity needs",
         "key_evidence": ["55 years", "moderate risk", "liquidity", "4.5%", "12%", "8%"],
         "reasoning_steps": 4,
         "key_terms": ["risk", "return", "liquidity", "allocation"]},
        "hard"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nCandidate A: PhD, 2 yrs exp, $140K, needs visa.\n"
        "Candidate B: MS, 5 yrs exp, $125K, available immediately.\n"
        "Candidate C: BS, 8 yrs exp, team lead at competitor, $150K, 4-week notice.\n\n"
        "Team needs: Production ML, urgent delivery.\nWho should we hire?",
        {"expected_conclusion": "Candidate B best fit: industry experience, available immediately, within budget",
         "key_evidence": ["5 years", "industry", "$125K", "available immediately", "production ML", "urgent"],
         "reasoning_steps": 4,
         "key_terms": ["candidate", "experience", "available", "production"]},
        "medium"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nLanguage A (Python): Easy learning curve, large ecosystem, slower runtime.\n"
        "Language B (Rust): Steep learning curve, memory safety, fast runtime.\n"
        "Language C (Go): Moderate learning curve, good concurrency, moderate ecosystem.\n\n"
        "Project: High-throughput API serving millions of requests.\nWhich language?",
        {"expected_conclusion": "Go offers best balance of performance and developer productivity for high-throughput API",
         "key_evidence": ["high-throughput", "concurrency", "learning curve", "fast runtime"],
         "reasoning_steps": 3,
         "key_terms": ["performance", "concurrency", "throughput", "API"]},
        "medium"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nDB A (PostgreSQL): ACID, JSON support, mature, complex queries.\n"
        "DB B (MongoDB): Flexible schema, horizontal scaling, document model.\n"
        "DB C (DynamoDB): Serverless, single-digit ms latency, pay-per-request.\n\n"
        "Use case: IoT platform, 100K events/second, simple key-value lookups, "
        "variable traffic.\nWhich database?",
        {"expected_conclusion": "DynamoDB best suited for high-throughput key-value IoT with variable traffic and serverless scaling",
         "key_evidence": ["100K events/second", "key-value", "serverless", "variable traffic"],
         "reasoning_steps": 3,
         "key_terms": ["IoT", "throughput", "serverless", "scaling"]},
        "medium"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nInsurer A: Premium $450/mo, Deductible $1000, Network: Large, Rating: A+.\n"
        "Insurer B: Premium $320/mo, Deductible $3000, Network: Medium, Rating: A.\n"
        "Insurer C: Premium $550/mo, Deductible $500, Network: Large, Rating: A+.\n\n"
        "Family of 4, moderate healthcare needs, prefer large network.\nWhich plan?",
        {"expected_conclusion": "Insurer A offers best balance of reasonable premium, low deductible, and large network",
         "key_evidence": ["$450/mo", "$1000 deductible", "Large network", "A+", "family of 4"],
         "reasoning_steps": 3,
         "key_terms": ["premium", "deductible", "network", "family"]},
        "easy"))

    items.append(_ex("reasoning", "comparative_analysis",
        "[DATA]\nCRM A (Salesforce): $150/user/mo. Enterprise features. Complex setup.\n"
        "CRM B (HubSpot): $50/user/mo. SMB focused. Easy setup. Limited reporting.\n"
        "CRM C (Zoho): $35/user/mo. Good value. Moderate features. API limitations.\n\n"
        "Company: 15 sales reps, startup, budget-conscious, need quick deployment.\n"
        "Which CRM?",
        {"expected_conclusion": "HubSpot offers best fit for startup: reasonable price, easy setup, SMB focus",
         "key_evidence": ["$50/user", "SMB", "easy setup", "15 reps", "startup", "budget"],
         "reasoning_steps": 3,
         "key_terms": ["CRM", "startup", "budget", "setup"]},
        "easy"))

    # -- contradiction_detection (10) ---
    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Sales Report]\nNew customers: 145. CAC: $320. Marketing spend: $52K.\n\n"
        "[DOCUMENT 2: Marketing Dashboard]\nLeads: 890. Conversion: 12%. Marketing spend: $48.5K.\n\n"
        "Identify inconsistencies.",
        {"expected_conclusion": "Marketing spend differs ($52K vs $48.5K); 12% of 890 is 107, not 145 customers",
         "key_evidence": ["$52,000", "$48,500", "145", "890", "12%"],
         "reasoning_steps": 3,
         "key_terms": ["inconsistency", "marketing spend", "conversion"]},
        "medium"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Compliance Statement]\nAll employees completed security training. Rate: 100%.\n\n"
        "[DOCUMENT 2: Training Report]\nTotal: 250. Completed: 237. Pending: 13.\n\n"
        "Are there contradictions?",
        {"expected_conclusion": "Statement claims 100% but 13 employees pending per training system",
         "key_evidence": ["100%", "250", "237", "13 pending"],
         "reasoning_steps": 2,
         "key_terms": ["contradiction", "compliance", "100%", "pending"]},
        "easy"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: CEO Letter]\n\"Strong growth in all segments. Employee satisfaction at all-time high. "
        "Invested heavily in R&D.\"\n\n"
        "[DOCUMENT 2: Financials]\nConsumer up 15%, Enterprise down 3%, Government flat. "
        "Turnover: 22% (avg: 15%). R&D decreased 8%.\n\n"
        "Identify contradictions.",
        {"expected_conclusion": "Three contradictions: not all segments grew, turnover suggests low satisfaction, R&D decreased",
         "key_evidence": ["all segments", "Enterprise down 3%", "all-time high", "22% turnover", "R&D decreased 8%"],
         "reasoning_steps": 4,
         "key_terms": ["contradiction", "segments", "satisfaction", "R&D"]},
        "hard"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Project Status]\nOn track. Budget: 75% used with 2 months remaining.\n\n"
        "[DOCUMENT 2: Vendor Invoices]\nInvoiced: $380K of $400K budget. Outstanding POs: $85K.\n\n"
        "Is the project on track financially?",
        {"expected_conclusion": "Project will exceed budget: $380K + $85K = $465K against $400K budget",
         "key_evidence": ["75%", "$380,000", "$400,000", "$85,000", "on track"],
         "reasoning_steps": 3,
         "key_terms": ["budget", "over budget", "outstanding"]},
        "medium"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Press Release]\n\"Company achieved record profitability with net income of $15M.\"\n\n"
        "[DOCUMENT 2: SEC Filing]\nNet income: $15M. Includes one-time gain of $12M from asset sale. "
        "Operating income: $3M (down 25% YoY).\n\n"
        "What is misleading about the press release?",
        {"expected_conclusion": "Record profitability is misleading because $12M of the $15M is a one-time gain; operating income actually declined",
         "key_evidence": ["$15M", "one-time gain $12M", "operating income $3M", "down 25%"],
         "reasoning_steps": 3,
         "key_terms": ["misleading", "one-time", "operating income", "record"]},
        "hard"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Job Posting]\n\"Minimum 5 years experience required. Master's degree preferred.\"\n\n"
        "[DOCUMENT 2: Selected Candidate Resume]\nBS Computer Science. 3 years experience. "
        "Strong GitHub portfolio.\n\n"
        "Does the selected candidate meet the posted requirements?",
        {"expected_conclusion": "Candidate does not meet minimum 5-year experience requirement and lacks preferred Master's degree",
         "key_evidence": ["5 years required", "3 years experience", "Master's preferred", "BS"],
         "reasoning_steps": 2,
         "key_terms": ["requirements", "experience", "degree", "contradiction"]},
        "easy"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Vendor Contract]\nPayment terms: Net 30. Penalty: 2% for late payment.\n\n"
        "[DOCUMENT 2: AP Records]\nInvoice dated Jan 15. Payment made Feb 28. No penalty applied.\n\n"
        "Is there a compliance issue?",
        {"expected_conclusion": "Payment was 14 days late (44 days vs Net 30) but no penalty was applied, violating contract terms",
         "key_evidence": ["Net 30", "Jan 15", "Feb 28", "2% penalty", "no penalty"],
         "reasoning_steps": 3,
         "key_terms": ["late payment", "Net 30", "penalty", "compliance"]},
        "medium"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Environmental Report]\n\"Zero emissions achieved at all facilities.\"\n\n"
        "[DOCUMENT 2: EPA Filing]\nFacility A: 0 tons CO2. Facility B: 0 tons CO2. "
        "Facility C: 420 tons CO2 (diesel generators). Carbon offsets purchased: 420 tons.\n\n"
        "Is the zero emissions claim accurate?",
        {"expected_conclusion": "Technically misleading: Facility C emits 420 tons but claims zero through offsets, not actual zero emissions",
         "key_evidence": ["zero emissions", "420 tons CO2", "diesel generators", "carbon offsets"],
         "reasoning_steps": 3,
         "key_terms": ["zero emissions", "offsets", "misleading", "diesel"]},
        "hard"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Inventory System]\nItem X: 500 units in stock. Location: Warehouse A.\n\n"
        "[DOCUMENT 2: Physical Count]\nItem X: 423 units found. 77 units unaccounted for.\n\n"
        "[DOCUMENT 3: Shipping Log]\nItem X: 45 units shipped but not yet deducted from system.\n\n"
        "Reconcile the inventory discrepancy.",
        {"expected_conclusion": "System shows 500 but 45 are shipped pending deduction; remaining 32 units (77-45) truly unaccounted for",
         "key_evidence": ["500 units", "423 units", "77 unaccounted", "45 shipped"],
         "reasoning_steps": 3,
         "key_terms": ["inventory", "discrepancy", "shipped", "unaccounted"]},
        "hard"))

    items.append(_ex("reasoning", "contradiction_detection",
        "[DOCUMENT 1: Board Minutes]\n\"Unanimously approved $2M marketing budget.\"\n\n"
        "[DOCUMENT 2: Board Member Email]\n\"I voted against the marketing budget increase. "
        "Please correct the minutes.\"\n\n"
        "What is the contradiction?",
        {"expected_conclusion": "Minutes claim unanimous approval but at least one board member voted against",
         "key_evidence": ["unanimously approved", "voted against", "correct the minutes"],
         "reasoning_steps": 2,
         "key_terms": ["unanimous", "voted against", "contradiction", "minutes"]},
        "easy"))

    # -- evidence_based_conclusion (10) ---
    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Patient Case Study]\n"
        "62-year-old male, hypertension, Type 2 diabetes.\n"
        "Labs: HbA1c 8.2% (target <7%), creatinine 1.8mg/dL, BP 148/92.\n"
        "Meds: Metformin 1000mg BID, Lisinopril 20mg.\n"
        "A1c trend: 7.5% -> 7.8% -> 8.2% over 9 months.\n\n"
        "What treatment adjustments should be considered?",
        {"expected_conclusion": "Diabetes worsening, need to adjust meds; kidney function declining suggests caution with metformin",
         "key_evidence": ["8.2%", "<7%", "creatinine 1.8", "7.5% -> 7.8% -> 8.2%", "Metformin"],
         "reasoning_steps": 4,
         "key_terms": ["HbA1c", "creatinine", "metformin", "diabetes", "kidney"]},
        "hard"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Hiring Analysis]\n"
        "Open: Data Scientist. Final candidates:\n"
        "A: PhD, 2yr exp, $140K, visa needed.\n"
        "B: MS, 5yr exp, industry, $125K, available now.\n"
        "C: BS, 8yr exp, team lead at competitor, $150K, 4-week notice.\n"
        "Team needs: Production ML, urgent timeline.\n\n"
        "Who should we hire?",
        {"expected_conclusion": "Candidate B: relevant experience, available immediately, within budget, matches production needs",
         "key_evidence": ["5 years", "industry", "$125K", "available immediately", "production ML"],
         "reasoning_steps": 4,
         "key_terms": ["candidate", "experience", "available", "production"]},
        "medium"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Cybersecurity Assessment]\n"
        "Pen test: 3 critical (SQL injection), 8 high (outdated SSL), 12 medium.\n"
        "Last test: 18 months ago (policy: annual). 2 breaches in 2024.\n"
        "Security budget: $200K (benchmark: $500K).\n\n"
        "Assess security posture and recommend actions.",
        {"expected_conclusion": "Critically deficient; immediate SQL injection remediation and budget increase needed",
         "key_evidence": ["3 critical", "SQL injection", "18 months", "2 breaches", "$200K", "$500K"],
         "reasoning_steps": 4,
         "key_terms": ["vulnerability", "SQL injection", "breach", "budget"]},
        "medium"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Real Estate Analysis]\n"
        "Property A: $450K, 3BR, 1500sqft, good schools, 45min commute.\n"
        "Property B: $380K, 2BR, 1100sqft, average schools, 15min commute.\n"
        "Property C: $520K, 4BR, 2000sqft, good schools, 30min commute.\n\n"
        "Family: 2 kids (school age), dual income, budget $500K.\n\n"
        "Which property best fits this family's needs?",
        {"expected_conclusion": "Property A balances good schools, adequate space, and stays within budget despite longer commute",
         "key_evidence": ["$450K", "3BR", "good schools", "$500K budget", "2 kids"],
         "reasoning_steps": 3,
         "key_terms": ["schools", "budget", "commute", "family"]},
        "easy"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Manufacturing Decision]\n"
        "Option A: In-house production. Setup: $2M, unit cost: $8, capacity: 100K/year.\n"
        "Option B: Outsource to China. No setup, unit cost: $5, MOQ: 50K, lead time: 8 weeks.\n"
        "Option C: Outsource to Mexico. Setup: $500K, unit cost: $6.50, lead time: 2 weeks.\n\n"
        "Expected demand: 80K units/year. Quality is critical.\n\n"
        "Which manufacturing strategy should we pursue?",
        {"expected_conclusion": "Option C offers best balance: moderate cost, fast lead time, nearshore quality control",
         "key_evidence": ["$2M setup", "$5 unit", "$6.50 unit", "2 weeks", "8 weeks", "quality"],
         "reasoning_steps": 4,
         "key_terms": ["cost", "lead time", "quality", "nearshore"]},
        "medium"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Software Migration Decision]\n"
        "Current: On-prem Oracle DB, $180K/year license, 2 DBAs needed.\n"
        "Option: Migrate to AWS Aurora PostgreSQL. Migration cost: $300K.\n"
        "Projected savings: $120K/year (no license, 1 DBA). Migration: 6 months.\n"
        "Risk: 3 weeks estimated downtime for data migration.\n\n"
        "Should we migrate?",
        {"expected_conclusion": "Migration worthwhile with 2.5 year payback but downtime risk needs mitigation planning",
         "key_evidence": ["$180K/year", "$300K", "$120K/year savings", "6 months", "3 weeks downtime"],
         "reasoning_steps": 3,
         "key_terms": ["migration", "payback", "savings", "downtime", "risk"]},
        "medium"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Marketing Channel Analysis]\n"
        "Google Ads: $50K spend, 200 leads, 25 conversions, avg deal $5K.\n"
        "LinkedIn: $30K spend, 80 leads, 15 conversions, avg deal $12K.\n"
        "Content Marketing: $20K spend, 150 leads, 10 conversions, avg deal $8K.\n"
        "Events: $40K spend, 50 leads, 20 conversions, avg deal $15K.\n\n"
        "Where should we increase investment?",
        {"expected_conclusion": "Events have highest ROI (7.5x) and conversion rate; LinkedIn second best for high-value deals",
         "key_evidence": ["$50K", "$30K", "$20K", "$40K", "200 leads", "25 conversions", "$15K", "$12K"],
         "reasoning_steps": 4,
         "key_terms": ["ROI", "conversion", "channel", "investment"]},
        "hard"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Employee Retention Study]\n"
        "Exit interviews (n=45): 40% cited compensation, 30% cited management, "
        "20% cited growth opportunities, 10% cited remote policy.\n"
        "Industry salary benchmarks show company pays 8% below median.\n"
        "Manager training budget: $0 last 2 years.\n"
        "Attrition cost: ~$50K per employee.\n\n"
        "What retention strategies should be prioritized?",
        {"expected_conclusion": "Compensation adjustment is top priority given 40% cite it and 8% gap; manager training second",
         "key_evidence": ["40% compensation", "30% management", "8% below median", "$0 training", "$50K attrition cost"],
         "reasoning_steps": 4,
         "key_terms": ["compensation", "management", "retention", "attrition"]},
        "medium"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Product Launch Readiness]\n"
        "Feature completeness: 92%. Bug backlog: 15 P1, 42 P2. "
        "Performance: meets targets. Security audit: 2 findings (1 critical). "
        "Documentation: 70% complete. Beta feedback: NPS 65.\n\n"
        "Is the product ready to launch?",
        {"expected_conclusion": "Not ready: critical security finding must be resolved and P1 bugs need fixing before launch",
         "key_evidence": ["92%", "15 P1", "1 critical security", "70% documentation", "NPS 65"],
         "reasoning_steps": 4,
         "key_terms": ["security", "bugs", "readiness", "launch", "critical"]},
        "medium"))

    items.append(_ex("reasoning", "evidence_based_conclusion",
        "[DOCUMENT: Warehouse Consolidation]\n"
        "Current: 3 warehouses. W1: 60% utilized, $120K/mo. "
        "W2: 45% utilized, $95K/mo. W3: 80% utilized, $150K/mo.\n"
        "Total capacity: 150K sqft. Used: 95K sqft.\n"
        "Consolidation to 2 warehouses: saves $95K/mo but requires $200K moving cost.\n\n"
        "Should we consolidate?",
        {"expected_conclusion": "Yes, consolidation saves ~$1.14M/year for $200K one-time cost; payback under 3 months",
         "key_evidence": ["60%", "45%", "80%", "$120K", "$95K", "$150K", "$200K moving"],
         "reasoning_steps": 3,
         "key_terms": ["consolidation", "utilization", "savings", "payback"]},
        "easy"))

    return items


# ===================================================================
# Track 5 -- KG-Augmented  (50 examples)
# ===================================================================

def _build_kg() -> List[dict]:
    items: List[dict] = []

    # -- entity_lookup (10) ---
    for i, (prompt, entities, rels, cites, diff) in enumerate([
        ("Who is the CEO of ACME Corp and what is the company's revenue?",
         ["ACME_CORP", "JOHN_SMITH"], ["JOHN_SMITH is CEO of ACME_CORP"], ["ACME_CORP", "JOHN_SMITH"], "easy"),
        ("What are the details of the contract between TechPro and MedCo?",
         ["CONTRACT_2025_001", "TECHPRO_LLC", "MEDCO_INC"], ["TECHPRO_LLC provides service to MEDCO_INC"], ["CONTRACT_2025_001"], "easy"),
        ("Who manages Project Alpha and which department owns it?",
         ["PROJ_ALPHA", "SARAH_LEE", "DEPT_ENG"], ["SARAH_LEE manages PROJ_ALPHA", "PROJ_ALPHA owned by DEPT_ENG"], ["PROJ_ALPHA", "SARAH_LEE", "DEPT_ENG"], "easy"),
        ("List all HR policies and their effective dates.",
         ["POLICY_HR_001", "POLICY_HR_002", "DEPT_HR"], ["DEPT_HR authored POLICY_HR_001"], ["POLICY_HR_001", "POLICY_HR_002"], "easy"),
        ("Which vendor has the highest rating?",
         ["VENDOR_300", "PremiumFreight"], ["PremiumFreight has rating 4.8"], ["VENDOR_300"], "easy"),
        ("What is the status of incident INC-2025-001?",
         ["INC_001", "TEAM_INFRA"], ["INC_001 assigned to TEAM_INFRA"], ["INC_001"], "easy"),
        ("Who is the primary contact for account ACC-500?",
         ["ACC_500", "CONTACT_JANE"], ["CONTACT_JANE is primary contact for ACC_500"], ["ACC_500", "CONTACT_JANE"], "easy"),
        ("What certifications does employee EMP-042 hold?",
         ["EMP_042", "CERT_AWS", "CERT_PMP"], ["EMP_042 holds CERT_AWS"], ["EMP_042"], "easy"),
        ("What is the warranty status of product PROD-X200?",
         ["PROD_X200", "WARRANTY_001"], ["WARRANTY_001 covers PROD_X200"], ["PROD_X200", "WARRANTY_001"], "easy"),
        ("Which office location has the most employees?",
         ["LOC_NYC", "LOC_SF", "LOC_CHI"], ["LOC_NYC has 200 employees"], ["LOC_NYC"], "easy"),
    ]):
        kg_block = (
            "[KG CONTEXT]\n"
            f"Entities: {', '.join(entities)}\n"
            f"Relationships: {'; '.join(rels)}\n\n"
        )
        items.append(_ex("kg", "entity_lookup",
            kg_block + prompt,
            {"expected_entities": entities,
             "expected_relationships": rels,
             "expected_citations": cites},
            diff))

    # -- relationship_reasoning (10) ---
    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "Entity: DRUG_A [Cardiostat, Beta-blocker]\n"
        "Entity: DRUG_B [Renopril, ACE-inhibitor]\n"
        "Entity: DRUG_C [Diuretix, Diuretic]\n"
        "DRUG_A --[treats]--> Hypertension\n"
        "DRUG_B --[treats]--> Hypertension\n"
        "DRUG_C --[treats]--> Hypertension\n"
        "DRUG_A --[interacts_with]--> DRUG_C [moderate]\n"
        "DRUG_B --[contraindicated_with]--> DRUG_A [hypotension risk]\n\n"
        "Patient on Cardiostat needs additional hypertension medication. Which is safe?",
        {"expected_entities": ["DRUG_A", "DRUG_B", "DRUG_C", "Cardiostat", "Diuretix"],
         "expected_relationships": ["DRUG_B contraindicated with DRUG_A", "DRUG_A interacts with DRUG_C"],
         "expected_citations": ["DRUG_A", "DRUG_C"]},
        "hard"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "EMP_001 [Alice, Engineering] --[reports_to]--> EMP_004 [Dave]\n"
        "EMP_002 [Bob, Engineering] --[reports_to]--> EMP_004\n"
        "EMP_004 --[reports_to]--> EMP_005 [Eve, VP Engineering]\n"
        "EMP_003 [Carol, Sales] --[collaborates_with]--> EMP_001\n\n"
        "Trace the reporting chain from Alice to VP level.",
        {"expected_entities": ["EMP_001", "EMP_004", "EMP_005", "Alice", "Dave", "Eve"],
         "expected_relationships": ["Alice reports to Dave", "Dave reports to Eve"],
         "expected_citations": ["EMP_001", "EMP_004", "EMP_005"]},
        "medium"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "SUPPLIER_A [ChipMaker] --[supplies]--> COMPONENT_1 [CPU Module] (lead: 6 weeks)\n"
        "SUPPLIER_B [BoardWorks] --[supplies]--> COMPONENT_2 [PCB Board] (lead: 2 weeks)\n"
        "PRODUCT_X [SmartWidget] --[requires]--> COMPONENT_1\n"
        "PRODUCT_X --[requires]--> COMPONENT_2\n\n"
        "What is the critical path for manufacturing SmartWidget?",
        {"expected_entities": ["SUPPLIER_A", "PRODUCT_X", "COMPONENT_1", "ChipMaker"],
         "expected_relationships": ["SUPPLIER_A supplies COMPONENT_1", "lead time 6 weeks"],
         "expected_citations": ["SUPPLIER_A", "PRODUCT_X"]},
        "medium"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "CASE_2024_001 [Settled] plaintiff: TechStart, defendant: DataGuard\n"
        "CASE_2024_002 [Active] plaintiff: DataGuard, defendant: James Wong\n"
        "James Wong --[former_employee_of]--> DataGuard\n\n"
        "What is DataGuard's involvement in both cases?",
        {"expected_entities": ["DataGuard", "CASE_2024_001", "CASE_2024_002", "James Wong"],
         "expected_relationships": ["DataGuard defendant in CASE_2024_001", "DataGuard plaintiff in CASE_2024_002"],
         "expected_citations": ["CASE_2024_001", "CASE_2024_002"]},
        "hard"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "FUND_A [Growth, 12%, High risk]\n"
        "FUND_B [Bond, 4%, Low risk]\n"
        "FUND_C [Balanced, 7%, Medium risk]\n"
        "CLIENT_1 [Mr. Park, age 62, Low risk tolerance]\n"
        "FUND_B --[inverse_correlation]--> FUND_A (-0.3)\n"
        "CLIENT_1 --[holds]--> FUND_B ($200K)\n\n"
        "Should Mr. Park diversify into other funds?",
        {"expected_entities": ["CLIENT_1", "FUND_B", "FUND_C", "Mr. Park"],
         "expected_relationships": ["CLIENT_1 holds FUND_B", "Mr. Park risk tolerance Low"],
         "expected_citations": ["CLIENT_1", "FUND_B", "FUND_C"]},
        "hard"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "REGULATION_GDPR [Right to erasure]\n"
        "REGULATION_CCPA [Right to delete]\n"
        "SYSTEM_CRM [SalesPro, personal data, US+EU]\n"
        "GDPR --[applies_to]--> SYSTEM_CRM [EU]\n"
        "CCPA --[applies_to]--> SYSTEM_CRM [US_CA]\n"
        "REQUEST_001 [Data Deletion, EU citizen]\n\n"
        "What regulations govern this deletion request?",
        {"expected_entities": ["REGULATION_GDPR", "SYSTEM_CRM", "REQUEST_001", "GDPR", "CCPA"],
         "expected_relationships": ["GDPR applies to SYSTEM_CRM"],
         "expected_citations": ["REGULATION_GDPR", "SYSTEM_CRM"]},
        "hard"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "SERVER_01 [PROD-DB-01, RHEL 8] --[hosts]--> APP_CORE\n"
        "SERVER_02 [PROD-APP-01, Windows 2016] --[hosts]--> APP_WEB\n"
        "CVE_2025_001 [Critical, affects Windows 2016]\n"
        "CVE_2025_001 --[affects]--> SERVER_02\n\n"
        "Which servers are at risk?",
        {"expected_entities": ["SERVER_02", "CVE_2025_001", "APP_WEB"],
         "expected_relationships": ["CVE_2025_001 affects SERVER_02", "SERVER_02 hosts APP_WEB"],
         "expected_citations": ["SERVER_02", "CVE_2025_001"]},
        "medium"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "PATENT_A [Filed 2023, Status: Granted, Owner: TechCo]\n"
        "PATENT_B [Filed 2024, Status: Pending, Owner: TechCo]\n"
        "PRODUCT_Z [SmartLock, uses: PATENT_A technology]\n"
        "COMPETITOR_X [filed objection to PATENT_B]\n\n"
        "What is the IP landscape for TechCo?",
        {"expected_entities": ["PATENT_A", "PATENT_B", "PRODUCT_Z", "TechCo", "COMPETITOR_X"],
         "expected_relationships": ["PRODUCT_Z uses PATENT_A", "COMPETITOR_X objection to PATENT_B"],
         "expected_citations": ["PATENT_A", "PATENT_B"]},
        "medium"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "STUDENT_A [GPA 3.8, Major: CS, Advisor: Dr. Kim]\n"
        "STUDENT_B [GPA 3.2, Major: CS, Advisor: Dr. Kim]\n"
        "COURSE_ML [prereq: COURSE_STATS, COURSE_LINALG]\n"
        "STUDENT_A --[completed]--> COURSE_STATS, COURSE_LINALG\n"
        "STUDENT_B --[completed]--> COURSE_STATS\n\n"
        "Which students are eligible for the ML course?",
        {"expected_entities": ["STUDENT_A", "COURSE_ML", "COURSE_STATS", "COURSE_LINALG"],
         "expected_relationships": ["STUDENT_A completed all prerequisites", "STUDENT_B missing LINALG"],
         "expected_citations": ["STUDENT_A", "COURSE_ML"]},
        "medium"))

    items.append(_ex("kg", "relationship_reasoning",
        "[KG CONTEXT]\n"
        "ALLERGEN_PEANUT --[present_in]--> PRODUCT_COOKIE\n"
        "ALLERGEN_GLUTEN --[present_in]--> PRODUCT_COOKIE\n"
        "ALLERGEN_DAIRY --[present_in]--> PRODUCT_CAKE\n"
        "PATIENT_X [allergies: PEANUT, DAIRY]\n\n"
        "Which products can Patient X safely consume?",
        {"expected_entities": ["PATIENT_X", "PRODUCT_COOKIE", "PRODUCT_CAKE", "ALLERGEN_PEANUT"],
         "expected_relationships": ["PEANUT present in COOKIE", "DAIRY present in CAKE"],
         "expected_citations": ["PATIENT_X", "PRODUCT_COOKIE", "PRODUCT_CAKE"]},
        "medium"))

    # -- multi_hop_query (10) ---
    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "ORG_1 [Apex Manufacturing] --[buys_from]--> ORG_2 [BrightSteel] (Steel Alloy)\n"
        "ORG_1 --[manufactures]--> PRODUCT_1 [Industrial Valve]\n"
        "ORG_3 [CoreLogistics] --[ships_for]--> ORG_1\n\n"
        "Trace the full supply chain for the Industrial Valve.",
        {"expected_entities": ["ORG_1", "ORG_2", "ORG_3", "PRODUCT_1", "Apex Manufacturing", "BrightSteel"],
         "expected_relationships": ["BrightSteel supplies Steel Alloy to Apex", "Apex manufactures Industrial Valve"],
         "expected_citations": ["ORG_1", "ORG_2", "ORG_3"]},
        "medium"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "DOC_A [Master Service Agreement] effective: 2023-01-01\n"
        "DOC_B [SOW Amendment 3] --[amends]--> DOC_A\n"
        "DOC_C [Change Order 12] --[references]--> DOC_B\n"
        "DOC_D [Invoice INV-2025-0088, $45K] --[bills_for]--> DOC_C\n\n"
        "What document chain justifies Invoice INV-2025-0088?",
        {"expected_entities": ["DOC_A", "DOC_B", "DOC_C", "DOC_D"],
         "expected_relationships": ["DOC_D bills for DOC_C", "DOC_C references DOC_B", "DOC_B amends DOC_A"],
         "expected_citations": ["DOC_A", "DOC_B", "DOC_C", "DOC_D"]},
        "medium"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "PATIENT_1 [Elena Vasquez] --[diagnosed]--> COND_1 [Diabetes], COND_2 [Hypertension]\n"
        "DR_2 [Dr. Singh, Endocrinology] --[prescribed]--> MED_1 [Metformin] for COND_1\n"
        "DR_1 [Dr. Park, Cardiology] --[prescribed]--> MED_2 [Atenolol] for COND_2\n"
        "MED_1 --[interaction_warning]--> MED_2 [monitor]\n\n"
        "Which doctors treat Elena and are there medication concerns?",
        {"expected_entities": ["PATIENT_1", "DR_1", "DR_2", "MED_1", "MED_2", "Elena Vasquez"],
         "expected_relationships": ["Dr. Singh prescribed Metformin", "Dr. Park prescribed Atenolol", "interaction warning"],
         "expected_citations": ["PATIENT_1", "DR_1", "DR_2", "MED_1", "MED_2"]},
        "hard"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "RISK_001 [Data breach, High likelihood, Critical impact]\n"
        "CONTROL_A [Encryption, 80% effectiveness] --[mitigates]--> RISK_001\n"
        "CONTROL_B [Access logging, 60%] --[mitigates]--> RISK_001\n"
        "CONTROL_C [MFA, 90%] --[mitigates]--> RISK_001\n"
        "AUDIT_2025 --[found_gap_in]--> CONTROL_B\n\n"
        "What is the residual risk considering audit findings?",
        {"expected_entities": ["RISK_001", "CONTROL_A", "CONTROL_B", "CONTROL_C", "AUDIT_2025"],
         "expected_relationships": ["CONTROL_B mitigates RISK_001 at 60%", "AUDIT_2025 found gap in CONTROL_B"],
         "expected_citations": ["RISK_001", "CONTROL_B", "AUDIT_2025"]},
        "hard"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "DEPT_MKT --[runs]--> CAMPAIGN_1 [Spring Launch, cost $200K]\n"
        "CAMPAIGN_1 --[generates]--> LEAD_BATCH [500 leads, High quality]\n"
        "DEPT_SALES --[works]--> LEAD_BATCH\n"
        "LEAD_BATCH --[converts_to]--> DEAL_POOL [$3M value, 25% close rate]\n\n"
        "What is the ROI of the Spring Launch campaign?",
        {"expected_entities": ["CAMPAIGN_1", "LEAD_BATCH", "DEAL_POOL", "DEPT_MKT"],
         "expected_relationships": ["CAMPAIGN_1 generates LEAD_BATCH", "LEAD_BATCH converts to DEAL_POOL"],
         "expected_citations": ["CAMPAIGN_1", "DEAL_POOL"]},
        "medium"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "PERSON_EX [Tom Wu, departed 2025-01-15]\n"
        "CLAUSE_NDA [Non-solicitation, 24 months]\n"
        "CLAUSE_NONCOMPETE [Non-compete, 12 months]\n"
        "PERSON_EX --[bound_by]--> CLAUSE_NDA, CLAUSE_NONCOMPETE\n"
        "PERSON_EX --[joined]--> COMPETITOR_X [date: 2025-02-01]\n\n"
        "Is Tom Wu violating contractual obligations?",
        {"expected_entities": ["PERSON_EX", "CLAUSE_NDA", "CLAUSE_NONCOMPETE", "COMPETITOR_X", "Tom Wu"],
         "expected_relationships": ["Tom Wu bound by non-compete 12 months", "Tom Wu joined competitor"],
         "expected_citations": ["PERSON_EX", "CLAUSE_NONCOMPETE", "COMPETITOR_X"]},
        "hard"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "POLICY_INS [General Liability, $5M limit, $25K deductible]\n"
        "INCIDENT_001 [Visitor slip and fall, $150K est. damage]\n"
        "CLAIM_001 --[under]--> POLICY_INS\n"
        "POLICY_INS --[exclusion]--> Intentional acts\n\n"
        "Is this claim covered?",
        {"expected_entities": ["POLICY_INS", "INCIDENT_001", "CLAIM_001"],
         "expected_relationships": ["CLAIM_001 under POLICY_INS", "slip and fall not intentional"],
         "expected_citations": ["POLICY_INS", "CLAIM_001"]},
        "medium"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "VENDOR_ABC [ABC Consulting]\n"
        "PO_001 [$50K, Approved] --[issued_to]--> VENDOR_ABC\n"
        "INV_001 [$50K, Unpaid] --[from]--> VENDOR_ABC\n"
        "INV_001 --[references]--> PO_001\n"
        "PO_001 --[charged_to]--> BUDGET_IT [$120K remaining]\n\n"
        "Should we approve payment for invoice INV_001?",
        {"expected_entities": ["VENDOR_ABC", "PO_001", "INV_001", "BUDGET_IT"],
         "expected_relationships": ["INV_001 references PO_001", "amounts match $50K"],
         "expected_citations": ["PO_001", "INV_001", "BUDGET_IT"]},
        "medium"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "BUILDING_A --[connected_to]--> HVAC_SYSTEM_1\n"
        "HVAC_SYSTEM_1 --[maintained_by]--> VENDOR_COOLCO\n"
        "HVAC_SYSTEM_1 --[last_service]--> 2024-06-15\n"
        "HVAC_SYSTEM_1 --[service_interval]--> 6 months\n"
        "TENANT_X --[occupies]--> BUILDING_A [complaints: 3 about temperature]\n\n"
        "Why is Tenant X complaining about temperature?",
        {"expected_entities": ["BUILDING_A", "HVAC_SYSTEM_1", "VENDOR_COOLCO", "TENANT_X"],
         "expected_relationships": ["HVAC overdue for service", "TENANT_X occupies BUILDING_A"],
         "expected_citations": ["HVAC_SYSTEM_1", "TENANT_X"]},
        "medium"))

    items.append(_ex("kg", "multi_hop_query",
        "[KG CONTEXT]\n"
        "EMPLOYEE_A [Dev, Team Lead] --[created]--> PR_100 [code change]\n"
        "PR_100 --[reviewed_by]--> EMPLOYEE_B [Dev]\n"
        "PR_100 --[merged_to]--> BRANCH_MAIN\n"
        "BRANCH_MAIN --[deployed_to]--> ENV_PROD [March 15]\n"
        "INCIDENT_500 [March 15, regression] --[caused_by]--> PR_100\n\n"
        "Trace the chain from code change to production incident.",
        {"expected_entities": ["EMPLOYEE_A", "PR_100", "BRANCH_MAIN", "ENV_PROD", "INCIDENT_500"],
         "expected_relationships": ["EMPLOYEE_A created PR_100", "PR_100 merged to MAIN", "INCIDENT caused by PR_100"],
         "expected_citations": ["PR_100", "INCIDENT_500"]},
        "medium"))

    # -- entity_disambiguation (10) ---
    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "PERSON_J_SMITH_1 [John Smith, Engineering, E1001] --[works_on]--> PROJ_ALPHA\n"
        "PERSON_J_SMITH_2 [John Smith, Legal, E2045] --[reviews]--> CONTRACT_100\n"
        "PERSON_J_SMITH_3 [James Smith, Engineering, E1022]\n\n"
        "Which John Smith works on Project Alpha?",
        {"expected_entities": ["PERSON_J_SMITH_1", "E1001"],
         "expected_relationships": ["PERSON_J_SMITH_1 works on PROJ_ALPHA"],
         "expected_citations": ["PERSON_J_SMITH_1", "E1001"]},
        "medium"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "ORG_MERCURY_1 [Mercury Solutions, IT, Austin]\n"
        "ORG_MERCURY_2 [Mercury Insurance, Insurance, LA]\n"
        "CONTRACT_500 [vendor: ORG_MERCURY_1, $300K]\n"
        "CLAIM_800 [insurer: ORG_MERCURY_2, $15K]\n\n"
        "We have a payment to Mercury. IT contract or insurance claim?",
        {"expected_entities": ["ORG_MERCURY_1", "ORG_MERCURY_2", "CONTRACT_500", "CLAIM_800"],
         "expected_relationships": ["CONTRACT_500 vendor Mercury Solutions", "CLAIM_800 insurer Mercury Insurance"],
         "expected_citations": ["ORG_MERCURY_1", "ORG_MERCURY_2"]},
        "medium"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "DOC_REV_1 [Security Policy v2.0, 2024-01-15, Superseded]\n"
        "DOC_REV_2 [Security Policy v3.0, 2025-03-01, Active]\n"
        "DOC_REV_2 --[supersedes]--> DOC_REV_1\n\n"
        "Which Security Policy version is currently in effect?",
        {"expected_entities": ["DOC_REV_2", "Security Policy"],
         "expected_relationships": ["DOC_REV_2 supersedes DOC_REV_1"],
         "expected_citations": ["DOC_REV_2"]},
        "easy"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "ACCT_1001 [Operating Account, First National, $450K]\n"
        "ACCT_1002 [Payroll Account, First National, $180K]\n"
        "ACCT_1003 [Operating Account, City Trust, $320K]\n\n"
        "What is the total balance across all operating accounts?",
        {"expected_entities": ["ACCT_1001", "ACCT_1003"],
         "expected_relationships": ["ACCT_1001 at First National", "ACCT_1003 at City Trust"],
         "expected_citations": ["ACCT_1001", "ACCT_1003"]},
        "medium"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "LOC_CHICAGO_HQ [Chicago Office, HQ, 200 employees]\n"
        "LOC_CHICAGO_WH [Chicago Warehouse, 50000 sqft]\n"
        "LOC_CHICAGO_DC [Chicago Data Center, 120 racks]\n\n"
        "We need to add headcount. Which Chicago location has employees?",
        {"expected_entities": ["LOC_CHICAGO_HQ", "Chicago Office"],
         "expected_relationships": ["LOC_CHICAGO_HQ has 200 employees"],
         "expected_citations": ["LOC_CHICAGO_HQ"]},
        "easy"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "PROJECT_PHOENIX_2024 [Status: Completed, Budget: $1.2M]\n"
        "PROJECT_PHOENIX_2025 [Status: Active, Budget: $2.0M]\n"
        "TASK_001 --[belongs_to]--> PROJECT_PHOENIX_2025\n\n"
        "What is the budget for the active Phoenix project?",
        {"expected_entities": ["PROJECT_PHOENIX_2025"],
         "expected_relationships": ["PROJECT_PHOENIX_2025 is Active"],
         "expected_citations": ["PROJECT_PHOENIX_2025"]},
        "easy"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "PRODUCT_WIDGET_V1 [Widget, Version 1.0, Discontinued]\n"
        "PRODUCT_WIDGET_V2 [Widget, Version 2.0, Active, Price: $49.99]\n"
        "ORDER_100 --[contains]--> PRODUCT_WIDGET_V2\n\n"
        "What is the current price of the Widget?",
        {"expected_entities": ["PRODUCT_WIDGET_V2"],
         "expected_relationships": ["PRODUCT_WIDGET_V2 is Active"],
         "expected_citations": ["PRODUCT_WIDGET_V2"]},
        "easy"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "MEETING_BOARD_Q1 [Board Meeting, Jan 15, 2025]\n"
        "MEETING_BOARD_Q2 [Board Meeting, Apr 15, 2025]\n"
        "ACTION_ITEM_42 --[from]--> MEETING_BOARD_Q1 [status: overdue]\n\n"
        "Which board meeting has overdue action items?",
        {"expected_entities": ["MEETING_BOARD_Q1", "ACTION_ITEM_42"],
         "expected_relationships": ["ACTION_ITEM_42 from Q1 meeting"],
         "expected_citations": ["MEETING_BOARD_Q1"]},
        "easy"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "INVOICE_ABC_001 [From: ABC Corp, Date: Jan 2025, $5K, Paid]\n"
        "INVOICE_ABC_002 [From: ABC Corp, Date: Mar 2025, $8K, Unpaid]\n"
        "INVOICE_ABC_003 [From: ABC Logistics, Date: Mar 2025, $3K, Unpaid]\n\n"
        "Which ABC invoice from March is from ABC Corp?",
        {"expected_entities": ["INVOICE_ABC_002", "ABC Corp"],
         "expected_relationships": ["INVOICE_ABC_002 from ABC Corp"],
         "expected_citations": ["INVOICE_ABC_002"]},
        "medium"))

    items.append(_ex("kg", "entity_disambiguation",
        "[KG CONTEXT]\n"
        "SERVER_PROD_DB [prod-db-01, PostgreSQL, Critical]\n"
        "SERVER_STAGING_DB [staging-db-01, PostgreSQL, Low]\n"
        "ALERT_001 [High CPU] --[on]--> SERVER_PROD_DB\n\n"
        "Which database server has the alert and what is its criticality?",
        {"expected_entities": ["SERVER_PROD_DB", "ALERT_001"],
         "expected_relationships": ["ALERT_001 on SERVER_PROD_DB"],
         "expected_citations": ["SERVER_PROD_DB"]},
        "easy"))

    # -- cross_document_kg (10) ---
    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "VENDOR_XYZ [XYZ Logistics]\n"
        "PO_100 [$80K, Approved] --[issued_to]--> VENDOR_XYZ\n"
        "INV_200 [$80K, Unpaid] --[from]--> VENDOR_XYZ\n"
        "INV_200 --[references]--> PO_100\n"
        "BUDGET_OPS [$250K remaining]\n"
        "PO_100 --[charged_to]--> BUDGET_OPS\n\n"
        "Verify the invoice and budget before approving payment.",
        {"expected_entities": ["VENDOR_XYZ", "PO_100", "INV_200", "BUDGET_OPS"],
         "expected_relationships": ["INV_200 references PO_100", "PO_100 charged to BUDGET_OPS"],
         "expected_citations": ["PO_100", "INV_200", "BUDGET_OPS"]},
        "medium"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "REGULATION_HIPAA [Health data protection]\n"
        "SYSTEM_EHR [ElectraHealth EHR, PHI data]\n"
        "REGULATION_HIPAA --[applies_to]--> SYSTEM_EHR\n"
        "AUDIT_FINDING_001 [encryption gap in SYSTEM_EHR]\n"
        "REMEDIATION_PLAN [due: April 30, 2025]\n\n"
        "What is the compliance status of the EHR system?",
        {"expected_entities": ["REGULATION_HIPAA", "SYSTEM_EHR", "AUDIT_FINDING_001", "REMEDIATION_PLAN"],
         "expected_relationships": ["HIPAA applies to SYSTEM_EHR", "encryption gap found"],
         "expected_citations": ["SYSTEM_EHR", "AUDIT_FINDING_001"]},
        "medium"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "EMPLOYEE_A [Alice, Senior Engineer, Performance: Exceeds]\n"
        "EMPLOYEE_B [Bob, Engineer, Performance: Meets]\n"
        "BUDGET_RAISE [$50K pool, Engineering]\n"
        "POLICY_COMP [Exceeds: 8-12% raise, Meets: 3-5% raise]\n"
        "EMPLOYEE_A --[salary]--> $120K\n"
        "EMPLOYEE_B --[salary]--> $95K\n\n"
        "Calculate raise amounts based on policy and budget constraints.",
        {"expected_entities": ["EMPLOYEE_A", "EMPLOYEE_B", "BUDGET_RAISE", "POLICY_COMP"],
         "expected_relationships": ["Alice Exceeds gets 8-12%", "Bob Meets gets 3-5%"],
         "expected_citations": ["EMPLOYEE_A", "EMPLOYEE_B", "POLICY_COMP"]},
        "hard"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "LEASE_001 [Office A, $45K/mo, expires: Dec 2025]\n"
        "LEASE_002 [Office B, $30K/mo, expires: Mar 2026]\n"
        "HEADCOUNT_PLAN [+50 employees by Q3 2025]\n"
        "OFFICE_A [capacity: 150, current: 140]\n"
        "OFFICE_B [capacity: 80, current: 60]\n\n"
        "Can the current offices accommodate the headcount growth?",
        {"expected_entities": ["LEASE_001", "LEASE_002", "HEADCOUNT_PLAN", "OFFICE_A", "OFFICE_B"],
         "expected_relationships": ["OFFICE_A near capacity", "OFFICE_B has 20 spots"],
         "expected_citations": ["OFFICE_A", "OFFICE_B", "HEADCOUNT_PLAN"]},
        "medium"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "INCIDENT_100 [Production outage, March 15, Priority: P1]\n"
        "CHANGE_200 [Deploy v2.8, March 15 14:00]\n"
        "SERVICE_APP [Core API, SLA: 99.9%]\n"
        "INCIDENT_100 --[caused_by]--> CHANGE_200\n"
        "INCIDENT_100 --[affected]--> SERVICE_APP\n"
        "SERVICE_APP --[downtime]--> 45 minutes\n\n"
        "What was the impact of the outage on SLA compliance?",
        {"expected_entities": ["INCIDENT_100", "CHANGE_200", "SERVICE_APP"],
         "expected_relationships": ["CHANGE_200 caused INCIDENT_100", "SERVICE_APP had 45 min downtime"],
         "expected_citations": ["INCIDENT_100", "SERVICE_APP"]},
        "medium"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "TRAINING_SEC [Security Awareness, mandatory, annual]\n"
        "EMPLOYEE_LIST [total: 250]\n"
        "COMPLETED_LIST [completed: 237, pending: 13]\n"
        "DEADLINE [March 31, 2025]\n"
        "PENALTY [non-compliance fine: $10K per employee]\n\n"
        "What is the compliance risk if pending employees don't complete training?",
        {"expected_entities": ["TRAINING_SEC", "EMPLOYEE_LIST", "COMPLETED_LIST", "DEADLINE", "PENALTY"],
         "expected_relationships": ["13 employees pending", "penalty $10K per employee"],
         "expected_citations": ["COMPLETED_LIST", "DEADLINE", "PENALTY"]},
        "medium"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "PRODUCT_ALPHA [launched: Jan 2025, category: SaaS]\n"
        "REVENUE_ALPHA [Jan: $50K, Feb: $85K, Mar: $120K]\n"
        "COST_ALPHA [CAC: $500, LTV: $3000, Churn: 5%]\n"
        "BENCHMARK [SaaS median: CAC $800, LTV $2500, Churn 7%]\n\n"
        "How is Product Alpha performing vs industry benchmarks?",
        {"expected_entities": ["PRODUCT_ALPHA", "REVENUE_ALPHA", "COST_ALPHA", "BENCHMARK"],
         "expected_relationships": ["Alpha CAC below benchmark", "Alpha LTV above benchmark"],
         "expected_citations": ["COST_ALPHA", "BENCHMARK"]},
        "medium"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "PATENT_X [Smart Lock, granted 2023, Owner: SecureTech]\n"
        "PRODUCT_Y [ProLock, launched 2024, Maker: RivalCo]\n"
        "CLAIM_IP [SecureTech vs RivalCo, patent infringement]\n"
        "PATENT_X --[covers]--> biometric_unlock_method\n"
        "PRODUCT_Y --[uses]--> biometric_unlock_method\n\n"
        "Does SecureTech have a valid infringement claim?",
        {"expected_entities": ["PATENT_X", "PRODUCT_Y", "CLAIM_IP", "SecureTech", "RivalCo"],
         "expected_relationships": ["PATENT_X covers biometric method", "PRODUCT_Y uses same method"],
         "expected_citations": ["PATENT_X", "PRODUCT_Y", "CLAIM_IP"]},
        "hard"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "CONTRACT_MSA [Master Agreement, effective 2023, auto-renew]\n"
        "AMENDMENT_1 [price increase 5%, effective Apr 2025]\n"
        "AMENDMENT_1 --[modifies]--> CONTRACT_MSA\n"
        "NOTICE_REQ [60 days written notice for changes]\n"
        "AMENDMENT_1 --[notice_sent]--> Feb 1, 2025\n\n"
        "Is the price amendment properly executed?",
        {"expected_entities": ["CONTRACT_MSA", "AMENDMENT_1", "NOTICE_REQ"],
         "expected_relationships": ["AMENDMENT_1 modifies CONTRACT_MSA", "notice sent Feb 1"],
         "expected_citations": ["CONTRACT_MSA", "AMENDMENT_1"]},
        "medium"))

    items.append(_ex("kg", "cross_document_kg",
        "[KG CONTEXT]\n"
        "APPLICANT_1 [Jane Kim, applied: March 2025]\n"
        "POSITION_1 [Data Engineer, dept: Analytics]\n"
        "INTERVIEW_1 [Panel: 3 interviewers, avg score: 4.2/5]\n"
        "REFERENCE_CHECK [2 of 3 completed, positive]\n"
        "BACKGROUND [clear, no issues]\n"
        "OFFER_TEMPLATE [salary range: $110K-$140K]\n"
        "APPLICANT_1 --[asks]--> $135K\n\n"
        "Is Jane Kim ready for an offer and at what salary?",
        {"expected_entities": ["APPLICANT_1", "POSITION_1", "INTERVIEW_1", "REFERENCE_CHECK", "OFFER_TEMPLATE"],
         "expected_relationships": ["Jane Kim scored 4.2", "asks $135K within range"],
         "expected_citations": ["APPLICANT_1", "INTERVIEW_1", "OFFER_TEMPLATE"]},
        "medium"))

    return items


# ===================================================================
# Track 6 -- Visualization  (50 examples)
# ===================================================================

def _build_visualization() -> List[dict]:
    items: List[dict] = []

    # -- bar_chart (8) ---
    for prompt, labels, values, ctype, diff in [
        ("Show quarterly revenue.\n\n[EVIDENCE]\nQ1: $1.2M, Q2: $1.5M, Q3: $1.8M, Q4: $2.1M.",
         ["Q1","Q2","Q3","Q4"], [1.2,1.5,1.8,2.1], "bar", "easy"),
        ("Show department headcount.\n\n[EVIDENCE]\nEngineering: 85, Sales: 42, Marketing: 28, HR: 15, Finance: 22.",
         ["Engineering","Sales","Marketing","HR","Finance"], [85,42,28,15,22], "bar", "easy"),
        ("Visualize monthly support tickets.\n\n[EVIDENCE]\nJan: 120, Feb: 145, Mar: 98, Apr: 167.",
         ["Jan","Feb","Mar","Apr"], [120,145,98,167], "bar", "easy"),
        ("Compare sales by region.\n\n[EVIDENCE]\nNorth: $3.2M, South: $2.8M, East: $4.1M, West: $3.5M.",
         ["North","South","East","West"], [3.2,2.8,4.1,3.5], "bar", "easy"),
        ("Compare Q1 and Q2 expenses by category.\n\n[EVIDENCE]\nSalaries Q1: $800K, Q2: $820K. Cloud Q1: $120K, Q2: $150K.",
         ["Salaries","Cloud"], [800,120], "grouped_bar", "medium"),
        ("Show project cost breakdown by phase.\n\n[EVIDENCE]\nAlpha: Design $50K, Dev $120K. Beta: Design $40K, Dev $90K.",
         ["Alpha","Beta"], [50,40], "stacked_bar", "medium"),
        ("Rank top 5 clients by value.\n\n[EVIDENCE]\nAcme: $2.1M, Globex: $1.8M, Initech: $1.5M, Umbrella: $1.2M, Stark: $0.9M.",
         ["Acme","Globex","Initech","Umbrella","Stark"], [2.1,1.8,1.5,1.2,0.9], "horizontal_bar", "easy"),
        ("Show website traffic by channel.\n\n[EVIDENCE]\nOrganic: 45000, Paid: 22000, Social: 18000, Direct: 8000.",
         ["Organic","Paid","Social","Direct"], [45000,22000,18000,8000], "bar", "easy"),
    ]:
        items.append(_ex("visualization", "bar_chart", prompt,
            {"expects_chart": True, "expected_chart_type": ctype,
             "expected_labels": labels, "expected_values": values}, diff))

    # -- line_chart (7) ---
    for prompt, labels, values, ctype, diff in [
        ("Show monthly revenue trend.\n\n[EVIDENCE]\nJan $400K, Feb $420K, Mar $390K, Apr $450K, May $480K, Jun $510K.",
         ["Jan","Feb","Mar","Apr","May","Jun"], [400,420,390,450,480,510], "line", "easy"),
        ("Show cumulative spending.\n\n[EVIDENCE]\nJan $50K, Feb $110K, Mar $175K, Apr $245K, May $320K.",
         ["Jan","Feb","Mar","Apr","May"], [50,110,175,245,320], "area", "easy"),
        ("Compare revenue and costs over quarters.\n\n[EVIDENCE]\nQ1 rev $1.2M cost $0.9M. Q2 rev $1.5M cost $1.0M. Q3 rev $1.8M cost $1.1M.",
         ["Q1","Q2","Q3"], [1.2,1.5,1.8], "multi_line", "medium"),
        ("Show daily active users.\n\n[EVIDENCE]\nMon: 12500, Tue: 13200, Wed: 14100, Thu: 13800, Fri: 11900.",
         ["Mon","Tue","Wed","Thu","Fri"], [12500,13200,14100,13800,11900], "line", "easy"),
        ("Show server response time trends.\n\n[EVIDENCE]\nWk1: 120ms, Wk2: 135ms, Wk3: 180ms, Wk4: 220ms.",
         ["Wk1","Wk2","Wk3","Wk4"], [120,135,180,220], "line", "easy"),
        ("Show subscription growth.\n\n[EVIDENCE]\nJan: 1200, Feb: 1350, Mar: 1480, Apr: 1620, May: 1790.",
         ["Jan","Feb","Mar","Apr","May"], [1200,1350,1480,1620,1790], "line", "easy"),
        ("Track bug count by sprint.\n\n[EVIDENCE]\nS1: 45, S2: 38, S3: 52, S4: 29, S5: 22.",
         ["S1","S2","S3","S4","S5"], [45,38,52,29,22], "line", "easy"),
    ]:
        items.append(_ex("visualization", "line_chart", prompt,
            {"expects_chart": True, "expected_chart_type": ctype,
             "expected_labels": labels, "expected_values": values}, diff))

    # -- pie_donut (7) ---
    for prompt, labels, values, ctype, diff in [
        ("What is the expense breakdown?\n\n[EVIDENCE]\nSalaries: 55%, Cloud: 20%, Marketing: 12%, Office: 8%, Travel: 5%.",
         ["Salaries","Cloud","Marketing","Office","Travel"], [55,20,12,8,5], "donut", "easy"),
        ("Show document types distribution.\n\n[EVIDENCE]\nPDFs: 340, Word: 180, Spreadsheets: 95, Images: 45.",
         ["PDFs","Word","Spreadsheets","Images"], [340,180,95,45], "donut", "easy"),
        ("Show market share.\n\n[EVIDENCE]\nUs: 22%, CompA: 35%, CompB: 18%, CompC: 15%, Others: 10%.",
         ["Us","CompA","CompB","CompC","Others"], [22,35,18,15,10], "pie", "easy"),
        ("Revenue by product line.\n\n[EVIDENCE]\nEnterprise: 45%, SMB: 30%, Consumer: 15%, Services: 10%.",
         ["Enterprise","SMB","Consumer","Services"], [45,30,15,10], "donut", "easy"),
        ("Time allocation.\n\n[EVIDENCE]\nCoding: 40%, Meetings: 25%, Review: 15%, Docs: 10%, Admin: 10%.",
         ["Coding","Meetings","Review","Docs","Admin"], [40,25,15,10,10], "pie", "easy"),
        ("Customer segments.\n\n[EVIDENCE]\nFinance: 35%, Health: 25%, Retail: 20%, Tech: 15%, Other: 5%.",
         ["Finance","Health","Retail","Tech","Other"], [35,25,20,15,5], "donut", "easy"),
        ("Incident types Q1.\n\n[EVIDENCE]\nNetwork: 30, Security: 22, Hardware: 18, Software: 45.",
         ["Network","Security","Hardware","Software"], [30,22,18,45], "pie", "easy"),
    ]:
        items.append(_ex("visualization", "pie_donut", prompt,
            {"expects_chart": True, "expected_chart_type": ctype,
             "expected_labels": labels, "expected_values": values}, diff))

    # -- specialized_chart (8) ---
    for prompt, labels, values, ctype, diff in [
        ("Rate vendor across criteria.\n\n[EVIDENCE]\nReliability: 9, Cost: 7, Support: 8, Innovation: 6, Compliance: 9.",
         ["Reliability","Cost","Support","Innovation","Compliance"], [9,7,8,6,9], "radar", "medium"),
        ("Show P&L waterfall.\n\n[EVIDENCE]\nRevenue: $5.0M. COGS: -$2.0M. Gross: $3.0M. Opex: -$1.5M. Net: $1.1M.",
         ["Revenue","COGS","Gross","Opex","Net"], [5.0,-2.0,3.0,-1.5,1.1], "waterfall", "medium"),
        ("Budget allocation by dept.\n\n[EVIDENCE]\nEngineering: $4.2M, Sales: $2.8M, Marketing: $1.5M, HR: $0.8M.",
         ["Engineering","Sales","Marketing","HR"], [4.2,2.8,1.5,0.8], "treemap", "medium"),
        ("SLA compliance rate.\n\n[EVIDENCE]\nCurrent: 97.3%. Target: 99.5%.",
         ["SLA Compliance"], [97.3], "gauge", "medium"),
        ("Plot deal size vs time.\n\n[EVIDENCE]\nDeal A: $50K 15d. Deal B: $120K 45d. Deal C: $30K 8d.",
         ["A","B","C"], [50,120,30], "scatter", "medium"),
        ("Show sales funnel.\n\n[EVIDENCE]\nLeads: 1000, Qualified: 400, Proposal: 150, Negotiation: 80, Closed: 35.",
         ["Leads","Qualified","Proposal","Negotiation","Closed"], [1000,400,150,80,35], "funnel", "medium"),
        ("Employee satisfaction heatmap.\n\n[EVIDENCE]\nEng/Culture: 4.5, Eng/Comp: 3.8. Sales/Culture: 3.9, Sales/Comp: 4.1.",
         ["Engineering","Sales"], [4.5,3.8,3.9,4.1], "heatmap", "hard"),
        ("CPU utilization gauge.\n\n[EVIDENCE]\nCurrent: 72%. Warning: 80%. Critical: 95%.",
         ["CPU Usage"], [72], "gauge", "easy"),
    ]:
        items.append(_ex("visualization", "specialized_chart", prompt,
            {"expects_chart": True, "expected_chart_type": ctype,
             "expected_labels": labels, "expected_values": values}, diff))

    # -- no_chart (10) ---
    for prompt in [
        "When does the contract expire?\n\n[EVIDENCE]\nContract effective January 1, 2025, term 24 months.",
        "Who is the landlord?\n\n[EVIDENCE]\nLandlord is Riverside Properties LLC, 500 Harbor Drive.",
        "What is the payment amount?\n\n[EVIDENCE]\nPayment: $15,000 to Johnson & Associates.",
        "What is the warranty period?\n\n[EVIDENCE]\nWarranty: 36 months from delivery.",
        "Who signed the agreement?\n\n[EVIDENCE]\nSigned by Maria Torres and James Park on March 15.",
        "What are the NDA terms?\n\n[EVIDENCE]\nConfidentiality: 5 years. Mutual NDA.",
        "Summarize the meeting decisions.\n\n[EVIDENCE]\nApproved hiring plan. Deferred expansion.",
        "What is the shipping address?\n\n[EVIDENCE]\nShip to: 2100 Innovation Dr, Austin TX.",
        "What insurance coverage applies?\n\n[EVIDENCE]\nGeneral liability up to $5M, $25K deductible.",
        "What is the employee's start date?\n\n[EVIDENCE]\nSarah Chen starts April 14, 2025.",
    ]:
        items.append(_ex("visualization", "no_chart", prompt,
            {"expects_chart": False, "expected_chart_type": "",
             "expected_labels": [], "expected_values": []}, "easy"))

    # -- flow_analysis (10) ---
    for prompt in [
        "Describe the invoice approval process.\n\n[EVIDENCE]\nInvoice submitted. Finance validates. Manager approves <$10K. Director >$10K. Payment Net 30.",
        "How does document onboarding work?\n\n[EVIDENCE]\nUpload. Extract text. Chunk. Embed. Store vectors.",
        "What is the offboarding procedure?\n\n[EVIDENCE]\nManager initiates. HR exit interview. IT revokes. Final paycheck. Equipment returned.",
        "Explain the CI/CD pipeline.\n\n[EVIDENCE]\nPush code. Unit tests. Integration tests. Security scan. Deploy staging. Deploy prod.",
        "How does leave request work?\n\n[EVIDENCE]\nSubmit in portal. Manager reviews. >5 days needs VP. HR updates calendar.",
        "Describe support escalation.\n\n[EVIDENCE]\nTier 1 receives. 2hrs to Tier 2. 4hrs to Tier 3. VP for P0.",
        "What is the procurement process?\n\n[EVIDENCE]\nPR submitted. 3 quotes >$5K. Committee >$50K. PO issued. Goods received. Payment.",
        "How does backup work?\n\n[EVIDENCE]\nIncremental 6hrs. Full daily 2AM. Weekly offsite. Monthly cold storage.",
        "Explain incident response.\n\n[EVIDENCE]\nAlert. On-call ack 15min. Severity assessed. War room P0/P1. Fix deployed. Post-mortem 48hrs.",
        "How does vendor onboarding work?\n\n[EVIDENCE]\nApplication. Legal review. Credit check. Compliance verify. Add to system. First PO.",
    ]:
        items.append(_ex("visualization", "flow_analysis", prompt,
            {"expects_chart": False, "expected_chart_type": "",
             "expected_labels": [], "expected_values": []}, "easy"))

    return items


# ===================================================================
# Public API
# ===================================================================

_ALL_BUILDERS = {
    "excel_csv": _build_excel_csv,
    "layout": _build_layout,
    "ocr_vision": _build_ocr_vision,
    "reasoning": _build_reasoning,
    "kg": _build_kg,
    "visualization": _build_visualization,
}

_CACHE: Optional[Dict[str, List[dict]]] = None


def _ensure_cache() -> Dict[str, List[dict]]:
    global _CACHE
    if _CACHE is None:
        _CACHE = {name: builder() for name, builder in _ALL_BUILDERS.items()}
    return _CACHE


def get_test_bank(track: Optional[str] = None) -> List[dict]:
    """Return frozen evaluation examples.

    Args:
        track: If provided, return only examples for that track.
               Valid: excel_csv, layout, ocr_vision, reasoning, kg, visualization.

    Returns:
        List of dicts with keys: track, category, prompt, reference, difficulty.
    """
    cache = _ensure_cache()
    if track is not None:
        if track not in cache:
            raise ValueError(f"Unknown track {track!r}. Valid: {sorted(cache.keys())}")
        return list(cache[track])
    result: List[dict] = []
    for name in sorted(cache.keys()):
        result.extend(cache[name])
    return result
