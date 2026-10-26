# Warehouse Spike Report: Databricks Free Edition vs. Alternatives

## 1. Executive Summary
This spike evaluated whether **Databricks Free Edition (Community Edition / Serverless Starter Warehouse)** is viable as the analytics data warehouse for the Ticket Triage Agent. Tests assessed workspace provisioning, Python connectivity (`databricks-sql-connector`), ACID CRUD on Delta Lake tables, dbt model transformations and incremental loads (`dbt-databricks`), and Power BI connectivity.

**Key Finding**: Databricks Free Edition with Serverless Starter Warehouse **successfully executed all tests (Tests 1–6) with 100% pass rate**. In modern workspaces, **Unity Catalog (`catalog: workspace`) is required**, as legacy Hive Metastore access is disabled (`UC_HIVE_METASTORE_DISABLED_EXCEPTION`).

---

## 2. Test Execution Matrix

| # | Test Item | Target / Scenario | Status | Measured Duration | Exact Output / Result Details |
|---|---|---|---|---|---|
| **1** | **Sign-up** | Create Free Edition workspace; check if credit card required | **PASS** | - | Workspace created. **No credit card asked.** |
| **2** | **Access & Auth** | Python connection via `databricks-sql-connector` using PAT | **PASS** | **8.51s** | Connected to SQL warehouse endpoint. User: `workspace-user`. |
| **3** | **Write & CRUD** | Load 1,000 rows, append 50 rows, update 1 row, delete 1 row | **PASS** | **35.67s** (total) | Delta table `workspace.default.spike_issues`: 1,000 loaded (18.04s), 50 appended (1.63s), 1 updated (4.51s), 1 deleted (1.95s). Verified final count: 1,049 rows. |
| **4** | **dbt Pipeline** | `dbt-databricks` build, schema tests, incremental model | **PASS** | Run 1: **38.64s**<br>Run 2: **44.51s** | Built `stg_spike_issues` (view), `inc_spike_issues` (incremental table). All 4 tests passed (`not_null`, `unique`). Incremental logic verified. |
| **5** | **Power BI** | Direct / Import connection to `spike_issues` table | **PASS** | - | Power BI Desktop connected via native Databricks connector using PAT. Table `spike_issues` loaded without Microsoft sign-in requirement. |
| **6** | **Quota & Latency**| Measure startup latency, query times, and resource caps | **PASS** | Initial connect: 8.5s | No compute quota or throttling errors encountered. Auto-suspending serverless warehouse. |

---

## 3. Detailed Test Logs & Execution Artifacts

### Test 1: Sign-up
- **Platform**: Databricks Free Edition / Serverless Starter.
- **Credit Card Required**: **No**. Full free tier access granted without payment details.

### Test 2: Access & Python Connectivity
- **Host**: `<databricks-workspace-host>`
- **HTTP Path**: `/sql/1.0/warehouses/<warehouse-id>`
- **Connector**: `databricks-sql-connector==4.4.0`
- **Measured Latency**: 8.51s to establish initial session and query `SELECT current_timestamp(), current_user(), current_schema()`.

### Test 3: Write & Delta Lake CRUD Operations
- **Table**: `workspace.default.spike_issues`
- **Dataset**: Real Home Assistant issue records (`ticket_number`, `title`, `state`, `created_at`, `integration`).
- **Operation Timings**:
  - `CREATE TABLE & TRUNCATE`: 5.50s
  - `INSERT 1,000 rows` (batch chunks): 18.04s
  - `APPEND 50 rows`: 1.63s
  - `UPDATE 1 row` (`ticket_number = 188139`): 4.51s
  - `DELETE 1 row` (`ticket_number = 188138`): 1.95s
  - `SELECT COUNT(*)` verification: 1,049 rows (exact match).

### Test 4: dbt-databricks Build
- **Adapter**: `dbt-databricks==1.12.8` with `dbt-core==1.12.5`
- **Catalog**: `workspace` (Unity Catalog).
- **Encountered & Resolved Note**: Omitting `catalog` triggered `[UC_HIVE_METASTORE_DISABLED_EXCEPTION]`. Setting `catalog: workspace` in `profiles.yml` resolved this immediately.
- **Run 1 (Initial Build + Tests)**: 38.64s total.
  - `stg_spike_issues` (view): Created in 3.01s.
  - `not_null_stg_spike_issues_ticket_number`: PASS (3.06s).
  - `unique_stg_spike_issues_ticket_number`: PASS (3.50s).
  - `inc_spike_issues` (incremental table initial load): Created in 6.39s.
  - `not_null_inc_spike_issues_ticket_number`: PASS (2.74s).
  - `unique_inc_spike_issues_ticket_number`: PASS (2.75s).
  - **Summary**: 6/6 passed, 0 errors, 0 warnings.
- **Run 2 (Incremental Build)**: 44.51s total.
  - Incremental filter executed against `workspace.default.inc_spike_issues`.
  - All tests re-verified successfully.

### Test 5: Power BI Desktop Connectivity
- **Connector**: Native Power BI Databricks Connector.
- **Authentication**: Personal Access Token (PAT).
- **Mode**: Import / DirectQuery verified. Data model loaded `spike_issues` successfully on local laptop without requiring Microsoft organizational sign-in.

### Test 6: Quotas, Limits & Cold Starts
- **Compute Type**: Serverless Starter Warehouse (`02a8f454a4c2acfa`).
- **Cold Start Time**: ~8.5 seconds.
- **Resource Limits**: No quota ceiling or query concurrency throttling was encountered on ~1,000 row workloads.
- **Delta Lake Capabilities**: Native support for ACID merges, updates, and deletes without requiring manual vacuuming or locking.

---

## 4. Platform Comparison Summary

| Metric / Dimension | Databricks Free Edition | BigQuery Sandbox | Snowflake 30-Day Trial |
|---|---|---|---|
| **Credit Card Required** | **No** (Community/Free tier) | **No** | **Yes** (or mobile verification depending on region) |
| **Trial Time Limit** | **No expiration** (ongoing) | **No expiration** | **30 days** ($400 credit window) |
| **Table Expiration** | **None** | Tables expire after 60 days | None during trial |
| **Data Formats** | Delta Lake (Parquet + Transaction Log) | Native Capacitor / BigQuery | Snowflake proprietary micro-partitions |
| **dbt Support** | Excellent (`dbt-databricks`) | Excellent (`dbt-bigquery`) | Excellent (`dbt-snowflake`) |
| **Incremental Models** | Native Delta Merge | Native Merge / Replace | Native Merge |
| **Power BI Integration** | Direct connector with PAT (no MS login needed) | Google OAuth / Service Account | Snowflake ODBC / Direct connector |
| **ACID Updates/Deletes** | Full support | Quota limits on DML (1,500/day/table) | Full support |

---

## 5. Final Recommendation
**Databricks Free Edition is highly recommended and fully validated as a data warehouse for this project.**
1. **Zero Financial / Expiration Friction**: No credit card required at sign-up, and no 30-day trial expiration risk like Snowflake or 60-day table drop policy like BigQuery Sandbox.
2. **First-Class dbt & Delta Lake Integration**: `dbt-databricks` executed views, incremental models, and schema tests cleanly within Unity Catalog.
3. **Power BI Ready**: Can be connected immediately using local Power BI Desktop via PAT without organizational Microsoft 365 licensing.
