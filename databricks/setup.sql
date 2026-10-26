-- ==============================================================================
-- Databricks Setup Script for Ticket Triage Agent
-- Catalog: workspace
-- Schemas: raw, bronze, silver, gold
-- ==============================================================================

CREATE SCHEMA IF NOT EXISTS workspace.raw
COMMENT 'Raw landing zone for GitHub issues and comments loaded from interim parquet';

CREATE SCHEMA IF NOT EXISTS workspace.bronze
COMMENT 'Bronze layer managed by dbt referencing raw landing tables';

CREATE SCHEMA IF NOT EXISTS workspace.silver
COMMENT 'Cleaned, typed, parsed, and normalized staging views/tables';

CREATE SCHEMA IF NOT EXISTS workspace.gold
COMMENT 'Curated analytical marts, feature store (fct_tickets), and monitoring marts';
