"""Capture real screenshots of local services using headless Playwright."""

import os
import time

from playwright.sync_api import sync_playwright

os.makedirs("docs/images", exist_ok=True)


def capture_airflow():
    """Capture real screenshot of Airflow DAG graph with a successful run."""
    print("Capturing Airflow DAG graph at http://127.0.0.1:8080/dags/triage_hourly/grid?tab=graph...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.goto("http://127.0.0.1:8080/login/", wait_until="networkidle")
        if page.locator('input[name="username"]').count() > 0:
            page.fill('input[name="username"]', "admin")
            page.fill('input[name="password"]', "admin")
            page.click('input[type="submit"]')
            page.wait_for_load_state("networkidle")
        page.goto(
            "http://127.0.0.1:8080/dags/triage_hourly/grid?dag_run_id=scheduled__2026-10-29T11%3A00%3A00%2B00%3A00&tab=graph",
            wait_until="networkidle",
        )
        time.sleep(5)
        page.screenshot(path="docs/images/airflow_dag_graph.png")
        browser.close()
    print("[OK] Saved docs/images/airflow_dag_graph.png")


def capture_n8n():
    """Capture real screenshot of n8n workflow canvas."""
    print("Capturing n8n workflow canvas at http://127.0.0.1:5678/workflow/8ll8jQrLAwRUhqJ4...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.goto("http://127.0.0.1:5678/workflow/8ll8jQrLAwRUhqJ4", wait_until="networkidle")
        time.sleep(3)
        for _ in range(3):
            page.keyboard.press("Escape")
            time.sleep(1)
        if page.locator('input[type="email"]').count() > 0:
            page.fill('input[type="email"]', "admin@example.com")
            page.fill('input[type="password"]', "Password123!")
            page.locator('button[type="submit"], button:has-text("Sign in")').click()
            page.wait_for_load_state("networkidle")
            page.goto("http://127.0.0.1:5678/workflow/8ll8jQrLAwRUhqJ4", wait_until="networkidle")
            time.sleep(3)
            page.keyboard.press("Escape")
            time.sleep(1)
        time.sleep(2)
        page.screenshot(path="docs/images/n8n_workflow_canvas.png")
        browser.close()
    print("[OK] Saved docs/images/n8n_workflow_canvas.png")


def capture_streamlit():
    print("Capturing Streamlit Review App at http://127.0.0.1:8501...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto("http://127.0.0.1:8501", wait_until="networkidle")
        time.sleep(10)
        page.screenshot(path="docs/images/streamlit_review_app.png")
        browser.close()
    print("[OK] Saved docs/images/streamlit_review_app.png")


def capture_mlflow():
    print("Capturing MLflow Experiments at http://127.0.0.1:5000...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto("http://127.0.0.1:5000", wait_until="networkidle")
        time.sleep(5)
        page.screenshot(path="docs/images/mlflow_experiments.png")
        browser.close()
    print("[OK] Saved docs/images/mlflow_experiments.png")


if __name__ == "__main__":
    capture_airflow()
    capture_n8n()
