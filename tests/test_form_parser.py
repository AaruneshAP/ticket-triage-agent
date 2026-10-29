"""Unit tests for issue form parser and target leakage prevention."""

from triage.parse.form_parser import (
    clean_text_for_label_prediction,
    extract_ha_version_parts,
    parse_form_sections,
    parse_issue,
    sanitize_field_value,
)

SAMPLE_STANDARD_BODY = """
### The problem

I have just started using the integration after updating my system.
The integration throws a 500 error when fetching sensor readings.

### What version of Home Assistant Core has the issue?

2026.10.2

### What was the last working version of Home Assistant Core?

_No response_

### What type of installation are you running?

Home Assistant OS

### Integration causing the issue

anglian_water

### Link to integration documentation on our website

https://www.home-assistant.io/integrations/anglian_water

### Anything in the logs that might be useful for us?

```txt
Traceback (most recent call last):
  File "/usr/src/homeassistant/components/anglian_water/sensor.py", line 42, in async_update
    data = await client.get_readings()
Exception: Connection timed out
```

### Additional information

Related to issue #18420.
"""

SAMPLE_HANDWRITTEN_BODY = """
Hello, my zigbee devices stopped reporting battery levels after the recent update.
Here is the error log from zigpy:
Traceback (most recent call last):
  File "zigpy/device.py", line 12, in handle_message
    raise ValueError("Invalid payload")
Please help! See also #12399.
"""


def test_sanitize_field_value():
    assert sanitize_field_value(None) is None
    assert sanitize_field_value("") is None
    assert sanitize_field_value("  _No response_  ") is None
    assert sanitize_field_value("None") is None
    assert sanitize_field_value("N/A") is None
    assert sanitize_field_value("2026.10.2") == "2026.10.2"


def test_parse_form_sections_standard():
    sections = parse_form_sections(SAMPLE_STANDARD_BODY)
    assert sections["The problem"] is not None
    assert "throws a 500 error" in sections["The problem"]
    assert sections["What version of Home Assistant Core has the issue?"] == "2026.10.2"
    assert sections["What was the last working version of Home Assistant Core?"] is None
    assert sections["What type of installation are you running?"] == "Home Assistant OS"
    assert sections["Integration causing the issue"] == "anglian_water"


def test_clean_text_for_label_prediction():
    cleaned = clean_text_for_label_prediction(SAMPLE_STANDARD_BODY)
    assert "### Integration causing the issue" not in cleaned
    assert "Link to integration documentation" not in cleaned
    assert "The problem" in cleaned
    assert "What version of Home Assistant Core" in cleaned


def test_parse_issue_standard():
    parsed = parse_issue(SAMPLE_STANDARD_BODY, title="Anglian Water sensor 500 error")
    assert parsed["is_template"] is True
    assert parsed["ha_version_year"] == 2026
    assert parsed["ha_version_month"] == 10
    assert parsed["ha_version_patch"] == 2
    assert parsed["has_last_working_version"] is False
    assert parsed["has_traceback"] is True
    assert parsed["has_logs"] is True
    assert parsed["n_code_blocks"] >= 1
    assert parsed["refers_to_issue"] is True
    assert 18420 in parsed["referenced_issues"]
    assert parsed["integration_in_form"] == "anglian_water"


def test_parse_issue_handwritten():
    parsed = parse_issue(SAMPLE_HANDWRITTEN_BODY, title="Zigbee battery bug")
    assert parsed["is_template"] is False
    assert parsed["problem"] is not None
    assert "zigbee devices stopped reporting" in parsed["problem"]
    assert parsed["has_traceback"] is True
    assert parsed["refers_to_issue"] is True
    assert 12399 in parsed["referenced_issues"]


def test_extract_ha_version_parts():
    y, m, p = extract_ha_version_parts("2026.10.2")
    assert (y, m, p) == (2026, 10, 2)

    y, m, p = extract_ha_version_parts("core-2025.12.0b1")
    assert (y, m, p) == (2025, 12, 0)

    y, m, p = extract_ha_version_parts("invalid version")
    assert (y, m, p) == (None, None, None)
