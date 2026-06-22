"""Real-browser (Selenium + headless chromium) steps: prove the dashboard
actually RENDERS the data, not just that the API returns it."""
import time
from behave import given, when, then

# Optional: only the @browser scenarios need selenium (run under the chromium
# nix-shell). Keep the module importable so the non-browser suite runs without it.
try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError:
    By = WebDriverWait = None


def _body_text(driver):
    return driver.find_element(By.TAG_NAME, "body").text


def _wait_text(context, text, timeout=40):
    WebDriverWait(context.driver, timeout).until(
        lambda d: text.lower() in _body_text(d).lower(),
        message=f"timed out waiting for {text!r}")


@given('I open the dashboard in a browser')
def step_open(context):
    context.driver.get(context.base_url + "/dashboard")
    # NO_AUTH -> goes straight to the app shell (sidebar nav with the tabs).
    _wait_text(context, "Analytics", timeout=40)
    _wait_text(context, "Activity", timeout=40)


@when('I click the "{tab}" tab')
def step_click_tab(context, tab):
    # Tab buttons are <button class='tab'><span class='navIcon'>X</span>Label</button>,
    # so the visible text is "<icon> Label" -> match by substring.
    deadline = time.time() + 20
    while time.time() < deadline:
        for el in context.driver.find_elements(By.CSS_SELECTOR, ".tab"):
            try:
                if tab.lower() in el.text.strip().lower() and el.is_displayed():
                    context.driver.execute_script("arguments[0].click();", el)
                    time.sleep(0.8)  # let the tab's loader fetch + render
                    return
            except Exception:
                continue
        time.sleep(0.3)
    raise AssertionError(f'tab {tab!r} not found/clickable')


@then('I see "{text}" rendered')
def step_see(context, text):
    _wait_text(context, text)


@when('I expand the first Activity row')
def step_expand_row(context):
    rows = context.driver.find_elements(By.CSS_SELECTOR, "#recent .actRow")
    assert rows, "no Activity rows to expand"
    context.driver.execute_script("arguments[0].click();", rows[0])
    time.sleep(0.5)
    open_details = context.driver.find_elements(
        By.CSS_SELECTOR, "#recent .actDetail:not(.hidden)")
    assert open_details, "row did not expand on click"


@when('I wait {seconds:d} seconds')
def step_wait(context, seconds):
    time.sleep(seconds)


@then('an Activity row is still expanded')
def step_still_expanded(context):
    open_details = context.driver.find_elements(
        By.CSS_SELECTOR, "#recent .actDetail:not(.hidden)")
    assert open_details, ("an expanded Activity row collapsed by itself "
                          "(auto-refresh bug regressed)")


@then('I do NOT see "{text}" rendered')
def step_not_see(context, text):
    time.sleep(1.0)
    assert text.lower() not in _body_text(context.driver).lower(), \
        f'unexpectedly saw {text!r} on screen'
